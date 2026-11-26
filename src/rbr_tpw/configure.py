"""Configure and enable a logger: clock, battery counter, schedule, erase, enable. RBRsolo (fwtypes 9 and 0),
RBRduet (102), RBRconcerto (103) and the Gen3 loggers (104, the RBRconcerto3; _apply_gen3).

The write sequence mirrors what Ruskin 2.26.1 sends to these loggers (from
~/Ruskin/logs/ruskin_serial.log, 2026-09-25): `lock OFF = <session key>`,
`stop`, `now = <UTC>`, `starttime = `, `endtime = `, `sampling mode = , period = `,
`verify`, `permit memclear`, `memclear`, `enable`, `lock on`.
Erasing is only allowed after the same session's download has been saved.

fwtype 0 (RBRsolo firmware 1.110): Ruskin's logs (2026-09-27 check, 13 loggers) show the same commands and
replies, except that there is no energy counter (no `powerstatus remaining =` write) and Ruskin never wrote
`endtime =` to one, so that write is skipped when the logger already holds the wanted end time.

RBRduet (102, fw 3.220) and RBRconcerto (103, fw 1.460): Ruskin 2.26.1's serial logs hold 5 duet enables
(2026-09-13..22) and 10 concerto enables (2026-09-11..16). Ruskin writes only what changes, and erases and enables
in one command:
`lock OFF = <key>`, `stop` (if running), `permit settings` + `settings offsetfromutc = 0.0`, `now = <UTC>`,
[`starttime = `], [`sampling period = N`], [`twistactivation state = on|off`, concerto], `permit settings` +
`settings ..., offsetfromutc = 0` (its rewrite of every setting), `verify` (E0402: memory not empty),
`enable erasememory = true` (duet; `enable erasememory = true, memformat = RAWBIN00` on the concerto) ->
`E0401 estimated memory usage exceeds capacity, enable = logging`, or `enable = gated` on a twist-gated concerto
that is twisted off; `lock on`.

A clock write leaves the UTC offset `unknown` (RBRduet SN082492, 2026-11-26: +0.00 before `now = X`, `unknown`
after it, also when `settings offsetfromutc = 0.0` came just before it, as Ruskin sends it). Ruskin's loggers read
+0.00 afterwards only because its later rewrite of every setting includes `offsetfromutc = 0`. So here the offset
is written after the clock (`permit settings` + `settings offsetfromutc = 0.0`), whenever it does not read zero,
and read back. The start, end, period and twist writes are made only when they change. Not copied from Ruskin: its
rewrites of the other unchanged settings and of the duet's `powerstatus remaining`, and the `sleep` it sends to a
concerto just before `enable` (the enable saves the settings anyway). An erase without enable (--no-enable) uses
`permit memclear` + `memclear`, as Ruskin's PhysicalL2.eraseMemory does, then checks `meminfo used` = 0.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .link import Link, LinkError, LoggerError, parse_pairs
from .lock import Session, unlock_key  # noqa: F401  (re-exported: the tests import unlock_key from here)
from .solo import DROP_DRAIN_S, NOMINAL_BATTERY_J, measure_clock_skew, memory, power

FAR_FUTURE = "20991231235959"  # Ruskin's "no end time"
PAST = "20000101000000"  # start time in the past = start as soon as enabled

SOLO = (0, 9)
SECTIONED_L2 = (102, 103)  # RBRduet, RBRconcerto: the L2 command set with a sectioned memory header
GEN3 = (104,)  # RBRconcerto3 and the other L3 loggers: the documented L3 command set
CONFIGURABLE = SOLO + SECTIONED_L2 + GEN3
TWIST = (103, 104)  # models with twist activation here
# Ready to deploy after an enable: logging, waiting for its start time, or waiting for its gate (a twist-gated
# RBRconcerto that is twisted off answers `enable = gated` and `status = gated`, Ruskin's logs 2026-09-11..16)
READY_STATUSES = ("logging", "pending", "gated")
ENABLE_TIMEOUT_S = 60.0  # `enable erasememory = true` answered in 0.4-0.6 s (Ruskin's logs); erasing may take longer


class ConfigError(Exception):
    """A configure step failed. `report` holds the steps done so far (e.g. whether memory was erased)."""

    report: dict | None = None


SPIN_S = 0.050  # set_clock spins (does not sleep) for this long before sending the time
IDLE_SAVE_S = 12.0  # settings are saved after 10 s idle (L3.5 ref 1.1.2; assumed, unverified, for L2 loggers)


@dataclass
class DeployConfig:
    set_clock: bool = True
    fresh_battery: bool = False  # reset the energy counter to a new cell's nominal capacity
    # A cell already used elsewhere: the counter is set to (life - used) / life of a new cell's capacity.
    battery_days_used: float | None = None  # days the cell has already powered an instrument
    battery_life_days: float | None = None  # that cell's expected life in that instrument, days
    erase: bool = True
    mode: str = "continuous"
    period_ms: int | None = None  # None = keep the logger's current period
    start: str = "now"  # "now" or ISO-8601 UTC
    end: str = "never"  # "never" or ISO-8601 UTC
    enable: bool = True
    clock_tolerance_s: float = 0.020
    twist_activation: bool | None = None  # RBRconcerto, RBRconcerto3: gate sampling on the twist switch; None: leave it

    @classmethod
    def from_yaml(cls, path: Path) -> DeployConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.pop("thresholds", None)  # read separately (cli.Thresholds)
        data = data.get("deploy", data)
        sched = data.pop("schedule", {}) or {}
        flat = {**data, **{k: v for k, v in sched.items()}}
        if "rate_hz" in flat:
            flat["period_ms"] = rate_period_ms(float(flat.pop("rate_hz")))
        known = {f.name for f in fields(cls)}
        unknown = set(flat) - known
        if unknown:
            raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(**flat)

    def battery_fraction(self) -> float | None:
        """Fraction of a new cell to write to the energy counter; None = leave the counter alone."""
        used, life = self.battery_days_used, self.battery_life_days
        if used is None and life is None:
            return 1.0 if self.fresh_battery else None
        if used is None or life is None:
            raise ConfigError("a used battery needs both battery_days_used and battery_life_days")
        if self.fresh_battery:
            raise ConfigError("fresh_battery and a used battery (battery_days_used) are mutually exclusive")
        if not (life > 0 and 0 <= used < life):
            raise ConfigError(f"used battery: need 0 <= days used ({used:g}) < life ({life:g} days)")
        return (life - used) / life


def rate_period_ms(rate_hz: float) -> int:
    """Sampling period in ms for a rate, rounded half up: 16 Hz -> 63 ms, as the Gen3 `availablefastperiods`
    list has it (Python's round() gives 62)."""
    return int(1000 / rate_hz + 0.5)


def _logger_time(value: str, now_ok: bool) -> str:
    """'now'/'never'/ISO-8601 (UTC) -> YYYYMMDDhhmmss."""
    v = str(value).strip()
    if v.lower() == "now" and now_ok:
        return PAST
    if v.lower() in ("never", "none"):
        return FAR_FUTURE
    t = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ConfigError(f"time {v!r} needs an explicit UTC offset, e.g. 2026-10-01T00:00:00Z")
    return t.astimezone(dt.UTC).strftime("%Y%m%d%H%M%S")


def period_problem(period: int, fwtype: int | None, limit_ms: int | None = None,
                   allowed: list[int] | None = None) -> str | None:
    """Why `period` (ms) will not do for this model, or None. fwtype None: acceptable to some model.

    RBRsolo: Ruskin offers 2 Hz or whole seconds. RBRduet and RBRconcerto: whole seconds, or 1000/n ms rounded for
    a whole rate of n Hz (500, 333, 250, 200, 167, 125, ...), no shorter than the logger's own `userperiodlimit`
    (the duets report 500 ms, the older concertos 167). The 1000/n rule is an assumption from the Gen3 list
    (`availablefastperiods = 500|250|125|63|42|31`, all of that form) and from 250 ms, which Ruskin set on a
    concerto on 2026-09-13; the logger's readback after the write is the check. Gen3 loggers list their sub-second
    periods themselves (`sampling availablefastperiods`, `allowed`): only those (L3 reference 4.1.2)."""
    if period <= 0:
        return "not a positive number of milliseconds"
    solo_ok = period == 500 or (period >= 1000 and period % 1000 == 0)
    if period >= 1000:
        l2_ok = period % 1000 == 0
    else:
        n = round(1000 / period)
        l2_ok = n >= 2 and rate_period_ms(n) == period
    if fwtype in SOLO:
        return None if solo_ok else "use 500 ms (2 Hz) or a whole number of seconds"
    if fwtype is None:
        return None if solo_ok or l2_ok else "use whole seconds, or 1000/n ms for a whole rate of n Hz"
    if not l2_ok:
        return ("use whole seconds, or 1000/n ms (rounded) for a whole rate of n Hz: 500, 333, 250, 200, 167, 125, "
                "...")
    if limit_ms and period < limit_ms:
        return f"shorter than this logger's shortest period, {limit_ms} ms (its userperiodlimit)"
    if allowed is not None and period < 1000 and period not in allowed:
        return f"not one of this logger's sub-second periods ({'|'.join(map(str, allowed)) or 'none'} ms)"
    return None


def validate(cfg: DeployConfig, current_period_ms: int, fwtype: int | None = 9,
             limit_ms: int | None = None, allowed: list[int] | None = None) -> tuple[str, str, int]:
    """(start, end, period) for the logger, or ConfigError. `fwtype` None checks what suits some model (the
    command line, before any logger is seen); `limit_ms` is the logger's userperiodlimit."""
    if cfg.mode != "continuous":
        raise ConfigError(f"sampling mode {cfg.mode!r} is not supported yet (continuous only)")
    period = cfg.period_ms or current_period_ms
    try:
        whole = float(period).is_integer()
    except (TypeError, ValueError):
        whole = False
    if not whole:
        raise ConfigError(f"period {period!r} ms: not a whole number of milliseconds")
    period = int(float(period))  # "500.0" from YAML must not reach the logger as "period = 500.0"
    why = period_problem(period, fwtype, limit_ms, allowed)
    if why:
        raise ConfigError(f"period {period} ms: {why}")
    if cfg.twist_activation is not None and fwtype is not None and fwtype not in TWIST:
        raise ConfigError("twist activation is supported on the RBRconcerto and RBRconcerto3 (fwtypes 103, 104) only")
    start = _logger_time(cfg.start, now_ok=True)
    end = _logger_time(cfg.end, now_ok=False)
    if end <= start:
        raise ConfigError(f"end {end} is not after start {start}")
    if end != FAR_FUTURE and end <= dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S"):
        raise ConfigError(f"end {end} is in the past")
    return start, end, period


def set_clock(link: Link, ntp_offset_s: float | None, tolerance_s: float, attempts: int = 4, log=print,
              pace_s: float | None = None, write: str = "now = {}", clock=None) -> dict:
    """Write `now = <UTC>` so the logger's second boundary lines up with UTC.

    The command is sent `lat` seconds before the target second; the resulting
    skew is measured and `lat` adjusted. Assumes the logger zeroes its
    sub-second counter when the time is written (checked by the re-measurement).
    With ntp_offset_s None (no NTP) the reference is the host clock, and the log says so. `pace_s` paces the
    re-measurement (solo.measure_clock_skew) on loggers that drop polls. `write` is the clock write with `{}` for
    the time (Gen3: `clock datetime = {}, offsetfromutc = 0.0`, as Ruskin sends it) and `clock` the read the skew
    is measured with (default: L2 `now`).
    """
    ref = "UTC" if ntp_offset_s is not None else "host"
    ntp_offset_s = ntp_offset_s or 0.0
    lat = 0.003  # initial guess at host->logger latency (Ruskin sends ~6 ms early)
    history = []
    for attempt in range(1, attempts + 1):
        utc_now = time.time() + ntp_offset_s
        target = math.floor(utc_now) + 2
        send_at_host = target - lat - ntp_offset_s
        # Sleep to within SPIN_S of the send time, then spin: sleep may overshoot by 15 ms or more (median
        # 15 ms for a 2 ms sleep on a GitHub macOS runner, 2026-09-26), which would send the command late.
        while (remaining := send_at_host - time.time()) > SPIN_S:
            time.sleep(remaining - SPIN_S)
        while time.time() < send_at_host:
            pass
        stamp = dt.datetime.fromtimestamp(target, dt.UTC).strftime("%Y%m%d%H%M%S")
        try:
            r = link.command(write.format(stamp))
        except LoggerError:
            raise
        except LinkError as err:  # a logger that drops commands: never re-send a clock write; measure where it is
            if pace_s is None:
                raise
            link.note(f"no reply to the clock write ({err}); measuring the clock to see whether it took")
            link._drain(DROP_DRAIN_S)
            r = None
        if r is not None and stamp not in r:
            raise ConfigError(f"clock set: unexpected reply {r!r}")
        skew = _skew(link, pace_s, clock)
        if not skew.get("n"):
            raise ConfigError("clock set: could not re-measure skew")
        s_utc = skew["skew_vs_host_s"] - ntp_offset_s
        history.append({"attempt": attempt, "lead_s": lat, "skew_vs_utc_s": s_utc,
                        "uncertainty_s": skew["uncertainty_s"], **({"write_reply_lost": True} if r is None else {})})
        log(f"    clock set attempt {attempt}: logger - {ref} = {s_utc * 1e3:+.1f} ms "
            f"(+/- {skew['uncertainty_s'] * 1e3:.1f} ms)")
        if abs(s_utc) <= tolerance_s:
            return {"ok": True, "skew_vs_utc_s": s_utc, "uncertainty_s": skew["uncertainty_s"], "reference": ref,
                    "history": history}
        if r is None:  # the write may never have arrived: the skew says nothing about the lead
            continue
        lat -= s_utc  # logger ahead (s > 0) -> we sent too early -> send later
        # lat may go negative (send just after the target second): the target is ~2 s ahead, so any lead in
        # (-0.5, 0.9) s is still in the future. Stopping at lat < 0 made any first skew above +3 ms fatal.
        if not -0.5 < lat < 0.9:
            break
    return {"ok": False, "skew_vs_utc_s": history[-1]["skew_vs_utc_s"] if history else math.nan, "reference": ref,
            "history": history}


def _skew(link: Link, pace_s: float | None, clock=None) -> dict:
    """Three skew measurements; paced only where asked, by `clock` if given (the solo tests replace this function's
    callee with one that takes only `reps`)."""
    kw = {k: v for k, v in (("pace_s", pace_s), ("clock", clock)) if v is not None}
    return measure_clock_skew(link, reps=3, **kw)


def configure(link: Link, serial: int, cfg: DeployConfig, ntp_offset_s: float | None, log=print,
              timing=nullcontext, fwtype: int = 9, driver=None) -> dict:
    """Apply `cfg`. Caller must already have saved this session's download if cfg.erase.

    `ntp_offset_s` is UTC minus host (None: no NTP, so the clock is set to the host clock).
    `timing(what)` wraps the host-timestamp-sensitive steps (see hostclock.timing_critical).
    `fwtype` is one of CONFIGURABLE (see the module docstring for what differs by model). `driver` (drivers.py),
    when given, paces the skew polls and reads the power back as the offload does."""
    if fwtype not in CONFIGURABLE:
        raise ConfigError(f"configure is implemented for fwtypes {', '.join(map(str, CONFIGURABLE))}, "
                          f"not fwtype {fwtype}")
    pace_s = getattr(driver, "skew_pace_s", None)
    report: dict = {"config": asdict(cfg), "steps": [],
                    "clock_reference": "UTC (host clock corrected by NTP)" if ntp_offset_s is not None else
                    "host clock (no NTP)"}

    # "<name>_sent" is recorded before a command that changes the logger, "<name>" once it is confirmed: after
    # a lost reply, "sent" alone means it may or may not have happened
    def step(name, **kw):
        report["steps"].append({"step": name, **kw})

    cur = link.query("sampling")
    limit = int(cur["userperiodlimit"]) if cur.get("userperiodlimit", "").isdigit() else None
    allowed = None
    if fwtype in GEN3:
        fast = link.query("sampling availablefastperiods").get("availablefastperiods", "none")
        allowed = [] if fast.strip() == "none" else [int(x) for x in fast.split("|") if x.strip().isdigit()]
    start, end, period = validate(cfg, int(cur.get("period", "0") or 0), fwtype=fwtype, limit_ms=limit,
                                  allowed=allowed)
    if fwtype not in SOLO and cur.get("mode", "continuous") != "continuous":  # checked before anything is written
        raise ConfigError(f"the logger's sampling mode is {cur.get('mode')!r}: only a continuous schedule is "
                          "supported here; set it with Ruskin")
    battery = cfg.battery_fraction()
    if battery is not None and fwtype == 0:  # checked before anything is written
        raise ConfigError("this logger (fwtype 0) has no energy counter: --fresh-battery and --used-battery do not "
                          "apply; leave them out")
    if battery is not None and fwtype not in SOLO:
        raise ConfigError(f"the battery counter of fwtype {fwtype} is not supported yet (its units and a new cell's "
                          "value are unverified): leave out --fresh-battery and --used-battery")
    if cfg.enable and not cfg.erase:  # checked before anything is written: the logger would refuse at `verify`
        used = _used(link, fwtype)
        if used > 0:
            raise ConfigError(f"logger memory holds {used} bytes: enabling needs an erase (the logger refuses with "
                              "E0402). Allow the erase, or use --no-enable to change settings only.")
    try:
        if fwtype in SOLO:
            _apply(link, serial, cfg, ntp_offset_s, log, timing, report, step, start, end, period, battery, fwtype)
        elif fwtype in GEN3:
            _apply_gen3(link, serial, cfg, ntp_offset_s, log, timing, step, start, end, period, cur, driver)
        else:
            _apply_l2(link, serial, cfg, ntp_offset_s, log, timing, step, start, end, period, cur, fwtype, pace_s)
        # Read back what the logger now holds.
        if fwtype in GEN3:
            dep = link.query("deployment")
            back = {"status": dep.get("status", ""), "starttime": dep.get("starttime", ""),
                    "endtime": dep.get("endtime", "")}
        else:
            back = {k: link.query(k)[k] for k in ("status", "starttime", "endtime")}
        back["sampling"] = link.query("sampling")
        if driver is not None:
            after = driver.after(link)
            back["meminfo"], back["power"] = after["meminfo"], after["power"]
        else:
            back["meminfo"], back["power"] = memory(link), power(link)
        if fwtype in GEN3:
            back["offsetfromutc"] = link.query("clock").get("offsetfromutc", "")
            if _offset_nonzero(back["offsetfromutc"]):
                raise ConfigError(f"the UTC offset reads {back['offsetfromutc']!r} after the configure, not 0")
            back["twistactivation"] = link.query("twistactivation")
        if fwtype in SECTIONED_L2:
            back["offsetfromutc"] = link.query("settings").get("offsetfromutc", "")
            if _offset_nonzero(back["offsetfromutc"]):
                raise ConfigError(f"the UTC offset reads {back['offsetfromutc']!r} after the configure, not 0")
        if fwtype == 103:
            back["twistactivation"] = link.query("twistactivation")
        with timing("clock check"):
            back["clock"] = _skew(link, pace_s, _clock_of(driver, fwtype))
        report["readback"] = back
    except (ConfigError, LinkError) as err:  # every failure carries what had been done (or sent) so far
        report["error"] = str(err)
        err.report = report
        raise
    if not cfg.enable:
        log(f"    keeping the port quiet for {IDLE_SAVE_S:.0f} s so the logger saves its settings")
        time.sleep(IDLE_SAVE_S)
    return report


def _apply(link, serial, cfg, ntp_offset_s, log, timing, report, step, start, end, period, battery, fwtype=9):
    with Session(link, serial) as s:
        status = link.query("status")["status"]
        if status in ("logging", "pending"):  # "stopped", "disabled", "finished" need no stop
            step("stop_sent")
            try:
                s.write("stop", expect="stopped")
            except LoggerError as err:
                if err.code != "0406":  # E0406: already not running (e.g. "stop = finished")
                    raise
            step("stop", was=status)
            log(f"    stopped (was {status})")

        if cfg.set_clock:
            with timing("clock set"):
                clk = set_clock(link, ntp_offset_s, cfg.clock_tolerance_s, log=log)
            step("set_clock", **clk)
            if not clk["ok"]:
                raise ConfigError(f"clock could not be set within {cfg.clock_tolerance_s * 1e3:.0f} ms: "
                                  f"{clk['skew_vs_utc_s']:+.3f} s")

        if battery is not None:
            mj = round(NOMINAL_BATTERY_J * battery * 1000)  # new cell: 33,696,000 mJ -> 2022900, as Ruskin stores it
            hex_mj = f"{mj:X}"
            echo = s.write(f"powerstatus remaining = {hex_mj}", expect="remaining")
            try:  # compare as a number: a zero-padded echo (02022900) is the same value
                echoed = int(parse_pairs(echo).get("remaining", ""), 16)
            except ValueError:
                echoed = None
            if echoed != mj:
                raise ConfigError(f"battery counter write: echo {echo!r} does not match {hex_mj}")
            got = power(link)
            if round(got["energy_remaining_J"] * 1000) != mj:
                raise ConfigError(f"battery counter reads {got['remaining_raw']} after writing {hex_mj}")
            step("battery_counter", fraction_of_new_cell=battery, days_used=cfg.battery_days_used,
                 life_days=cfg.battery_life_days, remaining_raw=got["remaining_raw"],
                 energy_remaining_J=got["energy_remaining_J"])
            what = "a new cell" if battery == 1 else (f"{battery:.0%} of a new cell ({cfg.battery_days_used:g} of "
                                                      f"{cfg.battery_life_days:g} days used)")
            log(f"    battery counter set to {got['energy_remaining_J']:.0f} J, {what}")

        s.write(f"starttime = {start}")
        # `endtime =` is verified on fwtype 9 (SN100689). Ruskin never wrote it to a fwtype-0 logger in the logs in
        # hand, so there it is only written when the logger's end time differs from the wanted one.
        endtime_written = True
        if fwtype == 0 and link.query("endtime").get("endtime") == end:
            endtime_written = False
            link.note(f"endtime already {end}: not written (the write is unverified on fwtype 0)")
        else:
            s.write(f"endtime = {end}")
        s.write(f"sampling mode = {cfg.mode}, period = {period}")
        step("schedule", starttime=start, endtime=end, mode=cfg.mode, period_ms=period, endtime_written=endtime_written)
        log(f"    schedule: {cfg.mode}, {period} ms, start {start}, end {end}")

        if cfg.erase:
            s.write("permit memclear", expect="memclear")
            t0 = time.monotonic()
            step("erase_sent")
            link.command_until_prompt("memclear", timeout=300)
            mem = memory(link)
            step("erase", seconds=round(time.monotonic() - t0, 1), meminfo=mem)
            if mem.get("used", -1) != 0:
                raise ConfigError(f"memory not empty after memclear: {mem}")
            log(f"    memory erased ({time.monotonic() - t0:.1f} s)")

        v_code, v_reply = _warn_ok(link, "verify")
        step("verify", reply=v_reply, warning=v_code)
        if v_code and v_code != "0401" and not (v_code == "0402" and not cfg.enable):  # E0402 matters only to enable
            raise ConfigError(f"verify failed: E{v_code} {v_reply}")

        if cfg.enable:
            s.unlock()
            step("enable_sent")
            e_code, e_reply = _warn_ok(link, "enable")
            step("enable", reply=e_reply, warning=e_code)
            if e_code and e_code != "0401":
                raise ConfigError(f"enable failed: E{e_code} {e_reply}")
            log(f"    enabled: {e_reply.strip(' ,')}" + (" (W: memory fills before end time)" if e_code else ""))


def _clock_of(driver, fwtype: int):
    """The skew probe's clock read: the driver's (`clock` on Gen3), or None for the L2 default `now`."""
    return driver.clock_now if driver is not None and fwtype in GEN3 else None


def _status(link: Link, fwtype: int) -> str:
    return link.query("deployment status")["status"] if fwtype in GEN3 else link.query("status")["status"]


def _used(link: Link, fwtype: int) -> int:
    """Bytes in memory: `meminfo used` (L2), or datasets 0 and 1 together (Gen3)."""
    if fwtype in GEN3:
        return sum(int(link.query(f"meminfo dataset {d}").get("used", 0) or 0) for d in (0, 1))
    return memory(link).get("used", 0)


def _offset_nonzero(value: str) -> bool:
    """`settings offsetfromutc` ('+0.00') is not zero, or not a number."""
    try:
        return float(str(value).replace("+", "").strip()) != 0
    except ValueError:
        return True


def _confirmed(link: Link, s: Session, cmd: str, confirm, expect: str | None = None) -> str:
    """A setting write to a logger that drops a command now and then (no reply at all; RBRduet, RBRconcerto):
    if no reply comes, `confirm()` reads the setting back. True: it took effect and only its reply was lost.
    False: it is sent once more. Every write here can be repeated safely; the clock write is not one of them
    (set_clock re-measures instead)."""
    try:
        return s.write(cmd, expect=expect)
    except LoggerError:
        raise
    except LinkError as err:
        link.note(f"no reply to {cmd!r} ({err}); reading it back")
        link._drain(DROP_DRAIN_S)
        if confirm():
            link.note(f"{cmd!r} took effect; its reply was lost")
            return ""
        link.note(f"{cmd!r} did not take effect; sending it once more")
        return s.write(cmd, expect=expect)


def _apply_l2(link, serial, cfg, ntp_offset_s, log, timing, step, start, end, period, cur, fwtype, pace_s):
    """RBRduet (102) and RBRconcerto (103): Ruskin's sequence (module docstring), writing only what changes. A
    write whose reply does not come is read back, then sent again once if it did not take (_confirmed)."""
    with Session(link, serial) as s:
        status = link.query("status")["status"]
        if status in READY_STATUSES:  # "gated": enabled and waiting for a twist; Ruskin stops it (2026-09-12)
            step("stop_sent")
            try:
                _confirmed(link, s, "stop", lambda: link.query("status")["status"] not in READY_STATUSES,
                           expect="stopped")
            except LoggerError as err:
                if err.code != "0406":
                    raise
            step("stop", was=status)
            log(f"    stopped (was {status})")

        if cfg.set_clock:
            with timing("clock set"):
                clk = set_clock(link, ntp_offset_s, cfg.clock_tolerance_s, log=log, pace_s=pace_s)
            step("set_clock", **clk)
            if not clk["ok"]:
                raise ConfigError(f"clock could not be set within {cfg.clock_tolerance_s * 1e3:.0f} ms: "
                                  f"{clk['skew_vs_utc_s']:+.3f} s")
        # after the clock: a clock write leaves the offset `unknown` (module docstring)
        off = link.query("settings").get("offsetfromutc", "")
        if _offset_nonzero(off):
            for attempt in (1, 2):
                _confirmed(link, s, "permit settings", lambda: False, expect="settings")
                try:
                    r = link.command("settings offsetfromutc = 0.0")  # at once: the permit is for the next one
                    break
                except LoggerError:
                    raise
                except LinkError as err:
                    link._drain(DROP_DRAIN_S)
                    r = "settings offsetfromutc = " + link.query("settings").get("offsetfromutc", "?")
                    if not _offset_nonzero(parse_pairs(r).get("offsetfromutc", "?")) or attempt == 2:
                        link.note(f"no reply to the offsetfromutc write ({err}); it reads back {r!r}")
                        break
            if _offset_nonzero(parse_pairs(r).get("offsetfromutc", "?")):
                raise ConfigError(f"settings offsetfromutc = 0.0: unexpected reply {r!r}")
            step("offsetfromutc", was=off, reply=r)
            log(f"    UTC offset was {off}: set to 0")

        written = []
        for key, want in (("starttime", start), ("endtime", end)):
            if link.query(key)[key] != want:
                _confirmed(link, s, f"{key} = {want}", lambda key=key, want=want: link.query(key)[key] == want)
                written.append(key)

        def period_now() -> int:
            return int(link.query("sampling").get("period", "0") or 0)

        if period_now() != period:
            _confirmed(link, s, f"sampling period = {period}", lambda: period_now() == period)  # Ruskin's form
            written.append("period")
        twist = None
        if fwtype == 103:
            twist = link.query("twistactivation").get("state", "")
            if cfg.twist_activation is not None and (twist == "on") != cfg.twist_activation:
                twist = "on" if cfg.twist_activation else "off"
                _confirmed(link, s, f"twistactivation state = {twist}",
                           lambda: link.query("twistactivation").get("state") == twist)
                written.append("twistactivation")
        step("schedule", starttime=start, endtime=end, mode=cfg.mode, period_ms=period, twist_activation=twist,
             written=written)
        log(f"    schedule: {cfg.mode}, {period} ms, start {start}, end {end}"
            + (f", twist activation {twist}" if twist is not None else "")
            + (f" (written: {', '.join(written)})" if written else " (unchanged)"))

        if cfg.enable:
            v_code, v_reply = _warn_ok(link, "verify")
            step("verify", reply=v_reply, warning=v_code)
            # E0402 (memory not empty) is what an erasing enable is for; without the erase it was refused earlier
            if v_code and v_code not in ("0401", "0402"):
                raise ConfigError(f"verify failed: E{v_code} {v_reply}")
            used_before = memory(link).get("used", 0)
            cmd = "enable"
            if cfg.erase:
                cmd = "enable erasememory = true" + (", memformat = RAWBIN00" if fwtype == 103 else "")
            if cfg.erase:
                step("erase_sent")
            step("enable_sent")
            t0 = time.monotonic()
            e_code, e_reply = _enable(link, s, cmd, used_before, fwtype)
            seconds = round(time.monotonic() - t0, 1)
            if e_code and e_code != "0401":
                raise ConfigError(f"{cmd} failed: E{e_code} {e_reply}")
            mem = memory(link)
            if cfg.erase:
                if used_before > 0 and mem.get("used", used_before) >= used_before:
                    raise ConfigError(f"memory not erased by {cmd!r}: {used_before} bytes before, {mem} after")
                step("erase", seconds=seconds, meminfo=mem, used_before=used_before, command=cmd)
                log(f"    memory erased by the enable ({used_before} -> {mem.get('used')} bytes, {seconds:.1f} s)")
            step("enable", reply=e_reply, warning=e_code)
            log(f"    enabled: {e_reply.strip(' ,')}" + (" (W: memory fills before end time)" if e_code else ""))
            return

        if cfg.erase:
            t0 = time.monotonic()
            for attempt in (1, 2):
                _confirmed(link, s, "permit memclear", lambda: False, expect="memclear")
                step("erase_sent")
                try:
                    link.command_until_prompt("memclear", timeout=300)
                    break
                except LoggerError:
                    raise
                except LinkError as err:
                    link._drain(DROP_DRAIN_S)
                    if memory(link).get("used", -1) == 0 or attempt == 2:
                        link.note(f"no prompt after memclear ({err}); memory now {memory(link)}")
                        break
                    link.note(f"no prompt after memclear ({err}) and memory not empty: erasing again")
            mem = memory(link)
            step("erase", seconds=round(time.monotonic() - t0, 1), meminfo=mem, command="memclear")
            if mem.get("used", -1) != 0:
                raise ConfigError(f"memory not empty after memclear: {mem}")
            log(f"    memory erased ({time.monotonic() - t0:.1f} s)")
        v_code, v_reply = _warn_ok(link, "verify")
        step("verify", reply=v_reply, warning=v_code)
        if v_code and v_code not in ("0401", "0402"):
            raise ConfigError(f"verify failed: E{v_code} {v_reply}")


def _apply_gen3(link, serial, cfg, ntp_offset_s, log, timing, step, start, end, period, cur, driver):
    """RBRconcerto3 and the other Gen3 loggers (fwtype 104, L3 command reference rev L), as Ruskin 2.26.1 configured
    them (its serial logs, 2026-09-10..16): `disable` (if running) -> `disable status = stopped`; `clock datetime =
    <UTC>, offsetfromutc = 0.0`; `deployment endtime = .., starttime = ..`; `sampling period = N`; `twistactivation
    enabled = true|false`; `verify` (`E0402 memory not empty, erase first`); `enable erasememory = true, memformat =
    CALBIN00` -> `enable status = gated, warning = none`. Writing the clock erases the UTC offset (L3 ref 4.1.1), so
    it is zeroed afterwards whenever it does not read zero. An erase without enable is `permit command = memclear` +
    `memclear` -> `memclear used = 0` (L3 ref 4.6.2). Only what changes is written; Ruskin's rewrites of the settings,
    `memformat newtype`, `powerinternal`, `powerexternal`, `power p1..e1` and `ddsampling` are not copied."""
    def status_now() -> str:
        return _status(link, 104)

    with Session(link, serial, clock_cmd="clock") as s:
        status = status_now()
        if status in READY_STATUSES:
            step("stop_sent")
            _confirmed(link, s, "disable", lambda: status_now() not in READY_STATUSES, expect="stopped")
            step("stop", was=status)
            log(f"    disabled (was {status})")

        if cfg.set_clock:
            with timing("clock set"):
                clk = set_clock(link, ntp_offset_s, cfg.clock_tolerance_s, log=log,
                                write="clock datetime = {}, offsetfromutc = 0.0", clock=_clock_of(driver, 104) or
                                (lambda lk, timeout=3.0, retries=1: lk.query("clock", timeout=timeout,
                                                                             retries=retries)["datetime"]))
            step("set_clock", **clk)
            if not clk["ok"]:
                raise ConfigError(f"clock could not be set within {cfg.clock_tolerance_s * 1e3:.0f} ms: "
                                  f"{clk['skew_vs_utc_s']:+.3f} s")
        off = link.query("clock").get("offsetfromutc", "")
        if _offset_nonzero(off):
            _confirmed(link, s, "clock offsetfromutc = 0.0",
                       lambda: not _offset_nonzero(link.query("clock").get("offsetfromutc", "")),
                       expect="offsetfromutc")
            step("offsetfromutc", was=off)
            log(f"    UTC offset was {off}: set to 0")

        written = []
        dep = link.query("deployment")
        if (dep.get("starttime"), dep.get("endtime")) != (start, end):
            def dep_ok() -> bool:
                d = link.query("deployment")
                return (d.get("starttime"), d.get("endtime")) == (start, end)

            _confirmed(link, s, f"deployment endtime = {end}, starttime = {start}", dep_ok,
                       expect=f"starttime = {start}")
            written.append("deployment")

        def period_now() -> int:
            return int(link.query("sampling").get("period", "0") or 0)

        if period_now() != period:
            _confirmed(link, s, f"sampling period = {period}", lambda: period_now() == period)
            written.append("period")
        tw = link.query("twistactivation")
        twist = "on" if tw.get("enabled", "").lower() == "true" else "off"
        if cfg.twist_activation is not None and (twist == "on") != cfg.twist_activation:
            want = "true" if cfg.twist_activation else "false"
            _confirmed(link, s, f"twistactivation enabled = {want}",
                       lambda: link.query("twistactivation").get("enabled", "").lower() == want,
                       expect=f"enabled = {want}")
            twist = "on" if cfg.twist_activation else "off"
            written.append("twistactivation")
        step("schedule", starttime=start, endtime=end, mode=cfg.mode, period_ms=period, twist_activation=twist,
             written=written)
        log(f"    schedule: {cfg.mode}, {period} ms, start {start}, end {end}, twist activation {twist}"
            + (f" (written: {', '.join(written)})" if written else " (unchanged)"))

        if cfg.enable:
            v_code, v_reply = _warn_ok(link, "verify")
            step("verify", reply=v_reply, warning=v_code)
            if v_code and v_code != "0402":
                raise ConfigError(f"verify failed: E{v_code} {v_reply}")
            used_before = _used(link, 104)
            cmd = "enable erasememory = true, memformat = CALBIN00" if cfg.erase else "enable"
            if cfg.erase:
                step("erase_sent")
            step("enable_sent")
            t0 = time.monotonic()
            e_code, e_reply = _enable(link, s, cmd, used_before, 104)
            seconds = round(time.monotonic() - t0, 1)
            if e_code:
                raise ConfigError(f"{cmd} failed: E{e_code} {e_reply}")
            reply = parse_pairs(e_reply)
            if reply.get("status") and reply["status"] not in READY_STATUSES:
                raise ConfigError(f"{cmd}: the logger answered {e_reply!r}")
            used = _used(link, 104)
            if cfg.erase:
                if used_before > 0 and used >= used_before:
                    raise ConfigError(f"memory not erased by {cmd!r}: {used_before} bytes before, {used} after")
                step("erase", seconds=seconds, used=used, used_before=used_before, command=cmd)
                log(f"    memory erased by the enable ({used_before} -> {used} bytes, {seconds:.1f} s)")
            step("enable", reply=e_reply, warning=reply.get("warning"))
            log(f"    enabled: {e_reply.strip()}")
            return

        if cfg.erase:
            t0 = time.monotonic()
            _confirmed(link, s, "permit command = memclear", lambda: False, expect="memclear")
            step("erase_sent")
            try:
                r = link.command("memclear", timeout=300)  # at once: the permit is for the next command
            except LoggerError:
                raise
            except LinkError as err:
                link._drain(DROP_DRAIN_S)
                r = f"memclear (no reply: {err})"
            used = _used(link, 104)
            step("erase", seconds=round(time.monotonic() - t0, 1), used=used, reply=r, command="memclear")
            if used != 0:
                raise ConfigError(f"memory not empty after memclear ({used} bytes): {r!r}")
            log(f"    memory erased ({time.monotonic() - t0:.1f} s)")
        v_code, v_reply = _warn_ok(link, "verify")
        step("verify", reply=v_reply, warning=v_code)
        if v_code and v_code != "0402":
            raise ConfigError(f"verify failed: E{v_code} {v_reply}")


def _enable(link: Link, s: Session, cmd: str, used_before: int, fwtype: int = 102) -> tuple[str | None, str]:
    """Send the enable (unlocked); if no reply comes, read the status and memory back: enabled and (for an
    erasing enable) less memory used than before means it took effect, still stopped means send it once more."""
    for attempt in (1, 2):
        s.unlock()
        try:
            return _warn_ok(link, cmd, timeout=ENABLE_TIMEOUT_S)
        except LinkError as err:  # LoggerError is answered inside _warn_ok
            link._drain(DROP_DRAIN_S)
            status = _status(link, fwtype)
            used = _used(link, fwtype)
            erased = "erasememory" not in cmd or used < used_before or used_before == 0
            if status in READY_STATUSES and erased:
                link.note(f"no reply to {cmd!r} ({err}), but it took effect: status {status}, {used} bytes used")
                return None, f"enable = {status} (reply lost; read back)"
            if attempt == 2 or status in READY_STATUSES:
                raise ConfigError(f"no reply to {cmd!r} and its effect is unclear: status {status}, {used} bytes "
                                  f"used (was {used_before})") from err
            link.note(f"no reply to {cmd!r} ({err}); status {status}, {used} bytes used: sending it once more")
    raise AssertionError("unreachable")


def _warn_ok(link: Link, cmd: str, timeout: float = 3.0) -> tuple[str | None, str]:
    """verify/enable reply either 'x = y' or 'E0401 , x = y' (L2 warning form)."""
    try:
        return None, link.command(cmd, timeout)
    except LoggerError as err:
        return err.code, err.text
