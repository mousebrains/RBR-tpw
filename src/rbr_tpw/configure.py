"""Configure and enable an RBRsolo (fwtype 9, and fwtype 0): clock, battery counter, schedule, erase, enable.

The write sequence mirrors what Ruskin 2.26.1 sends to these loggers (from
~/Ruskin/logs/ruskin_serial.log, 2026-09-25): `lock OFF = <session key>`,
`stop`, `now = <UTC>`, `starttime = `, `endtime = `, `sampling mode = , period = `,
`verify`, `permit memclear`, `memclear`, `enable`, `lock on`.
Erasing is only allowed after the same session's download has been saved.

fwtype 0 (RBRsolo firmware 1.110): Ruskin's logs (2026-09-27 check, 13 loggers) show the same commands and
replies, except that there is no energy counter (no `powerstatus remaining =` write) and Ruskin never wrote
`endtime =` to one, so that write is skipped when the logger already holds the wanted end time.
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
from .solo import NOMINAL_BATTERY_J, measure_clock_skew, memory, power

FAR_FUTURE = "20991231235959"  # Ruskin's "no end time"
PAST = "20000101000000"  # start time in the past = start as soon as enabled


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

    @classmethod
    def from_yaml(cls, path: Path) -> DeployConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.pop("thresholds", None)  # read separately (cli.Thresholds)
        data = data.get("deploy", data)
        sched = data.pop("schedule", {}) or {}
        flat = {**data, **{k: v for k, v in sched.items()}}
        if "rate_hz" in flat:
            flat["period_ms"] = round(1000 / float(flat.pop("rate_hz")))
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


def validate(cfg: DeployConfig, current_period_ms: int) -> tuple[str, str, int]:
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
    # Ruskin offers 2 Hz or whole seconds for these loggers.
    if not (period == 500 or (period >= 1000 and period % 1000 == 0)):
        raise ConfigError(f"period {period} ms: use 500 ms (2 Hz) or a whole number of seconds")
    start = _logger_time(cfg.start, now_ok=True)
    end = _logger_time(cfg.end, now_ok=False)
    if end <= start:
        raise ConfigError(f"end {end} is not after start {start}")
    if end != FAR_FUTURE and end <= dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S"):
        raise ConfigError(f"end {end} is in the past")
    return start, end, period


def set_clock(link: Link, ntp_offset_s: float | None, tolerance_s: float, attempts: int = 4, log=print) -> dict:
    """Write `now = <UTC>` so the logger's second boundary lines up with UTC.

    The command is sent `lat` seconds before the target second; the resulting
    skew is measured and `lat` adjusted. Assumes the logger zeroes its
    sub-second counter when the time is written (checked by the re-measurement).
    With ntp_offset_s None (no NTP) the reference is the host clock, and the log says so.
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
        r = link.command(f"now = {stamp}")
        if stamp not in r:
            raise ConfigError(f"clock set: unexpected reply {r!r}")
        skew = measure_clock_skew(link, reps=3)
        if not skew.get("n"):
            raise ConfigError("clock set: could not re-measure skew")
        s_utc = skew["skew_vs_host_s"] - ntp_offset_s
        history.append({"attempt": attempt, "lead_s": lat, "skew_vs_utc_s": s_utc,
                        "uncertainty_s": skew["uncertainty_s"]})
        log(f"    clock set attempt {attempt}: logger - {ref} = {s_utc * 1e3:+.1f} ms "
            f"(+/- {skew['uncertainty_s'] * 1e3:.1f} ms)")
        if abs(s_utc) <= tolerance_s:
            return {"ok": True, "skew_vs_utc_s": s_utc, "uncertainty_s": skew["uncertainty_s"], "reference": ref,
                    "history": history}
        lat -= s_utc  # logger ahead (s > 0) -> we sent too early -> send later
        # lat may go negative (send just after the target second): the target is ~2 s ahead, so any lead in
        # (-0.5, 0.9) s is still in the future. Stopping at lat < 0 made any first skew above +3 ms fatal.
        if not -0.5 < lat < 0.9:
            break
    return {"ok": False, "skew_vs_utc_s": history[-1]["skew_vs_utc_s"] if history else math.nan, "reference": ref,
            "history": history}


def configure(link: Link, serial: int, cfg: DeployConfig, ntp_offset_s: float | None, log=print,
              timing=nullcontext, fwtype: int = 9) -> dict:
    """Apply `cfg`. Caller must already have saved this session's download if cfg.erase.

    `ntp_offset_s` is UTC minus host (None: no NTP, so the clock is set to the host clock).
    `timing(what)` wraps the host-timestamp-sensitive steps (see hostclock.timing_critical).
    `fwtype` is 9 or 0 (see the module docstring for what differs on 0)."""
    if fwtype not in (0, 9):
        raise ConfigError(f"configure is implemented for RBRsolo fwtype 9 and 0, not fwtype {fwtype}")
    report: dict = {"config": asdict(cfg), "steps": [],
                    "clock_reference": "UTC (host clock corrected by NTP)" if ntp_offset_s is not None else
                    "host clock (no NTP)"}

    # "<name>_sent" is recorded before a command that changes the logger, "<name>" once it is confirmed: after
    # a lost reply, "sent" alone means it may or may not have happened
    def step(name, **kw):
        report["steps"].append({"step": name, **kw})

    cur = link.query("sampling")
    start, end, period = validate(cfg, int(cur.get("period", "0") or 0))
    battery = cfg.battery_fraction()
    if battery is not None and fwtype == 0:  # checked before anything is written
        raise ConfigError("this logger (fwtype 0) has no energy counter: --fresh-battery and --used-battery do not "
                          "apply; leave them out")
    if cfg.enable and not cfg.erase:  # checked before anything is written: the logger would refuse at `verify`
        used = memory(link).get("used", 0)
        if used > 0:
            raise ConfigError(f"logger memory holds {used} bytes: enabling needs an erase (the logger refuses with "
                              "E0402). Allow the erase, or use --no-enable to change settings only.")
    try:
        _apply(link, serial, cfg, ntp_offset_s, log, timing, report, step, start, end, period, battery, fwtype)
        # Read back what the logger now holds.
        back = {k: link.query(k)[k] for k in ("status", "starttime", "endtime")}
        back["sampling"] = link.query("sampling")
        back["meminfo"] = memory(link)
        back["power"] = power(link)
        with timing("clock check"):
            back["clock"] = measure_clock_skew(link, reps=3)
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


def _warn_ok(link: Link, cmd: str) -> tuple[str | None, str]:
    """verify/enable reply either 'x = y' or 'E0401 , x = y' (L2 warning form)."""
    try:
        return None, link.command(cmd)
    except LoggerError as err:
        return err.code, err.text
