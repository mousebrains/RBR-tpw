"""Configure and enable an RBRsolo (fwtype 9): clock, battery counter, schedule, erase, enable.

The write sequence mirrors what Ruskin 2.26.1 sends to these loggers (from
~/Ruskin/logs/ruskin_serial.log, 2026-09-25): `lock OFF = <session key>`,
`stop`, `now = <UTC>`, `starttime = `, `endtime = `, `sampling mode = , period = `,
`verify`, `permit memclear`, `memclear`, `enable`, `lock on`.
Erasing is only allowed after the same session's download has been saved.
"""

from __future__ import annotations

import datetime as dt
import math
import struct
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .crc import crc16_ccitt
from .link import Link, LinkError, LoggerError, parse_pairs
from .solo import NOMINAL_BATTERY_J, measure_clock_skew, memory, parse_logger_datetime, power

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


def unlock_key(serial: int, logger_seconds: int) -> int:
    """Write-unlock key for `lock OFF = <key>` on L2-family loggers.

    Challenge-response on the logger's most recently reported `now` (seconds
    since 2000-01-01) and its serial number. Reproduces Ruskin 2.26.1
    PhysicalL2.computeKey; verified against three keys Ruskin sent SN100689
    on 2026-09-25 (little-endian int32 bytes, CRC-16/CCITT-FALSE).
    """
    c_sn = crc16_ccitt(struct.pack("<i", serial))
    c_t = crc16_ccitt(struct.pack("<i", logger_seconds))
    hi = (((serial >> 16) & 0xFFFF) ^ c_t) & 0xFFFF
    lo = ((logger_seconds & 0xFFFF) ^ c_sn) & 0xFFFF
    return (hi << 16) | lo


class Session:
    """Unlocked write session; always re-locks on exit."""

    def __init__(self, link: Link, serial: int):
        self.link = link
        self.serial = serial

    def unlock(self):
        now = self.link.query("now")["now"]  # sets the logger's challenge
        secs = int((parse_logger_datetime(now) - dt.datetime(2000, 1, 1, tzinfo=dt.UTC)).total_seconds())
        r = self.link.command(f"lock OFF = {unlock_key(self.serial, secs)}")
        if "off" not in r.lower():
            raise ConfigError(f"unlock failed: {r!r}")

    def __enter__(self):
        self.unlock()
        return self

    def __exit__(self, *exc):
        try:
            self.link.command("lock on")
        except Exception as err:
            self.link.note(f"lock on failed: {err}")

    def write(self, cmd: str, expect: str | None = None, timeout: float = 3.0) -> str:
        """Send a setting; the logger echoes it back. `expect` is a substring the echo must contain."""
        self.unlock()
        r = self.link.command(cmd, timeout)
        want = (expect if expect is not None else cmd).replace(" ", "").lower()
        if want not in r.replace(" ", "").lower():
            raise ConfigError(f"{cmd!r}: unexpected reply {r!r}")
        return r


def set_clock(link: Link, ntp_offset_s: float, tolerance_s: float, attempts: int = 4, log=print) -> dict:
    """Write `now = <UTC>` so the logger's second boundary lines up with UTC.

    The command is sent `lat` seconds before the target second; the resulting
    skew is measured and `lat` adjusted. Assumes the logger zeroes its
    sub-second counter when the time is written (checked by the re-measurement).
    """
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
        log(f"    clock set attempt {attempt}: logger - UTC = {s_utc * 1e3:+.1f} ms "
            f"(+/- {skew['uncertainty_s'] * 1e3:.1f} ms)")
        if abs(s_utc) <= tolerance_s:
            return {"ok": True, "skew_vs_utc_s": s_utc, "uncertainty_s": skew["uncertainty_s"], "history": history}
        lat -= s_utc  # logger ahead (s > 0) -> we sent too early -> send later
        # lat may go negative (send just after the target second): the target is ~2 s ahead, so any lead in
        # (-0.5, 0.9) s is still in the future. Stopping at lat < 0 made any first skew above +3 ms fatal.
        if not -0.5 < lat < 0.9:
            break
    return {"ok": False, "skew_vs_utc_s": history[-1]["skew_vs_utc_s"] if history else math.nan,
            "history": history}


def configure(link: Link, serial: int, cfg: DeployConfig, ntp_offset_s: float, log=print, timing=nullcontext) -> dict:
    """Apply `cfg`. Caller must already have saved this session's download if cfg.erase.

    `timing(what)` wraps the host-timestamp-sensitive steps (see hostclock.timing_critical)."""
    report: dict = {"config": asdict(cfg), "steps": []}

    # "<name>_sent" is recorded before a command that changes the logger, "<name>" once it is confirmed: after
    # a lost reply, "sent" alone means it may or may not have happened
    def step(name, **kw):
        report["steps"].append({"step": name, **kw})

    cur = link.query("sampling")
    start, end, period = validate(cfg, int(cur.get("period", "0") or 0))
    battery = cfg.battery_fraction()
    if cfg.enable and not cfg.erase:  # checked before anything is written: the logger would refuse at `verify`
        used = memory(link).get("used", 0)
        if used > 0:
            raise ConfigError(f"logger memory holds {used} bytes: enabling needs an erase (the logger refuses with "
                              "E0402). Allow the erase, or use --no-enable to change settings only.")
    try:
        _apply(link, serial, cfg, ntp_offset_s, log, timing, report, step, start, end, period, battery)
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


def _apply(link, serial, cfg, ntp_offset_s, log, timing, report, step, start, end, period, battery):
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
        s.write(f"endtime = {end}")
        s.write(f"sampling mode = {cfg.mode}, period = {period}")
        step("schedule", starttime=start, endtime=end, mode=cfg.mode, period_ms=period)
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
