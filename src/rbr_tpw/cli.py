"""rbr-offload: wait for RBR loggers on USB, measure clock skew, download, write CF NetCDF, repeat.

Each connected logger gets its own worker thread, so several offload at once. Read-only toward the
logger unless --configure is given: the logger is left in whatever state it was found (still logging,
if it was).

Everything is logged. OUTDIR/raw/rbr-offload_<UTC>.log has every step of every logger and every
serial exchange; OUTDIR/raw/<SN>_<UTC>.log is one logger's serial transcript, written as it happens.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import serial
import yaml
from serial.tools import list_ports

from . import __version__
from .configure import FAR_FUTURE, PAST, ConfigError, DeployConfig, configure, validate
from .console import DEVICE, Console, setup_logging
from .drivers import DecodeUnavailable, DeploymentMismatch, Downloaded, Driver, Held, driver_for
from .hostclock import ntp_offset, timing_critical
from .link import Link, LinkError
from .ncwrite import skew_vs_utc
from .power import remaining
from .rawbin import event_name, scan_events
from .solo import RUSKIN_LOW_VOLTAGE_V, identify, measure_clock_skew, sha256

log = logging.getLogger(__name__)

POLL_S = 0.5  # port scan interval
SETTLE_S = 1.0  # let USB enumeration settle before opening a new port
STATUS_EVERY_S = 30.0  # one-line summary while more than one logger is in progress
ONCE_GRACE_S = 3.0  # --once: after the last logger finishes, wait this long for another still enumerating


class Stopped(Exception):
    """Ctrl-C: a download stops after its current block (it resumes on reconnect)."""


@dataclass
class Thresholds:
    min_battery_voltage: float = 3.3  # Li-SOCl2 rests near 3.6-3.7 V; lower means nearly exhausted or a bad contact
    min_days: float | None = None  # alarm if the lesser of memory- and energy-limited sampling time is shorter

    @classmethod
    def from_yaml(cls, path: Path) -> Thresholds:
        data = (yaml.safe_load(Path(path).read_text()) or {}).get("thresholds") or {}
        unknown = set(data) - {"min_battery_voltage", "min_days"}
        if unknown:
            raise ConfigError(f"{path}: unknown thresholds {sorted(unknown)}")
        for k, v in data.items():  # checked here: a string would only fail at the first offload, mid-way
            if (v is None and k != "min_days") or isinstance(v, bool) or not isinstance(v, int | float | None):
                raise ConfigError(f"{path}: thresholds.{k} must be a number, not {v!r}")
        return cls(**{k: float(v) if v is not None else None for k, v in data.items()})


@dataclass
class Settings:
    outdir: Path
    ntp_server: str | None = "time.apple.com"
    deploy: DeployConfig | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    assume_yes: bool = False
    console: Console = field(default_factory=Console)
    session_log: Path | None = None
    full_download: bool = False  # read the whole memory even when the deployment is on disk (and verify it)
    stop: threading.Event = field(default_factory=threading.Event)  # Ctrl-C
    configured: set[str] = field(default_factory=set)  # serial numbers configured this session (never twice)
    configure_failed: set[str] = field(default_factory=set)  # serial numbers whose configure failed this session
    not_ready: list[str] = field(default_factory=list)  # loggers that ended NOT READY TO DEPLOY this session
    failed: list[str] = field(default_factory=list)  # loggers whose offload or NetCDF failed this session


_stages: dict[str, tuple[str, str, str]] = {}  # port -> (device label, what its worker is doing, the detail shown)
_stages_lock = threading.Lock()
_status_console: Console | None = None  # the terminal's status line, while run() is on a terminal
_not_ready: dict[str, str] = {}  # port -> why its logger must not be deployed (configure failed or not logging)
_failed: dict[str, str] = {}  # port -> why its offload is incomplete (no download, or no NetCDF when one was due)


def _stage(port: str, what: str | None, detail: str | None = None):
    """What this logger's worker is doing, for the summaries and the status line (`detail`: the fuller text
    the status line shows while this is the only logger, e.g. the download bar)."""
    with _stages_lock:
        if what is None:
            _stages.pop(port, None)
        else:
            _stages[port] = (DEVICE.get(), what, detail or what)
    if what is not None and detail is None:
        log.debug("stage: %s", what)
    _refresh_status()


def _stage_summary() -> str:
    with _stages_lock:
        return "; ".join(f"{dev} {what}" for dev, what, _ in sorted(_stages.values())) or "none"


def _status_enabled() -> bool:
    return _status_console is not None and _status_console.status_enabled


def _refresh_status():
    """The terminal's status line: one logger's stage in full, or the summary when several are in progress."""
    if not _status_enabled():
        return
    with _stages_lock:
        stages = sorted(_stages.values())
    if not stages:
        _status_console.status(None)
    elif len(stages) == 1:
        dev, _, detail = stages[0]
        _status_console.status(f"[{dev}] {detail}")
    else:
        _status_console.status("in progress: " + "; ".join(f"{dev} {what}" for dev, what, _ in stages))


def _port_name(port: str) -> str:
    """A short name for a port that is safe in a file name on any OS: usbmodem101, ttyACM0, COM3, 127.0.0.1_5000."""
    name = port.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].removeprefix("cu.")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "port"


_ntp_cache: dict[str, tuple[float, dict]] = {}
_ntp_lock = threading.Lock()
NTP_REUSE_S = 300.0


def _ntp(server: str) -> dict:
    """ntp_offset(server), reused for NTP_REUSE_S so several loggers (or an offline laptop's timeout) cost one query."""
    with _ntp_lock:
        hit = _ntp_cache.get(server)
        if hit and time.monotonic() - hit[0] < NTP_REUSE_S:
            return dict(hit[1])
        result = ntp_offset(server)
        result["measured_at"] = iso(utcnow())
        _ntp_cache[server] = (time.monotonic(), result)
        return dict(result)


# Ruskin 2.26.1 (SerialServer) treats a port as an RBR logger when its USB vendor ID is 0x0451 (Texas
# Instruments) and its product ID is 0xBEF0-0xBEFF. The manufacturer string is kept as a second test; Windows'
# own USB-serial driver reports "Microsoft" there, so on Windows only the IDs identify a logger.
RBR_USB_VID = 0x0451
RBR_USB_PIDS = range(0xBEF0, 0xBF00)


def is_rbr(p) -> bool:
    return ((p.vid == RBR_USB_VID and p.pid in RBR_USB_PIDS)
            or "RBR" in (p.manufacturer or "") or "RBR" in (p.product or ""))


def rbr_ports() -> set[str]:
    """RBR loggers' serial ports. macOS lists each twice (/dev/cu.* and /dev/tty.*): use cu.*, which does not
    wait for carrier. Linux: /dev/ttyACM*. Windows: COM<n>."""
    darwin = platform.system() == "Darwin"
    return {p.device for p in list_ports.comports() if is_rbr(p) and (p.device.startswith("/dev/cu.") or not darwin)}


def port_info(port: str) -> dict:
    """What the OS reports about a port (USB IDs, names), for the session log and the offload record."""
    for p in list_ports.comports():
        if p.device == port:
            return {"vid": f"0x{p.vid:04X}" if p.vid is not None else None,
                    "pid": f"0x{p.pid:04X}" if p.pid is not None else None,
                    "serial_number": p.serial_number, "manufacturer": p.manufacturer, "product": p.product,
                    "description": p.description, "location": p.location, "hwid": p.hwid}
    return {}


def _present(port: str | None) -> set[str]:
    """Ports to handle now: every RBR port, or just --port (used as given, whatever it reports) while it exists.
    --port may also be a pyserial URL such as socket://host:port."""
    if not port:
        return rbr_ports()
    if "://" in port:
        return {port}
    if platform.system() == "Windows":  # COM ports are not file-system paths
        return {port} if port.upper() in _windows_com_ports() else set()
    return {port} if os.path.exists(port) else set()


def _windows_com_ports() -> set[str]:
    """Every COM port Windows knows, upper-case. pyserial lists only devices of the standard Ports class, so
    it misses e.g. com0com's virtual ports; the registry's SERIALCOMM list (what .NET GetPortNames reads) has
    them all."""
    names = {p.device.upper() for p in list_ports.comports()}
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as key:
            i = 0
            while True:
                try:
                    names.add(str(winreg.EnumValue(key, i)[1]).upper())
                except OSError:
                    break
                i += 1
    except (ImportError, OSError):
        pass
    return names


def ruskin_running() -> bool:
    """Is Ruskin running? It polls every RBR port it sees. (On Windows it would also hold the port: opening
    it then fails with "Access is denied", which the worker reports as in use.)"""
    try:
        if platform.system() == "Windows":
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Ruskin.exe", "/NH"], capture_output=True,
                               text=True, check=False)
            return "ruskin.exe" in r.stdout.lower()
        # exact process-name match; "-f" would also match any shell whose command line mentions Ruskin
        return subprocess.run(["pgrep", "-x", "Ruskin"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:  # no pgrep/tasklist: cannot tell
        return False


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def alarm(lines: list[str], s: Settings):
    """Loud, hard-to-miss warning: bell and red banner on the console, then (interactively) an acknowledgement."""
    for x in lines:
        log.warning("%s", x, extra={"banner": True})
    if not s.assume_yes and s.console.interactive:
        s.console.ask(f"  [{DEVICE.get()}] press Enter to acknowledge the alarm ")
        log.debug("alarm acknowledged")


BAR_WIDTH = 24


def _progress_logger(s: Settings, port: str):
    """download() progress callback: the status line's bar on a terminal, a log line every 10% (at DEBUG on a
    terminal, where the bar shows it; INFO otherwise); raises Stopped after a block once Ctrl-C was hit."""
    next_pct = [10.0]

    def progress(done: int, total: int, rate: float):
        pct = 100 * done / total
        eta = (total - done) / rate if rate > 0 else math.inf
        eta_s = f"{int(eta // 60)}:{int(eta % 60):02d}" if math.isfinite(eta) else "?"
        k = int(BAR_WIDTH * done / total)
        bar = "=" * k + (">" if k < BAR_WIDTH else "") + " " * max(BAR_WIDTH - k - 1, 0)
        _stage(port, f"downloading {pct:.0f}%",
               f"downloading {done / 1e6:.2f} of {total / 1e6:.2f} MB [{bar}] {pct:3.0f}%  {rate / 1e3:.0f} kB/s  "
               f"{eta_s} left")
        if pct >= next_pct[0] or done == total:
            log.log(logging.DEBUG if _status_enabled() else logging.INFO,
                    "downloaded %.2f of %.2f MB (%.0f%%), %.1f kB/s, %s left",
                    done / 1e6, total / 1e6, pct, rate / 1e3, eta_s)
            next_pct[0] = (pct // 10 + 1) * 10
        if s.stop.is_set() and done < total:
            raise Stopped()

    return progress


def _write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _store_dataset(path: Path, blob: bytes, offset: int, header_changed: list[int]):
    """Save a dataset: whole (atomically) for a new file or a full read, else by appending the bytes past
    `offset` to the deployment's file, and refreshing its header bytes when the logger's changed."""
    if offset == 0 or not path.exists() or path.stat().st_size != offset:
        _write_bytes(path, blob)
        return
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(blob[offset:])
        if header_changed:
            n = max(header_changed) + 1
            f.seek(0)
            f.write(blob[:n])
        f.flush()
        os.fsync(f.fileno())


@dataclass
class Deployment:
    """A deployment already on disk: its stem, its offload records in order, and its verified raw datasets."""

    stem: str
    records: list[dict]
    data: dict[str, bytes]

    @property
    def latest(self) -> dict:
        return self.records[-1]

    @property
    def next_index(self) -> int:
        return int(self.latest["deployment"].get("offload_index", len(self.records) - 1)) + 1


def _find_deployment(rawdir: Path, sn: str) -> Deployment | None:
    """The latest deployment of this logger on disk, its raw files checked against its latest record. None if
    there is none, or the files cannot be trusted: a new deployment then starts and nothing on disk is touched."""
    by_stem: dict[str, list[tuple[tuple, dict]]] = {}
    for p in sorted(rawdir.glob(f"{sn}_*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        dep = rec.get("deployment") if isinstance(rec, dict) else None
        if not isinstance(dep, dict) or not dep.get("stem"):
            continue
        key = (int(dep.get("offload_index", 0) or 0), str(rec.get("offload_started", "")))
        by_stem.setdefault(str(dep["stem"]), []).append((key, rec))
    if not by_stem:
        return None
    stem, entries = max(by_stem.items(), key=lambda kv: max(e[0] for e in kv[1]))
    entries.sort(key=lambda e: e[0])
    records = [rec for _, rec in entries]
    latest = records[-1]
    datasets = latest.get("datasets") or ({"dataset1": latest["raw"]} if latest.get("raw") else {})
    data: dict[str, bytes] = {}
    for name, info in datasets.items():
        path = rawdir / Path(str(info.get("file", ""))).name
        want, want_sha = int(info.get("bytes", 0) or 0), str(info.get("sha256", ""))
        if not path.is_file():
            log.error("deployment %s: %s is missing; starting a new deployment (nothing on disk is changed)",
                      stem, path.name)
            return None
        blob = path.read_bytes()
        if len(blob) != want or sha256(blob) != want_sha:
            if len(blob) > want and sha256(blob[:want]) == want_sha:
                log.warning("deployment %s: %s holds %d bytes past its record (an interrupted offload); "
                            "truncating it to %d", stem, path.name, len(blob) - want, want)
                with open(path, "r+b") as f:
                    f.truncate(want)
                    f.flush()
                    os.fsync(f.fileno())
                blob = blob[:want]
            else:
                log.error("deployment %s: %s does not match its record (%d bytes, expected %d%s); starting a new "
                          "deployment (nothing on disk is changed)", stem, path.name, len(blob), want,
                          "" if len(blob) != want else ", checksum differs")
                return None
        data[name] = blob
    log.debug("deployment %s on disk: %d record(s), %s", stem, len(records),
              ", ".join(f"{k} {len(v)} B" for k, v in data.items()))
    return Deployment(stem, records, data)


def _changes_since(ident: dict, snap: dict, skew: dict, ntp: dict, previous: dict) -> list[str]:
    """What differs from the deployment's previous offload record: settings, status, and a clock skew that
    jumped more than drift allows (the clock was set or reset in between)."""
    out = []
    pid, psnap = previous.get("id") or {}, previous.get("snapshot_before") or {}
    if str(pid.get("version", "")) != str(ident.get("version", "")):
        out.append(f"firmware version {pid.get('version')} -> {ident.get('version')}")
    for key in ("mode", "period"):
        a, b = (psnap.get("sampling") or {}).get(key), (snap.get("sampling") or {}).get(key)
        if str(a) != str(b):
            out.append(f"sampling {key} {a} -> {b}")
    for key in ("starttime", "endtime"):
        if str(psnap.get(key, "")) != str(snap.get(key, "")):
            out.append(f"{key} {psnap.get(key)} -> {snap.get(key)}")
    fmt_a = (psnap.get("memformat") or {}).get("type", "")
    fmt_b = (snap.get("memformat") or {}).get("type", "")
    if fmt_a != fmt_b:
        out.append(f"memory format {fmt_a} -> {fmt_b}")

    def table(sn):
        return [(str(c.get("type")), int(c.get("status", 0) or 0), json.dumps(c.get("coefficients", {}), sort_keys=True,
                                                                              default=str))
                for c in sn.get("channels_all") or sn.get("channel_list") or []]

    if table(psnap) != table(snap):
        out.append("channel table (types, statuses or calibration coefficients)")
    pstatus = (previous.get("after") or {}).get("status") or psnap.get("status")
    if str(pstatus) != str(snap.get("status")):
        out.append(f"status {pstatus} -> {snap.get('status')}")
    prev_skew = skew_vs_utc(previous)
    now_skew = skew_vs_utc({"clock_skew": skew, "host_ntp": ntp})
    if prev_skew is not None and now_skew is not None and math.isfinite(prev_skew) and math.isfinite(now_skew):
        try:
            elapsed = (utcnow() - dt.datetime.fromisoformat(previous["offload_started"].replace("Z", "+00:00"))
                       ).total_seconds()
        except (KeyError, ValueError):
            elapsed = 0.0
        tol = 2.0 + 50e-6 * max(elapsed, 0.0)
        if abs(now_skew - prev_skew) > tol:
            out.append(f"clock skew {prev_skew:+.3f} s -> {now_skew:+.3f} s ({elapsed / 86400:.1f} days apart, "
                       f"beyond {tol:.1f} s): the clock was set or reset in between")
    return out


def _write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _active_ms(snap: dict) -> float:
    ch = snap.get("channels", {})
    return float(ch.get("latency", 0) or 0) + float(ch.get("readtime", 0) or 0)


def _remaining(snap: dict, mem: dict, pwr: dict, period_ms: int, driver: Driver) -> dict | None:
    if not period_ms:
        return None
    return remaining(pwr.get("energy_remaining_J", math.nan), mem["used"], mem["remaining"], period_ms,
                     len(snap["channel_list"]), _active_ms(snap), bytes_per_sample=driver.bytes_per_sample(snap),
                     energy_model=driver.energy_model)


def _describe(rem: dict) -> str:
    if not math.isfinite(rem["energy_days"]):
        return f"{rem['days']:.0f} days, limited by {rem['limited_by']}"
    return (f"{rem['days']:.0f} days, limited by {rem['limited_by']} "
            f"(energy ~{rem['energy_days']:.0f} d modelled, memory {rem['memory_days']:.0f} d)")


def _time_checks(sn: str, label: str, rem: dict | None, v: float, s: Settings) -> list[str]:
    """Threshold checks; returns alarm lines (empty if all is well)."""
    out = []
    th = s.thresholds
    if not v >= th.min_battery_voltage:
        what = "dead, missing or not making contact" if v < RUSKIN_LOW_VOLTAGE_V else "nearly exhausted?"
        out.append(f"SN{sn} {label}: BATTERY {v:.3f} V < {th.min_battery_voltage:.2f} V ({what})")
    if th.min_days is not None and rem is not None and rem["days"] < th.min_days:
        out.append(f"SN{sn} {label}: ONLY {rem['days']:.0f} DAYS OF SAMPLING LEFT (< {th.min_days:g}), "
                   f"limited by {rem['limited_by']}")
    return out


def _configure_state(done: list[str]) -> str:
    """What a failed configure did to the logger, from its step report ("x_sent" without "x": unconfirmed)."""
    def said(step: str, yes: str, maybe: str, no: str | None) -> str | None:
        return yes if step in done else maybe if f"{step}_sent" in done else no

    parts = [said("erase", "MEMORY WAS ERASED", "MEMORY MAY HAVE BEEN ERASED (sent, not confirmed)",
                  "memory not erased"),
             said("stop", "logging was stopped", "logging may have been stopped", None),
             said("enable", "logging was enabled (a later step failed)", "logging may have been enabled", None)]
    return ", ".join(p for p in parts if p)


def _configure_step(link: Link, s: Settings, snap: dict, ntp: dict, sn: str, erase_note: str,
                    driver: Driver | None = None) -> dict:
    cfg = s.deploy
    if driver is not None and not driver.configurable:
        log.warning("--configure is implemented only for RBRsolo (fwtype 9 and 0) so far; SN%s (fwtype %s) was "
                    "offloaded but not changed", sn, driver.fwtype)
        _not_ready[link.port] = f"not configured: --configure is not supported for fwtype {driver.fwtype}"
        return {"skipped": f"configure not supported for fwtype {driver.fwtype}"}
    plan = []
    no_ntp = ntp.get("offset_s") is None
    if cfg.set_clock:
        plan.append("set clock to UTC" if not no_ntp else "set clock to HOST time (" + (
            f"NTP failed: {ntp.get('error')}" if s.ntp_server else "--no-ntp") + ")")
    battery = cfg.battery_fraction()
    if battery == 1:
        plan.append("reset battery counter to a fresh cell")
    elif battery is not None:
        plan.append(f"set battery counter to {battery:.0%} of a fresh cell "
                    f"({cfg.battery_days_used:g} of {cfg.battery_life_days:g} days used)")
    period = f"{cfg.period_ms} ms" if cfg.period_ms else "period unchanged"
    plan.append(f"schedule {cfg.mode}, {period}, start {cfg.start}, end {cfg.end}")
    if cfg.erase:
        plan.append(f"ERASE memory ({erase_note})")
    if cfg.enable:
        plan.append("enable logging")
    log.info("configure SN%s: %s", sn, "; ".join(plan))
    if cfg.set_clock and no_ntp and s.ntp_server:  # asked for UTC, getting the unchecked host clock: say it loudly
        log.warning("SN%s: THE CLOCK WILL BE SET FROM THE HOST CLOCK, NOT CHECKED AGAINST UTC (NTP failed: %s)",
                    sn, ntp.get("error"), extra={"banner": True})
    if sn in s.configured:
        log.warning("SN%s was already configured in this session; not configuring it again (it may have been "
                    "unplugged and replugged). Restart rbr-offload to configure it again.", sn)
        if snap.get("status") not in ("logging", "pending"):
            _not_ready[link.port] = f"configured earlier in this session, but its status is now '{snap.get('status')}'"
        return {"skipped": "already configured in this session"}
    if s.stop.is_set():
        log.warning("stopping (Ctrl-C): configuration skipped; logger unchanged")
        _not_ready[link.port] = "not configured: stopped (Ctrl-C) first; logger unchanged"
        return {"skipped": "stopping"}
    if sn in s.configure_failed:  # safe to retry: the offload above saved whatever the logger held
        log.warning("SN%s: configure failed earlier in this session; trying again (its memory was downloaded and "
                    "saved again first)", sn)
    if not s.assume_yes:
        answer = s.console.ask(f"  [{DEVICE.get()}] configure SN{sn} as above? [y/N] ")
        log.info("operator answered %r", answer)
        if answer.strip().lower() not in ("y", "yes"):
            log.info("configuration skipped; logger unchanged")
            _not_ready[link.port] = "not configured: the operator declined; logger unchanged"
            return {"skipped": True}
    _stage(link.port, "configuring")
    try:
        # Runs to the end even after Ctrl-C: stopping between erase and enable would leave the logger idle.
        report = configure(link, int(sn), cfg, ntp.get("offset_s"), log=log.info, timing=timing_critical,
                           fwtype=driver.fwtype if driver is not None else 9)
    except (ConfigError, LinkError) as err:
        log.error("CONFIGURE FAILED: %s", err)
        log.debug("configure traceback", exc_info=True)
        done = [st["step"] for st in ((getattr(err, "report", None) or {}).get("steps", []))]
        try:
            status = link.query("status")["status"]
        except Exception:  # the link itself may be what failed
            status = "unknown"
        state = _configure_state(done)
        s.configure_failed.add(sn)
        _not_ready[link.port] = f"configure failed ({state}; status now {status})"
        alarm([f"SN{sn}: CONFIGURE FAILED: {err}",
               f"SN{sn}: {state}; status is now '{status}'. DO NOT DEPLOY until it is configured and logging."], s)
        return {"error": str(err), "steps_done": done, "status_after": status}
    s.configured.add(sn)
    rb = report["readback"]
    if cfg.enable and rb["status"] not in ("logging", "pending"):
        _not_ready[link.port] = f"status is '{rb['status']}' after enable"
        alarm([f"SN{sn}: STATUS IS '{rb['status']}' AFTER ENABLE, NOT LOGGING OR PENDING. DO NOT DEPLOY."], s)
    elif not cfg.enable and rb["status"] not in ("logging", "pending"):
        _not_ready[link.port] = f"not enabled (--no-enable); status '{rb['status']}'"
        log.warning("SN%s: configured but not enabled (--no-enable): status '%s', it is not logging", sn,
                    rb["status"])
    clk = rb["clock"]
    off = ntp.get("offset_s") or 0.0
    log.info("now: status %s, %s %s ms, start %s, end %s, memory used %s B, battery %.3f V / %.0f J, "
             "clock %+.1f ms vs %s", rb["status"], rb["sampling"].get("mode"), rb["sampling"].get("period"),
             rb["starttime"], rb["endtime"], rb["meminfo"]["used"], rb["power"]["battery_voltage_V"],
             rb["power"]["energy_remaining_J"], 1e3 * (clk["skew_vs_host_s"] - off), "host" if no_ntp else "UTC")

    rem = _remaining(snap, rb["meminfo"], rb["power"], int(rb["sampling"].get("period", 0) or 0), driver or
                     driver_for(9))
    report["remaining_time"] = rem
    if rem:
        log.info("expected sampling time: %s", _describe(rem))
        alarms = _time_checks(sn, "new deployment", rem, rb["power"]["battery_voltage_V"], s)
        if rb["endtime"] != FAR_FUTURE:
            start = rb["starttime"] if rb["starttime"] != PAST else utcnow().strftime("%Y%m%d%H%M%S")
            fmt = "%Y%m%d%H%M%S"
            span = (dt.datetime.strptime(rb["endtime"], fmt) - dt.datetime.strptime(start, fmt)).total_seconds()
            if rem["days"] * 86400 < span:
                alarms.append(f"SN{sn} new deployment: RUNS OUT ({rem['limited_by']}) "
                              f"{span / 86400 - rem['days']:.0f} DAYS BEFORE THE END TIME")
        if alarms:
            report["alarms"] = alarms
            alarm(alarms, s)
    return report


def offload(port: str, s: Settings) -> Path | None:
    """Offload one logger (and configure it with --configure). Runs in that logger's worker thread."""
    started = utcnow()
    tag = started.strftime("%Y%m%dT%H%M%SZ")
    rawdir = s.outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    # The transcript is named by serial number once the logger reports it (by port if it never does).
    with Link(port, fallback_transcript=rawdir / f"{tag}_{_port_name(port)}.log") as link:
        _stage(port, "identifying")
        ident = identify(link)
        sn = ident["serial"]
        DEVICE.set(f"SN{sn}@{_port_name(port)}")
        rec_stem = f"{sn}_{tag}"  # this offload's record and transcript; the deployment's files may be older
        stem = rec_stem
        link.start_transcript(rawdir / f"{rec_stem}.log")
        log.info("%s SN%s, firmware %s, fwtype %s on %s", ident["model"], sn, ident["version"], ident["fwtype"], port)
        log.debug("serial transcript: %s", link.transcript_path)
        driver = driver_for(ident["fwtype"])
        if driver is None:
            log.error("fwtype %s is not supported yet; nothing downloaded, nothing changed on the logger",
                      ident["fwtype"])
            _failed[port] = f"fwtype {ident['fwtype']} is not supported; nothing downloaded"
            return None
        driver.serial = int(sn) if str(sn).isdigit() else None  # None: no unlock; the channel table is read locked
        log.debug("driver: %s (%s)", type(driver).__name__, driver.family)

        _stage(port, "NTP query")
        ntp = _ntp(s.ntp_server) if s.ntp_server else {"error": "disabled"}
        log.debug("host NTP: %s", ntp)
        if s.ntp_server and ntp.get("offset_s") is None:
            log.warning("NTP query to %s failed (%s): clock skews are relative to the host clock, not UTC",
                        s.ntp_server, ntp.get("error"))
        _stage(port, "measuring clock skew")
        try:
            with timing_critical("clock skew"):
                skew = measure_clock_skew(link, clock=driver.clock_now)
        except LinkError as err:  # a logger that stops answering for a moment: not a reason to skip the download
            link.note(f"clock skew not measured: {err}")
            skew = {"n": 0, "error": str(err)}
        log.debug("clock skew: %s", skew)
        if skew.get("n"):
            off = ntp.get("offset_s")
            vs_utc = skew["skew_vs_host_s"] - off if off is not None else skew["skew_vs_host_s"]
            log.info("clock skew (logger - %s): %+.3f s +/- %.3f s (n=%d)", "UTC" if off is not None else "host",
                     vs_utc, skew["uncertainty_s"] + (ntp.get("uncertainty_s") or 0), skew["n"])
        else:
            log.warning("clock skew could not be measured")

        _stage(port, "reading settings")
        snap = driver.snapshot(link)
        log.debug("settings: %s", json.dumps(snap, default=str))
        plan = driver.datasets(snap)
        total = sum(n for _, _, n in plan)
        log.info("status %s, sampling %s %s ms, %d bytes in memory%s", snap["status"],
                 snap["sampling"].get("mode"), snap["sampling"].get("period"), total,
                 "" if len(plan) == 1 else " (" + ", ".join(f"{name} {n}" for name, _, n in plan) + ")")

        part_dir = rawdir / ".partial"
        held = _find_deployment(rawdir, sn) if total > 0 else None
        warnings = []
        if held is not None:
            for change in _changes_since(ident, snap, skew, ntp, held.latest):
                warnings.append(f"since the previous offload of deployment {held.stem}: {change}")
        dl: Downloaded | None = None
        if total > 0:
            if s.stop.is_set():
                raise Stopped()
            _stage(port, "downloading")
            prog = _progress_logger(s, port)
            if held is not None:
                try:
                    dl = driver.download(link, snap, part_dir, sn, prog, held=Held(held.stem, held.data),
                                         full=s.full_download)
                except DeploymentMismatch as err:
                    log.warning("SN%s: the logger does not hold deployment %s (%s): starting a new deployment; "
                                "the old files are left as they are", sn, held.stem, err)
                    warnings.append(f"not deployment {held.stem}: {err}; new deployment")
                    held = None
                    if err.downloaded is not None:  # a full read was made: no need to repeat it
                        dl = err.downloaded
                        dl.kind, dl.tail_check, dl.header_changed = "full", None, []
                        dl.segments = {k: (0, len(v)) for k, v in dl.data.items()}
            if dl is None:
                dl = driver.download(link, snap, part_dir, sn, prog, full=s.full_download)
        after = driver.after(link)
        log.debug("after download: %s", after)

        data: dict[str, bytes] = dl.data if dl is not None else {}
        stem = held.stem if held is not None else rec_stem
        datasets = {}
        for name, blob in data.items():
            if not blob:
                continue
            fname = f"{stem}.bin" if list(data) == ["dataset1"] else f"{stem}_{name.replace('/', '_')}.bin"
            off, nbytes = dl.segments.get(name, (0, len(blob)))
            _store_dataset(rawdir / fname, blob, off, dl.header_changed if name == driver.identity_dataset else [])
            datasets[name] = {"file": f"raw/{fname}", "bytes": len(blob), "sha256": sha256(blob), "offset": off,
                              "segment_bytes": nbytes}
        deployment = None
        if datasets:
            primary = driver.primary(data)
            identity = driver.identity(data)
            off, nbytes = dl.segments.get(primary, (0, len(data[primary])))
            deployment = {
                "stem": stem, "identity_dataset": driver.identity_dataset, "header_bytes": len(identity),
                "header_sha256": sha256(identity), "tail_check": dl.tail_check, "header_changed": dl.header_changed,
                "offload_index": held.next_index if held is not None else 0,
                "segment": {"offset": off, "bytes": nbytes, "sha256": sha256(data[primary][off:off + nbytes])},
                "image_bytes": len(data[primary]), "image_sha256": datasets[primary]["sha256"],
                "download": dl.kind,
            }
            if dl.header_changed:
                warnings.append(f"the memory header changed since the previous offload at byte(s) "
                                f"{', '.join(map(str, dl.header_changed))}; refreshed in the saved image")
            if dl.kind == "incremental" and driver.family == "L2" and nbytes:
                new_events = scan_events(data[primary], off)
                if new_events:
                    warnings.append("events in the new data: " + ", ".join(event_name(t) for _, t, _ in new_events))
        if skew.get("n") and abs(skew["skew_vs_host_s"]) > 60:
            warnings.append(f"clock skew {skew['skew_vs_host_s']:+.1f} s exceeds 60 s "
                            "(clock reset, or set to local time instead of UTC?)")
        try:  # derived numbers: an odd reply here must not cost the record of a saved download
            rem = _remaining(snap, after["meminfo"], after["power"], int(snap["sampling"].get("period", 0) or 0),
                             driver)
            alarms = _time_checks(sn, "at offload", rem if after["status"] in ("logging", "pending", "gated")
                                  else None, after["power"]["battery_voltage_V"], s)
        except Exception as err:
            log.error("remaining sampling time not estimated (%s: %s); the download and its record are saved",
                      type(err).__name__, err)
            log.debug("traceback", exc_info=True)
            rem, alarms = None, []
            warnings.append(f"remaining sampling time not estimated: {type(err).__name__}: {err}")
        warnings.extend(alarms)
        record = {
            "tool": {"name": "rbr-tpw", "version": __version__},
            "port": port,
            "port_info": port_info(port),
            "offload_started": iso(started),
            "offload_finished": iso(utcnow()),
            "id": ident,
            "family": driver.family,
            "host_ntp": ntp,
            "clock_skew": skew,
            "snapshot_before": snap,
            "after": after,
            "remaining_time": rem,
            "bytes_per_sample": driver.bytes_per_sample(snap),
            "thresholds": vars(s.thresholds),
            "deployment": deployment,
            "raw": datasets.get("dataset1") or next((v for k, v in datasets.items() if k.endswith("/data")), None)
            or next(iter(datasets.values()), None),
            "datasets": datasets,
            "transcript": f"raw/{rec_stem}.log",
            "session_log": f"raw/{s.session_log.name}" if s.session_log else None,
            "warnings": warnings,
        }
        _write_json(rawdir / f"{rec_stem}.json", record)  # saved before anything is written to the logger
        log.debug("wrote %s", rawdir / f"{rec_stem}.json")
        for f in driver.part_files(part_dir, sn) if part_dir.is_dir() else []:  # only now: all is saved
            f.unlink(missing_ok=True)
        if deployment is not None:
            if dl.kind == "incremental":
                log.info("incremental: %d new bytes from offset %d (deployment %s, offload %d)", dl.new_bytes,
                         deployment["segment"]["offset"], stem, deployment["offload_index"])
            elif dl.kind == "full-verified":
                log.info("full download, verified against deployment %s: %d new bytes (offload %d)", stem,
                         dl.new_bytes, deployment["offload_index"])
            else:
                log.info("new deployment %s: %d bytes", stem, deployment["image_bytes"])
        mem, pwr = after["meminfo"], after["power"]
        energy = (f", energy counter {pwr['energy_remaining_J']:.0f} J of {pwr['energy_nominal_J']:.0f} J nominal"
                  if math.isfinite(pwr.get("energy_remaining_J", math.nan)) else "")
        free = f"{100 * mem['remaining'] / mem['size']:.1f}%" if mem.get("size") else "?"
        log.info("at offload, memory: %.1f of %.1f MB free (%s); battery %.3f V%s", mem["remaining"] / 1e6,
                 mem["size"] / 1e6, free, pwr["battery_voltage_V"], energy)
        if rem:
            log.info("at offload, sampling time left at %s ms: %s", snap["sampling"].get("period"), _describe(rem))
        for w in warnings:
            if w not in alarms:
                log.warning("%s", w)
        if alarms:
            alarm(alarms, s)
        if s.deploy is not None:
            note = f"{total} bytes, saved to raw/{stem}*.bin" if datasets else "already empty"
            logged = int(after["meminfo"].get("used", 0) or 0) - int((snap.get("meminfo") or {}).get("used", 0) or 0)
            if logged > 0:  # still logging during the download: these came after the snapshot and the erase loses them
                note += f"; {logged} bytes logged since the download began are NOT saved (the erase removes them)"
            report = _configure_step(link, s, snap, ntp, sn, note, driver)
            _write_json(rawdir / f"{rec_stem}_configure.json", report)
            log.debug("wrote %s", rawdir / f"{rec_stem}_configure.json")

    if not datasets:
        log.info("logger memory is empty; no NetCDF written")
        return None
    _stage(port, "writing NetCDF")
    nc_path = s.outdir / f"{stem}.nc"
    records = [*(held.records if held is not None else []), record]
    try:  # on the main thread: see console.Console.run_in_main
        all_warnings, t_ms = s.console.run_in_main(driver.write_netcdf, data, records, nc_path)
    except Exception as err:
        how = "no decoder yet" if isinstance(err, DecodeUnavailable) else f"{type(err).__name__}"
        log.warning("NetCDF not written (%s: %s). The download is saved; convert it later with: "
                    "rbr-offload %s --rebuild %s", how, err, s.outdir, rawdir / f"{rec_stem}.json")
        if not isinstance(err, DecodeUnavailable):  # no decoder yet: the saved download is the intended result
            _failed[port] = f"download saved, but NetCDF not written ({how}: {err})"
        log.debug("NetCDF traceback", exc_info=True)
        return None
    log.info("%d samples written%s", len(t_ms),
             f", {iso(dt.datetime.fromtimestamp(t_ms[0] / 1e3, dt.UTC))} to "
             f"{iso(dt.datetime.fromtimestamp(t_ms[-1] / 1e3, dt.UTC))}" if len(t_ms) else "")
    for w in all_warnings:
        if w not in warnings:
            log.warning("%s", w)
    log.info("wrote %s", nc_path)
    return nc_path


def _worker(port: str, s: Settings):
    """One logger, start to finish; never raises."""
    DEVICE.set(_port_name(port))
    _stage(port, "starting")
    try:
        time.sleep(SETTLE_S)
        offload(port, s)
    except Stopped:
        log.warning("download stopped (Ctrl-C) after a complete block; reconnect the logger to resume it")
        _failed[port] = "download stopped (Ctrl-C) before it was complete"
    except serial.SerialException as err:
        if "exclusively lock" in str(err) or "Access is denied" in str(err) or "PermissionError" in str(err):
            # not a failure: another rbr-offload (one per logger is supported) has this port
            log.info("%s is in use by another program (another rbr-offload?); skipped", port)
        else:
            log.error("serial port error: %s", err)
            log.debug("traceback", exc_info=True)
            _failed[port] = f"serial port error: {err}"
    except Exception as err:
        log.error("%s: %s", type(err).__name__, err)
        log.debug("traceback", exc_info=True)
        log.info("A partial download resumes if you reconnect the logger. Details in %s",
                 s.session_log or "the serial transcript")
        _failed[port] = f"{type(err).__name__}: {err}"
    finally:
        _stage(port, None)
        failed, reason = _failed.pop(port, None), _not_ready.pop(port, None)
        if failed:
            s.failed.append(f"{DEVICE.get()}: {failed}")
        if reason:
            s.not_ready.append(f"{DEVICE.get()}: {reason}")
            log.error("done with %s: disconnect it, but it is NOT READY TO DEPLOY: %s", port, reason)
        elif failed:
            log.error("done with %s: disconnect it; OFFLOAD INCOMPLETE: %s", port, failed)
        else:
            log.info("done with %s: disconnect the logger", port)


def run(s: Settings, once: bool, port: str | None):
    """Watch for loggers and give each its own worker thread. Main thread; answers the workers' questions."""
    global _status_console
    workers: dict[str, threading.Thread] = {}
    finished: set[str] = set()  # ports whose worker ended, until they are unplugged
    announced = ruskin_warned = started_any = False
    last_status = last_new = time.monotonic()  # last_new: a worker started or ended (--once grace)
    with _stages_lock:
        _stages.clear()  # this run owns the stage table
    _status_console = s.console
    try:
        _run(s, once, port, workers, finished, announced, ruskin_warned, started_any, last_status, last_new)
    finally:
        if s.console.status_enabled:
            s.console.status(None)
        _status_console = None


def _run(s: Settings, once: bool, port: str | None, workers, finished, announced, ruskin_warned, started_any,
         last_status, last_new):
    try:
        while True:
            present = _present(port)
            for p, t in list(workers.items()):
                if not t.is_alive():
                    del workers[p]
                    finished.add(p)
                    last_new = time.monotonic()  # the --once grace also runs from the last worker's end
            for p in sorted(finished - present):
                finished.discard(p)
                log.info("%s disconnected", p)
            if once and started_any and not workers and time.monotonic() - last_new >= ONCE_GRACE_S:
                return
            new = sorted(present - set(workers) - finished)
            if new and ruskin_running():
                if not ruskin_warned:
                    log.warning("Ruskin is running and polls every RBR port it sees; quit Ruskin to continue.")
                    ruskin_warned = True
                new = []
            elif new:
                ruskin_warned = False
            for p in new:
                log.debug("new port %s: %s", p, port_info(p))
                t = threading.Thread(target=_worker, args=(p, s), name=f"offload-{_port_name(p)}", daemon=True)
                workers[p] = t
                t.start()
                started_any, announced = True, False
                last_new = time.monotonic()
                log.debug("started %s for %s", t.name, p)
            if not workers and not announced:
                log.info("Waiting for an RBR logger on USB (Ctrl-C to quit)...")
                announced = True
            if len(workers) > 1 and time.monotonic() - last_status >= STATUS_EVERY_S and not _status_enabled():
                log.info("in progress: %s", _stage_summary())  # on a terminal the status line shows this
                last_status = time.monotonic()
            s.console.serve(POLL_S)
    except KeyboardInterrupt:
        _shutdown(s, workers)


def _shutdown(s: Settings, workers: dict[str, threading.Thread]):
    s.stop.set()
    s.console.stop.set()
    alive = [t for t in workers.values() if t.is_alive()]
    if not alive:
        log.info("Stopped.")
        return
    log.warning("Ctrl-C: stopping. Downloads stop after their current block and resume on reconnect; a logger "
                "being configured finishes first. Press Ctrl-C again to quit at once. In progress: %s",
                _stage_summary())
    try:
        while any(t.is_alive() for t in alive):
            s.console.serve(0.2)  # workers still need the main thread to write their NetCDF files
    except KeyboardInterrupt:
        busy = _stage_summary()
        log.error("quit at once; interrupted: %s", busy)
        if "configuring" in busy:
            log.error("a logger was interrupted while being configured: check it (status, memory) before deploying")
        # The workers are abandoned (daemon threads), so their own bookkeeping never runs: one blocked in
        # run_in_main() with its NetCDF write queued would otherwise leave the exit status 0 (issue #9 item 4).
        with _stages_lock:
            interrupted = list(_stages.values())
        for dev, what, _ in interrupted:
            if not any(f.startswith(f"{dev}:") for f in s.failed):
                s.failed.append(f"{dev}: interrupted (Ctrl-C twice) while {what}")
            if what == "configuring" and not any(r.startswith(f"{dev}:") for r in s.not_ready):
                s.not_ready.append(f"{dev}: interrupted while being configured; check it before deploying")
        return
    log.info("Stopped.")


class RebuildError(Exception):
    pass


def _rebuild(rec_path: Path, outdir: Path, done: set[str] | None = None):
    """NetCDF from one saved offload record (raw/SN_time.json) and its raw files. A record of a deployment
    (several offloads) rebuilds the deployment's file from all of its records; `done` skips a stem rebuilt
    already in this run."""
    record = json.loads(rec_path.read_text())
    if not isinstance(record, dict) or "fwtype" not in (record.get("id") or {}):
        kind = "a configure report" if isinstance(record, dict) and "steps" in record else "not an offload record"
        log.info("%s: %s; skipped", rec_path, kind)
        return
    driver = driver_for(int(record["id"]["fwtype"]))
    if driver is None:
        raise RebuildError(f"fwtype {record['id']['fwtype']} is not supported")
    if not (record.get("datasets") or record.get("raw")):
        log.warning("%s: the logger's memory was empty at this offload; nothing to rebuild", rec_path)
        return
    data = _raw_files(rec_path, record, prefix_ok=True)  # its own files, checked (a prefix once the deployment grew)
    dep = record.get("deployment")
    records = [record]
    stem = rec_path.stem
    if isinstance(dep, dict) and dep.get("stem"):
        stem = str(dep["stem"])
        if done is not None and stem in done:
            log.info("%s: deployment %s was rebuilt already in this run", rec_path.name, stem)
            return
        by_index: dict[tuple, dict] = {}
        for p in sorted(rec_path.parent.glob(f"{record['id']['serial']}_*.json")):
            if p.resolve() == rec_path.resolve():
                continue
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            d = r.get("deployment") if isinstance(r, dict) else None
            if isinstance(d, dict) and d.get("stem") == stem:
                by_index.setdefault((int(d.get("offload_index", 0) or 0), str(r.get("offload_started", ""))), r)
        by_index[(int(dep.get("offload_index", 0) or 0), str(record.get("offload_started", "")))] = record
        records = [by_index[k] for k in sorted(by_index)]
        if records[-1] is not record:
            data = _raw_files(rec_path, records[-1])
        log.info("deployment %s: %d offload record(s)", stem, len(records))
    nc_path = outdir / f"{stem}.nc"
    try:
        warnings, _ = driver.write_netcdf(data, records, nc_path)
    except DecodeUnavailable as err:  # the saved download is all there is for now: not a failure
        log.warning("%s: not written: %s", rec_path, err)
        return
    if done is not None:
        done.add(stem)
    for w in warnings:
        if w not in records[-1].get("warnings", []):
            log.warning("%s: %s", rec_path.name, w)
    log.info("wrote %s", nc_path)


def _raw_files(rec_path: Path, record: dict, prefix_ok: bool = False) -> dict[str, bytes]:
    """The datasets a record names, read and checked against its checksums. With `prefix_ok`, a file that has
    grown since (an earlier record of a deployment) passes when the bytes the record names still match."""
    data = {}
    datasets = record.get("datasets") or ({"dataset1": record["raw"]} if record.get("raw") else {})
    roots = [rec_path.parent.parent.resolve(), rec_path.parent.resolve()]
    for name, info in datasets.items():
        rel = Path(info["file"])
        if rel.is_absolute() or rel.anchor or ".." in rel.parts:  # anchor: "/x" is not absolute on Windows
            raise RebuildError(f"refusing raw file path {info['file']!r}")
        # recorded as raw/<file> relative to OUTDIR; also accept the file beside a moved record
        where = [rec_path.parent.parent / rel, rec_path.parent / rel.name]
        found = next((w for w in where if w.is_file()), None)
        if found is not None and not any(found.resolve().is_relative_to(r) for r in roots):  # e.g. a symlink out
            raise RebuildError(f"refusing raw file {info['file']!r}: it resolves outside {roots[0]}")
        if found is None:
            raise RebuildError(f"{info['file']} not found (looked in {', '.join(map(str, where))})")
        blob = found.read_bytes()
        want = int(info.get("bytes", len(blob)) or 0)
        if prefix_ok and len(blob) > want:
            blob = blob[:want]
        if sha256(blob) != info["sha256"]:
            raise RebuildError(f"{info['file']} checksum does not match the record")
        data[name] = blob
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rbr-offload", description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path, help="directory for NetCDF files (raw downloads go in OUTDIR/raw)")
    ap.add_argument("--once", action="store_true",
                    help="handle the logger(s) connected now, or the first to appear, then exit")
    ap.add_argument("--port", help="only use this serial device (default: any RBR USB port)")
    ap.add_argument("--config", type=Path, metavar="SETTINGS.yaml",
                    help="YAML with 'thresholds' and 'deploy' sections (see deploy.example.yaml)")
    ap.add_argument("--ntp-server", default="time.apple.com",
                    help="NTP server used to reference the host clock to UTC (default %(default)s)")
    ap.add_argument("--no-ntp", action="store_true", help="skip the NTP query; skew is then relative to the host")
    ap.add_argument("--rebuild", nargs="+", type=Path, metavar="RECORD.json",
                    help="regenerate NetCDF from saved raw/*.json + .bin (no logger needed), then exit")
    ap.add_argument("--full-download", action="store_true",
                    help="read the whole memory even when the deployment is already on disk, and verify that "
                         "the memory extends the saved image (default: read only the new bytes)")
    t = ap.add_argument_group("alarms (loud warning at offload and after configure)")
    t.add_argument("--min-voltage", type=float, metavar="V",
                   help="alarm if the internal battery reads below V volts (default 3.3)")
    t.add_argument("--min-days", type=float, metavar="D",
                   help="alarm if fewer than D days of sampling remain (lesser of memory and modelled energy)")
    g = ap.add_argument_group("configure and enable after offload (writes to the logger)")
    g.add_argument("--configure", nargs="?", const="", metavar="SETTINGS.yaml",
                   help="after a successful offload, configure and enable the logger, using the 'deploy' "
                        "section of SETTINGS.yaml (or of --config) and the options below")
    g.add_argument("--period-ms", type=int, help="sampling period, ms (500, or whole seconds)")
    g.add_argument("--rate-hz", type=float, help="sampling rate, Hz (alternative to --period-ms)")
    g.add_argument("--start", help="'now' or ISO-8601 UTC, e.g. 2026-10-01T00:00:00Z")
    g.add_argument("--end", help="'never' or ISO-8601 UTC")
    g.add_argument("--fresh-battery", action="store_true", default=None,
                   help="reset the battery energy counter (a new cell was installed)")
    g.add_argument("--used-battery", metavar="USED/LIFE",
                   help="a used cell was installed: set the energy counter to (LIFE - USED) / LIFE of a new "
                        "cell, both in days, e.g. 14/50 for 14 days used of an expected 50-day life")
    g.add_argument("--no-erase", dest="erase", action="store_false", default=None, help="do not erase memory")
    g.add_argument("--no-enable", dest="enable", action="store_false", default=None, help="configure only")
    g.add_argument("--no-clock", dest="set_clock", action="store_false", default=None, help="leave the clock alone")
    g.add_argument("--yes", action="store_true", help="do not ask before configuring or acknowledging alarms")
    args = ap.parse_args(argv)

    if args.config and args.configure:
        ap.error("give the settings file to --config or to --configure, not both")
    cfg_file = Path(args.configure) if args.configure else args.config
    try:
        thresholds = Thresholds.from_yaml(cfg_file) if cfg_file else Thresholds()
        deploy = None
        if args.configure is not None:
            deploy = DeployConfig.from_yaml(cfg_file) if cfg_file else DeployConfig()
    except (ConfigError, TypeError, yaml.YAMLError) as err:
        ap.error(str(err))
    if args.min_voltage is not None:
        thresholds.min_battery_voltage = args.min_voltage
    if args.min_days is not None:
        thresholds.min_days = args.min_days
    overrides = {k: getattr(args, k) for k in ("period_ms", "start", "end", "fresh_battery", "erase", "enable",
                                               "set_clock") if getattr(args, k) is not None}
    if args.rate_hz:
        overrides["period_ms"] = round(1000 / args.rate_hz)
    if args.used_battery and args.fresh_battery:
        ap.error("--used-battery and --fresh-battery are mutually exclusive")
    if args.used_battery:
        try:
            used, life = (float(x) for x in args.used_battery.split("/"))
        except ValueError:
            ap.error(f"--used-battery {args.used_battery!r}: expected USED/LIFE in days, e.g. 14/50")
        overrides.update(battery_days_used=used, battery_life_days=life, fresh_battery=False)
    elif args.fresh_battery:  # the command line wins over a used battery in the YAML file
        overrides.update(battery_days_used=None, battery_life_days=None)
    if deploy is not None:
        for k, v in overrides.items():
            setattr(deploy, k, v)
        try:
            deploy.battery_fraction()
            # period, start and end now, not after a download; the period is checked again per logger
            validate(deploy, deploy.period_ms or 500)
        except ConfigError as err:
            ap.error(str(err))
    elif overrides:
        ap.error("configuration options need --configure")

    args.outdir.mkdir(parents=True, exist_ok=True)
    console = Console()
    if args.rebuild:
        setup_logging(console)
        failed = []
        done: set[str] = set()
        for rec_path in args.rebuild:
            try:
                _rebuild(rec_path, args.outdir, done)
            except (RebuildError, OSError, ValueError, KeyError) as err:  # report it, and go on to the next
                why = str(err) if isinstance(err, RebuildError) else f"{type(err).__name__}: {err}"
                log.error("%s: NOT rebuilt: %s", rec_path, why)
                log.debug("rebuild traceback", exc_info=True)
                failed.append(f"{rec_path}: {why}")
        if failed:
            sys.exit(f"{len(failed)} of {len(args.rebuild)} record(s) not rebuilt: " + "; ".join(failed))
        return
    rawdir = args.outdir / "raw"
    rawdir.mkdir(exist_ok=True)
    session_log = rawdir / f"rbr-offload_{utcnow():%Y%m%dT%H%M%SZ}.log"
    setup_logging(console, session_log)
    s = Settings(outdir=args.outdir, ntp_server=None if args.no_ntp else args.ntp_server, deploy=deploy,
                 thresholds=thresholds, assume_yes=args.yes, console=console, session_log=session_log,
                 full_download=args.full_download)
    log.debug("rbr-tpw %s; Python %s; pyserial %s; %s %s; argv %s", __version__, platform.python_version(),
              serial.__version__, platform.system(), platform.release(), sys.argv)
    log.debug("settings: thresholds %s; deploy %s; ntp %s; port %s; once %s; yes %s", vars(thresholds),
              vars(deploy) if deploy else None, s.ntp_server, args.port, args.once, args.yes)
    log.info("session log: %s", session_log)
    try:
        run(s, args.once, args.port)
    except Exception:
        log.critical("unexpected error; stopping", exc_info=True)
        raise
    # exit status bits, so scripts (e.g. with --once --yes) can tell: 1 an offload or its NetCDF failed,
    # 2 a logger must not be deployed (configure failed, skipped or declined, or it is not logging)
    if s.failed:
        log.error("%d OFFLOAD(S) INCOMPLETE: %s", len(s.failed), "; ".join(s.failed))
    if s.not_ready:
        log.error("%d logger(s) NOT READY TO DEPLOY: %s", len(s.not_ready), "; ".join(s.not_ready))
    if s.failed or s.not_ready:
        sys.exit((1 if s.failed else 0) | (2 if s.not_ready else 0))


if __name__ == "__main__":
    main()
