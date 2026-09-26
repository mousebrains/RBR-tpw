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
from .configure import FAR_FUTURE, PAST, ConfigError, DeployConfig, configure
from .console import DEVICE, Console, setup_logging
from .drivers import DecodeUnavailable, Driver, driver_for
from .hostclock import ntp_offset, timing_critical
from .link import Link, LinkError
from .power import remaining
from .solo import RUSKIN_LOW_VOLTAGE_V, identify, measure_clock_skew, sha256

log = logging.getLogger(__name__)

POLL_S = 0.5  # port scan interval
SETTLE_S = 1.0  # let USB enumeration settle before opening a new port
STATUS_EVERY_S = 30.0  # one-line summary while more than one logger is in progress


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
        return cls(**data)


@dataclass
class Settings:
    outdir: Path
    ntp_server: str | None = "time.apple.com"
    deploy: DeployConfig | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    assume_yes: bool = False
    console: Console = field(default_factory=Console)
    session_log: Path | None = None
    stop: threading.Event = field(default_factory=threading.Event)  # Ctrl-C
    configured: set[str] = field(default_factory=set)  # serial numbers configured this session (never twice)


_stages: dict[str, tuple[str, str]] = {}  # port -> (device label, what its worker is doing)
_stages_lock = threading.Lock()
_not_ready: dict[str, str] = {}  # port -> why its logger must not be deployed (configure failed or not logging)


def _stage(port: str, what: str | None):
    with _stages_lock:
        if what is None:
            _stages.pop(port, None)
        else:
            _stages[port] = (DEVICE.get(), what)
    if what is not None:
        log.debug("stage: %s", what)


def _stage_summary() -> str:
    with _stages_lock:
        return "; ".join(f"{dev} {what}" for dev, what in sorted(_stages.values())) or "none"


def _port_name(port: str) -> str:
    return Path(port).name.removeprefix("cu.")


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


def rbr_ports() -> set[str]:
    """RBR loggers' serial ports. macOS lists each twice (/dev/cu.* and /dev/tty.*): use cu.*, which does not
    wait for carrier. Linux: /dev/ttyACM* (CDC ACM)."""
    darwin = platform.system() == "Darwin"
    return {p.device for p in list_ports.comports()
            if (p.device.startswith("/dev/cu.") or not darwin)
            and ("RBR" in (p.manufacturer or "") or "RBR" in (p.product or ""))}


def _present(port: str | None) -> set[str]:
    """Ports to handle now: every RBR port, or just --port (as given, whatever it reports) while it exists."""
    if port:
        return {port} if os.path.exists(port) else set()
    return rbr_ports()


def ruskin_running() -> bool:
    # exact process-name match; "-f" would also match any shell whose command line mentions Ruskin
    r = subprocess.run(["pgrep", "-x", "Ruskin"], capture_output=True, check=False)
    return r.returncode == 0


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


def _progress_logger(s: Settings, port: str):
    """download() progress callback: a log line every 10%; raises Stopped after a block once Ctrl-C was hit."""
    next_pct = [10.0]

    def progress(done: int, total: int, rate: float):
        pct = 100 * done / total
        _stage(port, f"downloading {pct:.0f}%")
        if pct >= next_pct[0] or done == total:
            eta = (total - done) / rate if rate > 0 else math.inf
            eta_s = f"{int(eta // 60)}:{int(eta % 60):02d}" if math.isfinite(eta) else "?"
            log.info("downloaded %.2f of %.2f MB (%.0f%%), %.1f kB/s, %s left",
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


def _configure_step(link: Link, s: Settings, snap: dict, ntp: dict, sn: str, erase_note: str,
                    driver: Driver | None = None) -> dict:
    cfg = s.deploy
    if driver is not None and not driver.configurable:
        log.warning("--configure is implemented only for RBRsolo fwtype 9 so far; SN%s (fwtype %s) was offloaded "
                    "but not changed", sn, driver.fwtype)
        return {"skipped": f"configure not supported for fwtype {driver.fwtype}"}
    plan = []
    if cfg.set_clock:
        plan.append("set clock to UTC" if ntp.get("offset_s") is not None else "set clock to host time (NTP failed)")
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
    if sn in s.configured:
        log.warning("SN%s was already configured in this session; not configuring it again (it may have been "
                    "unplugged and replugged). Restart rbr-offload to configure it again.", sn)
        return {"skipped": "already configured in this session"}
    if s.stop.is_set():
        log.warning("stopping (Ctrl-C): configuration skipped; logger unchanged")
        return {"skipped": "stopping"}
    if not s.assume_yes:
        answer = s.console.ask(f"  [{DEVICE.get()}] configure SN{sn} as above? [y/N] ")
        log.info("operator answered %r", answer)
        if answer.strip().lower() not in ("y", "yes"):
            log.info("configuration skipped; logger unchanged")
            return {"skipped": True}
    _stage(link.port, "configuring")
    try:
        # Runs to the end even after Ctrl-C: stopping between erase and enable would leave the logger idle.
        report = configure(link, int(sn), cfg, ntp.get("offset_s") or 0.0, log=log.info, timing=timing_critical)
    except (ConfigError, LinkError) as err:
        log.error("CONFIGURE FAILED: %s", err)
        log.debug("configure traceback", exc_info=True)
        done = [st["step"] for st in ((getattr(err, "report", None) or {}).get("steps", []))]
        try:
            status = link.query("status")["status"]
        except Exception:  # the link itself may be what failed
            status = "unknown"
        state = ("MEMORY WAS ERASED" if "erase" in done else "memory not erased") + (
            ", logging was stopped" if "stop" in done else "")
        _not_ready[link.port] = f"configure failed ({state}; status now {status})"
        alarm([f"SN{sn}: CONFIGURE FAILED: {err}",
               f"SN{sn}: {state}; status is now '{status}'. DO NOT DEPLOY until it is configured and logging."], s)
        return {"error": str(err), "steps_done": done, "status_after": status}
    s.configured.add(sn)
    rb = report["readback"]
    if cfg.enable and rb["status"] not in ("logging", "pending"):
        _not_ready[link.port] = f"status is '{rb['status']}' after enable"
        alarm([f"SN{sn}: STATUS IS '{rb['status']}' AFTER ENABLE, NOT LOGGING OR PENDING. DO NOT DEPLOY."], s)
    clk = rb["clock"]
    off = ntp.get("offset_s") or 0.0
    log.info("now: status %s, %s %s ms, start %s, end %s, memory used %s B, battery %.3f V / %.0f J, "
             "clock %+.1f ms vs UTC", rb["status"], rb["sampling"].get("mode"), rb["sampling"].get("period"),
             rb["starttime"], rb["endtime"], rb["meminfo"]["used"], rb["power"]["battery_voltage_V"],
             rb["power"]["energy_remaining_J"], 1e3 * (clk["skew_vs_host_s"] - off))

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
    # The transcript is named by port until the logger says who it is.
    with Link(port, transcript=rawdir / f"{tag}_{_port_name(port)}.log") as link:
        _stage(port, "identifying")
        ident = identify(link)
        sn = ident["serial"]
        DEVICE.set(f"SN{sn}@{_port_name(port)}")
        stem = f"{sn}_{tag}"
        link.rename_transcript(rawdir / f"{stem}.log")
        log.info("%s SN%s, firmware %s, fwtype %s on %s", ident["model"], sn, ident["version"], ident["fwtype"], port)
        log.debug("serial transcript: %s", link.transcript_path)
        driver = driver_for(ident["fwtype"])
        if driver is None:
            log.info("fwtype %s is not supported yet; skipping, nothing was changed on the logger.", ident["fwtype"])
            return None
        log.debug("driver: %s (%s)", type(driver).__name__, driver.family)

        _stage(port, "NTP query")
        ntp = _ntp(s.ntp_server) if s.ntp_server else {"error": "disabled"}
        log.debug("host NTP: %s", ntp)
        _stage(port, "measuring clock skew")
        with timing_critical("clock skew"):
            skew = measure_clock_skew(link, clock=driver.clock_now)
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
        log.info("status %s, sampling %s %s ms, %d bytes to download%s", snap["status"],
                 snap["sampling"].get("mode"), snap["sampling"].get("period"), total,
                 "" if len(plan) == 1 else " (" + ", ".join(f"{name} {n}" for name, _, n in plan) + ")")

        part_dir = rawdir / ".partial"
        data: dict[str, bytes] = {}
        if total > 0:
            if s.stop.is_set():
                raise Stopped()
            _stage(port, "downloading")
            data = driver.download(link, snap, part_dir, sn, _progress_logger(s, port))
        after = driver.after(link)
        log.debug("after download: %s", after)

        datasets = {}
        for name, blob in data.items():
            if not blob:
                continue
            fname = f"{stem}.bin" if list(data) == ["dataset1"] else f"{stem}_{name.replace('/', '_')}.bin"
            _write_bytes(rawdir / fname, blob)
            datasets[name] = {"file": f"raw/{fname}", "bytes": len(blob), "sha256": sha256(blob)}
        for f in driver.part_files(part_dir, sn) if part_dir.is_dir() else []:
            f.unlink(missing_ok=True)
        warnings = []
        if skew.get("n") and abs(skew["skew_vs_host_s"]) > 60:
            warnings.append(f"clock skew {skew['skew_vs_host_s']:+.1f} s exceeds 60 s "
                            "(clock reset, or set to local time instead of UTC?)")
        rem = _remaining(snap, after["meminfo"], after["power"], int(snap["sampling"].get("period", 0) or 0), driver)
        alarms = _time_checks(sn, "at offload", rem if after["status"] in ("logging", "pending", "gated") else None,
                              after["power"]["battery_voltage_V"], s)
        warnings.extend(alarms)
        record = {
            "tool": {"name": "rbr-tpw", "version": __version__},
            "port": port,
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
            "raw": datasets.get("dataset1") or next(iter(datasets.values()), None),
            "datasets": datasets,
            "transcript": f"raw/{stem}.log",
            "session_log": f"raw/{s.session_log.name}" if s.session_log else None,
            "warnings": warnings,
        }
        _write_json(rawdir / f"{stem}.json", record)  # saved before anything is written to the logger
        log.debug("wrote %s", rawdir / f"{stem}.json")
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
            report = _configure_step(link, s, snap, ntp, sn, note, driver)
            _write_json(rawdir / f"{stem}_configure.json", report)
            log.debug("wrote %s", rawdir / f"{stem}_configure.json")

    if not datasets:
        log.info("logger memory is empty; no NetCDF written")
        return None
    _stage(port, "writing NetCDF")
    nc_path = s.outdir / f"{stem}.nc"
    try:  # on the main thread: see console.Console.run_in_main
        all_warnings, t_ms = s.console.run_in_main(driver.write_netcdf, data, record, nc_path)
    except Exception as err:
        how = "no decoder yet" if isinstance(err, DecodeUnavailable) else f"{type(err).__name__}"
        log.warning("NetCDF not written (%s: %s). The download is saved; convert it later with: "
                    "rbr-offload %s --rebuild %s", how, err, s.outdir, rawdir / f"{stem}.json")
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
    except serial.SerialException as err:
        if "exclusively lock" in str(err):
            log.info("%s is in use by another program (another rbr-offload?); skipped", port)
        else:
            log.error("serial port error: %s", err)
            log.debug("traceback", exc_info=True)
    except Exception as err:
        log.error("%s: %s", type(err).__name__, err)
        log.debug("traceback", exc_info=True)
        log.info("A partial download resumes if you reconnect the logger. Details in %s",
                 s.session_log or "the serial transcript")
    finally:
        _stage(port, None)
        reason = _not_ready.pop(port, None)
        if reason:
            log.error("done with %s: disconnect it, but it is NOT READY TO DEPLOY: %s", port, reason)
        else:
            log.info("done with %s: disconnect the logger", port)


def run(s: Settings, once: bool, port: str | None):
    """Watch for loggers and give each its own worker thread. Main thread; answers the workers' questions."""
    workers: dict[str, threading.Thread] = {}
    finished: set[str] = set()  # ports whose worker ended, until they are unplugged
    announced = ruskin_warned = started_any = False
    last_status = time.monotonic()
    try:
        while True:
            present = _present(port)
            for p, t in list(workers.items()):
                if not t.is_alive():
                    del workers[p]
                    finished.add(p)
            for p in sorted(finished - present):
                finished.discard(p)
                log.info("%s disconnected", p)
            if once and started_any and not workers:
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
                t = threading.Thread(target=_worker, args=(p, s), name=f"offload-{_port_name(p)}", daemon=True)
                workers[p] = t
                t.start()
                started_any, announced = True, False
                log.debug("started %s for %s", t.name, p)
            if not workers and not announced:
                log.info("Waiting for an RBR logger on USB (Ctrl-C to quit)...")
                announced = True
            if len(workers) > 1 and time.monotonic() - last_status >= STATUS_EVERY_S:
                log.info("in progress: %s", _stage_summary())
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
        return
    log.info("Stopped.")


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
        except ConfigError as err:
            ap.error(str(err))
    elif overrides:
        ap.error("configuration options need --configure")

    args.outdir.mkdir(parents=True, exist_ok=True)
    console = Console()
    if args.rebuild:
        setup_logging(console)
        for rec_path in args.rebuild:
            record = json.loads(rec_path.read_text())
            driver = driver_for(int(record["id"]["fwtype"]))
            if driver is None:
                sys.exit(f"{rec_path}: fwtype {record['id']['fwtype']} is not supported")
            data = {}
            for name, info in (record.get("datasets") or {"dataset1": record["raw"]}).items():
                # recorded as raw/<file> relative to OUTDIR; also accept the file beside a moved record
                where = [rec_path.parent.parent / info["file"], rec_path.parent / Path(info["file"]).name]
                found = next((w for w in where if w.is_file()), None)
                if found is None:
                    sys.exit(f"{rec_path}: {info['file']} not found (looked in {', '.join(map(str, where))})")
                blob = found.read_bytes()
                if sha256(blob) != info["sha256"]:
                    sys.exit(f"{rec_path}: {info['file']} checksum does not match the record")
                data[name] = blob
            nc_path = args.outdir / (rec_path.stem + ".nc")
            try:
                driver.write_netcdf(data, record, nc_path)
            except DecodeUnavailable as err:
                log.warning("%s: not written: %s", rec_path, err)
                continue
            log.info("wrote %s", nc_path)
        return
    rawdir = args.outdir / "raw"
    rawdir.mkdir(exist_ok=True)
    session_log = rawdir / f"rbr-offload_{utcnow():%Y%m%dT%H%M%SZ}.log"
    setup_logging(console, session_log)
    s = Settings(outdir=args.outdir, ntp_server=None if args.no_ntp else args.ntp_server, deploy=deploy,
                 thresholds=thresholds, assume_yes=args.yes, console=console, session_log=session_log)
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


if __name__ == "__main__":
    main()
