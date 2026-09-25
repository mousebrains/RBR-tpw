"""rbr-offload: wait for RBR loggers on USB, measure clock skew, download, write CF NetCDF, repeat.

Read-only toward the logger unless --configure is given: the logger is left
in whatever state it was found (still logging, if it was).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from serial.tools import list_ports

from . import __version__
from .configure import FAR_FUTURE, PAST, ConfigError, DeployConfig, configure
from .hostclock import ntp_offset
from .link import Link, LinkError
from .ncwrite import write_netcdf
from .power import remaining
from .solo import (
    RUSKIN_LOW_VOLTAGE_V,
    SUPPORTED_FWTYPES,
    download,
    identify,
    measure_clock_skew,
    memory,
    power,
    sha256,
    snapshot,
)

POLL_S = 0.5


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


def rbr_ports() -> set[str]:
    return {p.device for p in list_ports.comports()
            if p.device.startswith("/dev/cu.") and ("RBR" in (p.manufacturer or "") or "RBR" in (p.product or ""))}


def ruskin_running() -> bool:
    # exact process-name match; "-f" would also match any shell whose command line mentions Ruskin
    r = subprocess.run(["pgrep", "-x", "Ruskin"], capture_output=True, check=False)
    return r.returncode == 0


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def alarm(lines: list[str], s: Settings):
    """Loud, hard-to-miss warning: bell, red banner, and (interactively) an acknowledgement."""
    on = sys.stdout.isatty()
    red, reset = ("\033[1;97;41m", "\033[0m") if on else ("", "")
    width = max(len(x) for x in lines) + 6
    print("\a", end="")
    print(f"{red}{'!' * width}{reset}")
    for x in lines:
        print(f"{red}!! {x.ljust(width - 6)} !!{reset}")
    print(f"{red}{'!' * width}{reset}", flush=True)
    if not s.assume_yes and sys.stdin.isatty():
        try:
            input("  press Enter to acknowledge ")
        except EOFError:
            pass


def _progress(done: int, total: int, rate: float):
    eta = (total - done) / rate if rate > 0 else math.inf
    eta_s = f"{int(eta // 60)}:{int(eta % 60):02d}" if math.isfinite(eta) else "?"
    print(f"\r  downloading {done / 1e6:8.2f} / {total / 1e6:.2f} MB ({100 * done / total:5.1f}%) "
          f"{rate / 1e3:6.1f} kB/s  ETA {eta_s}   ", end="", flush=True)


def _write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_transcript(path: Path, link: Link):
    with open(path, "w") as f:
        for e in link.transcript:
            t = dt.datetime.fromtimestamp(e.t_ns / 1e9, dt.UTC)
            f.write(f"{iso(t)}  {e.direction:4s}  {e.text}\n")


def _active_ms(snap: dict) -> float:
    ch = snap["channels"]
    return float(ch.get("latency", 0) or 0) + float(ch.get("readtime", 0) or 0)


def _remaining(snap: dict, mem: dict, pwr: dict, period_ms: int) -> dict | None:
    if not period_ms:
        return None
    return remaining(pwr["energy_remaining_J"], mem["used"], mem["remaining"], period_ms,
                     len(snap["channel_list"]), _active_ms(snap))


def _describe(rem: dict) -> str:
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


def _configure_step(link: Link, s: Settings, snap: dict, ntp: dict, sn: str, erase_note: str) -> dict:
    cfg = s.deploy
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
    print(f"  configure SN{sn}: " + "; ".join(plan))
    if not s.assume_yes:
        try:
            ok = input("  proceed? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            ok = False
        if not ok:
            print("  configuration skipped; logger unchanged.")
            return {"skipped": True}
    try:
        report = configure(link, int(sn), cfg, ntp.get("offset_s") or 0.0)
    except (ConfigError, LinkError) as err:
        print(f"  CONFIGURE FAILED: {err}")
        return {"error": str(err)}
    rb = report["readback"]
    clk = rb["clock"]
    off = ntp.get("offset_s") or 0.0
    print(f"  now: status {rb['status']}, {rb['sampling'].get('mode')} {rb['sampling'].get('period')} ms, "
          f"start {rb['starttime']}, end {rb['endtime']}, memory used {rb['meminfo']['used']} B, "
          f"battery {rb['power']['battery_voltage_V']:.3f} V / {rb['power']['energy_remaining_J']:.0f} J, "
          f"clock {1e3 * (clk['skew_vs_host_s'] - off):+.1f} ms vs UTC")

    rem = _remaining(snap, rb["meminfo"], rb["power"], int(rb["sampling"].get("period", 0) or 0))
    report["remaining_time"] = rem
    if rem:
        print(f"  expected sampling time: {_describe(rem)}")
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
    started = utcnow()
    rawdir = s.outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    with Link(port) as link:
        ident = identify(link)
        sn = ident["serial"]
        print(f"{ident['model']} SN{sn}, firmware {ident['version']}, fwtype {ident['fwtype']} on {port}")
        if ident["fwtype"] not in SUPPORTED_FWTYPES:
            print(f"  fwtype {ident['fwtype']} is not supported yet (supported: {sorted(SUPPORTED_FWTYPES)}); "
                  "skipping, nothing was changed on the logger.")
            return None

        ntp = ntp_offset(s.ntp_server) if s.ntp_server else {"error": "disabled"}
        skew = measure_clock_skew(link)
        if skew.get("n"):
            off = ntp.get("offset_s")
            vs_utc = skew["skew_vs_host_s"] - off if off is not None else skew["skew_vs_host_s"]
            print(f"  clock skew (logger - {'UTC' if off is not None else 'host'}): {vs_utc:+.3f} s "
                  f"+/- {skew['uncertainty_s'] + (ntp.get('uncertainty_s') or 0):.3f} s (n={skew['n']})")
        else:
            print("  clock skew: could not be measured")

        snap = snapshot(link)
        total = snap["meminfo"]["used"]
        print(f"  status {snap['status']}, sampling {snap['sampling'].get('mode')} "
              f"{snap['sampling'].get('period')} ms, {total} bytes to download")

        stem = f"{sn}_{started.strftime('%Y%m%dT%H%M%SZ')}"
        part = rawdir / ".partial" / f"{sn}.bin.part"
        image = None
        if total > 0:
            image = download(link, total, part, _progress)
            print()
        after = {"meminfo": memory(link), "power": power(link), "status": link.query("status")["status"]}

        raw_path = rawdir / f"{stem}.bin"
        if image is not None:
            os.replace(part, raw_path)
        warnings = []
        if skew.get("n") and abs(skew["skew_vs_host_s"]) > 60:
            warnings.append(f"clock skew {skew['skew_vs_host_s']:+.1f} s exceeds 60 s "
                            "(clock reset, or set to local time instead of UTC?)")
        rem = _remaining(snap, after["meminfo"], after["power"], int(snap["sampling"].get("period", 0) or 0))
        alarms = _time_checks(sn, "at offload", rem if after["status"] in ("logging", "pending") else None,
                              after["power"]["battery_voltage_V"], s)
        warnings.extend(alarms)
        record = {
            "tool": {"name": "rbr-tpw", "version": __version__},
            "port": port,
            "offload_started": iso(started),
            "offload_finished": iso(utcnow()),
            "id": ident,
            "host_ntp": ntp,
            "clock_skew": skew,
            "snapshot_before": snap,
            "after": after,
            "remaining_time": rem,
            "thresholds": vars(s.thresholds),
            "raw": ({"file": f"raw/{raw_path.name}", "bytes": len(image), "sha256": sha256(image)}
                    if image is not None else None),
            "warnings": warnings,
        }
        _write_json(rawdir / f"{stem}.json", record)  # saved before anything is written to the logger
        mem, pwr = after["meminfo"], after["power"]
        print(f"  at offload, memory: {mem['remaining'] / 1e6:.1f} of {mem['size'] / 1e6:.1f} MB free "
              f"({100 * mem['remaining'] / mem['size']:.1f}%); battery {pwr['battery_voltage_V']:.3f} V, "
              f"energy counter {pwr['energy_remaining_J']:.0f} J of {pwr['energy_nominal_J']:.0f} J nominal")
        if rem:
            print(f"  at offload, sampling time left at {snap['sampling'].get('period')} ms: {_describe(rem)}")
        if alarms:
            alarm(alarms, s)
        if s.deploy is not None:
            note = f"{total} bytes, saved to raw/{raw_path.name}" if image is not None else "already empty"
            report = _configure_step(link, s, snap, ntp, sn, note)
            _write_json(rawdir / f"{stem}_configure.json", report)
        _write_transcript(rawdir / f"{stem}.log", link)

    if image is None:
        print("  logger memory is empty; no NetCDF written")
        return None
    nc_path = s.outdir / f"{stem}.nc"
    _, all_warnings, t_ms = write_netcdf(image, record, nc_path)
    print(f"  {len(t_ms)} samples written"
          + (f", {iso(dt.datetime.fromtimestamp(t_ms[0] / 1e3, dt.UTC))} to "
             f"{iso(dt.datetime.fromtimestamp(t_ms[-1] / 1e3, dt.UTC))}" if len(t_ms) else ""))
    for w in all_warnings:
        if w not in alarms:
            print(f"  WARNING: {w}")
    print(f"  wrote {nc_path}")
    return nc_path


def wait_loop(s: Settings, once: bool, port: str | None):
    handled: set[str] = set()
    announced = ruskin_warned = False
    while True:
        present = {port} & rbr_ports() if port else rbr_ports()
        handled &= present
        new = sorted(present - handled)
        if not new:
            if not announced:
                print("\nWaiting for an RBR logger on USB (Ctrl-C to quit)...", flush=True)
                announced = True
            time.sleep(POLL_S)
            continue
        if ruskin_running():
            if not ruskin_warned:
                print("Ruskin is running and polls every RBR port it sees; quit Ruskin to continue.", flush=True)
                ruskin_warned = True
            time.sleep(2)
            continue
        ruskin_warned = False
        dev = new[0]
        time.sleep(1.0)  # let USB enumeration settle
        try:
            offload(dev, s)
        except KeyboardInterrupt:
            raise
        except Exception as err:
            print(f"\n  ERROR: {err}")
            traceback.print_exc(file=sys.stdout)
            print("  A partial download resumes if you reconnect the logger.")
        handled.add(dev)
        print(f"\nDone with {dev}. Disconnect the logger" + (" and connect the next one." if not once else "."),
              flush=True)
        while dev in rbr_ports():
            time.sleep(POLL_S)
        print("Disconnected.")
        announced = False
        if once:
            return


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rbr-offload", description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path, help="directory for NetCDF files (raw downloads go in OUTDIR/raw)")
    ap.add_argument("--once", action="store_true", help="handle one logger, then exit")
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
    if args.rebuild:
        for rec_path in args.rebuild:
            record = json.loads(rec_path.read_text())
            image = (rec_path.parent.parent / record["raw"]["file"]).read_bytes()
            if sha256(image) != record["raw"]["sha256"]:
                sys.exit(f"{rec_path}: raw file checksum does not match the record")
            nc_path = args.outdir / (rec_path.stem + ".nc")
            write_netcdf(image, record, nc_path)
            print(f"wrote {nc_path}")
        return
    s = Settings(outdir=args.outdir, ntp_server=None if args.no_ntp else args.ntp_server, deploy=deploy,
                 thresholds=thresholds, assume_yes=args.yes)
    try:
        wait_loop(s, args.once, args.port)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
