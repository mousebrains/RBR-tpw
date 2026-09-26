"""RBRsolo (fwtype 9, "L2"-era compact logger) queries, clock-skew measurement, and download.

Every command here is read-only; nothing changes logger settings, the
schedule, the clock, or memory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import statistics
import time
from pathlib import Path

from .link import Link, LoggerError
from .rawbin import hexfloat

SUPPORTED_FWTYPES = {9}
CHUNK = 68000  # Ruskin's block size for these loggers

# Ruskin 2.26.1 PhysicalSolo.resetBatteryCounter() writes 33,696,000 (x1000 on the wire) for an
# RBRsolo "fresh battery": one AA 3.6 V x 2.6 Ah Li-SOCl2 cell = 33,696 J.
NOMINAL_BATTERY_J = 33_696.0
# Ruskin's L2EnableListener warns "insufficient power for the deployment" below this internal voltage.
RUSKIN_LOW_VOLTAGE_V = 2.0


def parse_logger_datetime(s: str) -> dt.datetime:
    return dt.datetime.strptime(s.strip(), "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)


def identify(link: Link) -> dict:
    link.wake()
    r = link.query("id")
    return {"model": r.get("model", ""), "version": r.get("version", ""),
            "serial": r.get("serial", ""), "fwtype": int(r.get("fwtype", "-1"))}


def _coefficient(v: str) -> float | str:
    v = v.strip()
    if len(v) == 8 and all(ch in "0123456789ABCDEFabcdef" for ch in v):
        return hexfloat(v)  # fwtype 9 reports IEEE-754 singles as hex
    try:
        return float(v)  # duet/concerto/Gen3: decimal, e.g. 3.4740720e-003
    except ValueError:
        return v  # e.g. `n1 = value` on derived channels


def snapshot(link: Link) -> dict:
    """Configuration, state, memory, and power, as reported by the logger."""
    snap = {"now": link.query("now")["now"], "status": link.query("status")["status"],
            "starttime": link.query("starttime")["starttime"], "endtime": link.query("endtime")["endtime"],
            "sampling": link.query("sampling"), "channels": link.query("channels")}
    channels = []
    for i in range(1, int(snap["channels"]["count"]) + 1):
        ch = link.query(f"channel {i}")
        cal = link.query(f"calibration {i}")
        ch["index"] = i
        ch["status"] = int(ch.get("status", "0") or 0)  # bit 0x04: not stored in memory (L3 ref; pyRSKtools)
        ch["calibration_datetime"] = cal.pop("datetime", "")
        cal.pop("type", None)
        ch["coefficients"] = {k: _coefficient(v) for k, v in cal.items()}
        ch["coefficients_as_reported"] = cal
        channels.append(ch)
    snap["channel_list"] = channels
    snap["meminfo"] = memory(link)
    snap["power"] = power(link)
    return snap


def memory(link: Link) -> dict:
    m = link.query("meminfo")
    return {k: int(v) for k, v in m.items()}


def power(link: Link) -> dict:
    """powerstatus (units not documented by RBR for fwtype 9):

    int        millivolts. Verified 2026-09-25: SN100689 read 3634-3650 against a meter's 3.665 V
               open-circuit on the same cell. Ruskin compares the converted value to 2.0 V.
    remaining  hex, millijoules. Ruskin parses it base-16 and divides by 1000; its resulting
               'used' for SN100689 (2220 J) is consistent with that reading. It is a counter
               the logger decrements by its own accounting, not a measurement of the cell,
               and it only means something if it was reset when a fresh cell went in. Units
               inferred from Ruskin, not verified.
    """
    p = link.query("powerstatus")
    out = {"source": p.get("source", ""), "int_raw": p.get("int", ""), "remaining_raw": p.get("remaining", "")}
    out.update({f"{k}_raw": v for k, v in p.items() if k not in ("source", "int", "remaining")})
    out["battery_voltage_V"] = parse_voltage(p.get("int", ""))
    try:
        out["energy_remaining_J"] = int(p["remaining"], 16) / 1000.0
    except (KeyError, ValueError):
        out["energy_remaining_J"] = float("nan")
    out["energy_nominal_J"] = NOMINAL_BATTERY_J
    out["energy_remaining_fraction"] = out["energy_remaining_J"] / NOMINAL_BATTERY_J
    return out


def parse_voltage(v: str) -> float:
    """`int` from powerstatus/power: millivolts on RBRsolos ("3634"), volts elsewhere ("3.61", "11.49")."""
    v = (v or "").strip()
    try:
        return float(v) if "." in v else int(v) / 1000.0
    except ValueError:
        return float("nan")


def _now(link: Link) -> str:
    return link.query("now")["now"]


def measure_clock_skew(link: Link, reps: int = 3, max_seconds: float = 15.0, clock=None) -> dict:
    """Logger clock minus host clock, resolved to a few ms.

    The logger reports whole seconds (`now`, or `clock(link)` for other families), so poll it
    back-to-back until the second increments. The tick lies between the send time of the last old
    reading and the receive time of the first new one; take the midpoint and report half the
    bracket as the uncertainty. Repeat `reps` times.
    """
    clock = clock or _now
    samples = []  # (skew_s, half_width_s)
    prev = None
    polls = 0
    t_end = time.monotonic() + max_seconds
    while len(samples) < reps and time.monotonic() < t_end:
        t_send = time.time_ns()
        value = clock(link)
        t_recv = time.time_ns()
        polls += 1
        logger_s = parse_logger_datetime(value).timestamp()
        if prev is not None and logger_s == prev[2] + 1:
            lo, hi = prev[0], t_recv
            mid = (lo + hi) / 2e9
            samples.append((logger_s - mid, (hi - lo) / 2e9))
            time.sleep(0.85)  # next tick is ~1 s away; stop hammering the logger
            prev = None
            continue
        prev = (t_send, t_recv, logger_s)
    if not samples:
        return {"n": 0, "polls": polls}
    skews = [s for s, _ in samples]
    return {
        "n": len(samples),
        "polls": polls,
        "skew_vs_host_s": statistics.median(skews),
        "uncertainty_s": max(h for _, h in samples),
        "spread_s": max(skews) - min(skews),
        "individual_s": skews,
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
    }


def download(link: Link, total: int, part_path: Path, progress=None, dataset: int = 1, l3: bool = False) -> bytes:
    """Read `total` bytes of `dataset` (Gen3 `readdata` if l3), appending to `part_path` so an
    interrupted download of the same deployment resumes where it stopped.

    A partial file is reused only if its header bytes match the logger's
    current header (same deployment).
    """
    head_len = min(512, total)
    head = link.read_data(dataset, head_len, 0, l3=l3)
    if part_path.exists():
        with open(part_path, "rb") as f:
            old_head = f.read(head_len)
        if old_head != head:
            link.note(f"partial file {part_path.name} is from a different deployment; restarting")
            part_path.unlink()
        else:
            link.note(f"resuming from {part_path.stat().st_size} bytes in {part_path.name}")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    with open(part_path, "ab") as f:
        offset = f.tell()
        if offset > total:  # logger memory shrank?  (should not happen without an erase)
            raise RuntimeError(f"partial file ({offset} B) is longer than logger memory ({total} B)")
        if offset == 0:
            f.write(head)
            offset = head_len
        t0, b0 = time.monotonic(), offset
        while offset < total:
            n = min(CHUNK, total - offset)
            block = link.read_data(dataset, n, offset, l3=l3)
            if len(block) != n:
                raise RuntimeError(f"short block at {offset}: {len(block)} of {n} bytes")
            f.write(block)
            f.flush()
            os.fsync(f.fileno())
            offset += n
            if progress:
                progress(offset, total, (offset - b0) / max(time.monotonic() - t0, 1e-6))
    image = part_path.read_bytes()
    if len(image) != total:
        raise RuntimeError(f"downloaded {len(image)} bytes, expected {total}")
    return image


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["SUPPORTED_FWTYPES", "LoggerError", "identify", "snapshot", "memory", "power",
           "measure_clock_skew", "download", "parse_logger_datetime", "parse_voltage", "sha256",
           "NOMINAL_BATTERY_J", "RUSKIN_LOW_VOLTAGE_V"]
