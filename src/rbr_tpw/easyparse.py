"""Decode Gen3 (fwtype 104, e.g. RBRconcerto3) EasyParse ("calbin00") memory, L3 command reference rev L §5.

The datasets are downloaded separately:
  dataset 1  sample sets: uint64 LE ms since 1970 (logger clock) + one float32 LE per stored channel, already
             in engineering units (calibrated, compensated, derived channels included). A reading the logger
             could not produce is a NaN whose bit pattern is 0xFF8xxxxx + error code (§5.2.1).
  dataset 0  events, 16 bytes each: <CRC16 BE of bytes 2..15><type><0xF4><uint64 LE ms><uint32 LE payload> (§5.2.2).
  dataset 2  deployment header: sections <id><uint16 LE length incl. id and length><content>, then a CRC (§5.3.1).

Verified 2026-09-25 against Ruskin 2.26.1 on SN233442 (firmware 1.158 and 1.162) and SN243188 (1.161):
  - dataset 1 = the `downloads` table of 10 EPdesktop .rsk files (6 stored channels, 32-byte records);
    decoded timestamps and values equal Ruskin's `data` table exactly (float32 widened to float64);
  - event layout and CRC on the 58 event records Ruskin's serial log dumped (dataset 0 reads);
  - header fields listed in parse_deployment_header() on the 41 header dumps in Ruskin's serial log
    (first 128 bytes of each read).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .crc import crc16_ccitt

EVENT_SIZE = 16
EVENT_MARKER = 0xF4
NAN_ERROR_BASE = 0xFF800000  # IEEE-754 -NaN with a zero payload; error code = pattern - base


@dataclass
class EasyParse:
    time_ms: np.ndarray  # (n,) int64 logger-clock ms since 1970 from each sample record
    values: np.ndarray  # (n, nchan) float64 engineering units; NaN where the logger stored a NaN error code
    error_codes: np.ndarray  # (n, nchan) uint32: 0, or the raw 0xFF8xxxxx NaN pattern (L3 ref 5.2.1 error table)
    events: list[tuple[int, int, int]] = field(default_factory=list)  # (logger-clock ms, type code, payload)
    trailing_bytes: int = 0  # bytes after the last whole record in dataset 1
    bad_events: int = 0  # dataset-0 records that failed the CRC or marker check (not in `events`)


def record_dtype(nchan: int) -> np.dtype:
    return np.dtype([("t", "<u8"), ("v", "<f4", (nchan,))]) if nchan > 1 else np.dtype([("t", "<u8"), ("v", "<f4")])


def decode_easyparse(data1: bytes, nchan: int, data0: bytes | None = None) -> EasyParse:
    """Split dataset 1 into timestamped sample sets of `nchan` float32 values (and dataset 0 into events).

    Raises ValueError if the timestamps show that `nchan` is wrong (they must never decrease).
    """
    if nchan < 1:
        raise ValueError("nchan must be >= 1")
    size = 8 + 4 * nchan
    n = len(data1) // size
    rec = np.frombuffer(data1[: n * size], dtype=record_dtype(nchan))
    t = rec["t"].astype(np.int64)
    raw_f32 = rec["v"].reshape(n, nchan)
    if n > 1 and (np.diff(t) < 0).any():
        bad = int(np.flatnonzero(np.diff(t) < 0)[0])
        raise ValueError(f"timestamps decrease at sample {bad + 1} of {n}: is nchan={nchan} right "
                         f"(record size {size} bytes)?")
    bits = raw_f32.view("<u4")
    isnan = np.isnan(raw_f32)
    error_codes = np.where(isnan, bits, 0).astype(np.uint32)
    with np.errstate(invalid="ignore"):  # 0xFF81xxxx error codes are signalling NaNs; they stay NaN
        values = raw_f32.astype(np.float64)
    events, bad = decode_events(data0) if data0 is not None else ([], 0)
    return EasyParse(time_ms=t, values=values, error_codes=error_codes, events=events,
                     trailing_bytes=len(data1) - n * size, bad_events=bad)


def event_ok(rec: bytes) -> bool:
    """16-byte dataset-0 record: marker 0xF4 and CRC-16/CCITT-FALSE of bytes 2..15 stored big-endian in bytes 0..1."""
    return len(rec) == EVENT_SIZE and rec[3] == EVENT_MARKER and struct.unpack_from(">H", rec)[0] == crc16_ccitt(
        rec[2:])


def decode_events(data0: bytes) -> tuple[list[tuple[int, int, int]], int]:
    """(events as (logger-clock ms, type code, payload), number of records that failed the check)."""
    events, bad = [], 0
    for i in range(0, len(data0) - EVENT_SIZE + 1, EVENT_SIZE):
        rec = data0[i : i + EVENT_SIZE]
        if not event_ok(rec):
            bad += 1
            continue
        ms, payload = struct.unpack_from("<QI", rec, 4)
        events.append((int(ms), rec[2], payload))
    return events, bad


def _cstr(b: bytes) -> str:
    return b.split(b"\x00", 1)[0].split(b"\xff", 1)[0].decode("ascii", errors="replace")


def parse_deployment_header(data2: bytes) -> dict:
    """Best-effort parse of the dataset-2 deployment header (L3 ref §5.3.1).

    Verified fields (Ruskin serial-log dumps, SN233442/243188): header_version, header_length, fwtype,
    firmware_version, serial, model, part_number, power_supply_part_number, dataset_format, enabled_s2000,
    start_s2000, end_s2000, period_ms, status_code, feature_flags, burst_interval_ms, burst_length.
    Later deployment-section fields (thresholding, regimes, WiFi, UTC offset, battery types, ...) and the
    channels section are not parsed: their layout could not be checked. The header CRC is not checked.
    Times are seconds since 2000-01-01 on the logger's clock. Returns {"sections": {id: length}, ...}.
    """
    out: dict = {"sections": {}}
    i = 0
    while i + 3 <= len(data2):
        sid, length = data2[i], struct.unpack_from("<H", data2, i + 1)[0]
        if length < 3 or sid == 0xFF:
            break
        out["sections"][sid] = length
        body = data2[i + 3 : i + length]
        if sid == 0x01 and len(body) >= 6:
            out["header_version"], out["header_length"] = struct.unpack_from("<IH", body)
        elif sid == 0x02 and len(body) >= 30:
            out["fwtype"], out["firmware_version"], out["serial"] = struct.unpack_from("<III", body)
            out["model"] = _cstr(body[12:28])
            pn_len = struct.unpack_from("<H", body, 28)[0]
            out["part_number"] = _cstr(body[30 : 30 + pn_len])
            j = 30 + pn_len
            if len(body) >= j + 2:
                ps_len = struct.unpack_from("<H", body, j)[0]
                out["power_supply_part_number"] = _cstr(body[j + 2 : j + 2 + ps_len])
        elif sid == 0x03 and len(body) >= 36:
            names = ("dataset_format", "enabled_s2000", "start_s2000", "end_s2000", "period_ms", "status_code",
                     "feature_flags", "burst_interval_ms", "burst_length")
            out.update(zip(names, struct.unpack_from("<9I", body), strict=True))
            out["dataset_format_name"] = {0: "rawbin00", 1: "calbin00"}.get(out["dataset_format"], "?")
        i += length
    return out
