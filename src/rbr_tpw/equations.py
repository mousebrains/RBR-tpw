"""Calibration equations for raw Standard-format (rawbin00) readings, and the memory layout of L2 loggers
with a sectioned deployment header (RBRduet fwtype 102, RBRconcerto fwtype 103).

Equations (RBR L3 command reference rev L, 0005199revL, section 7):
  lin, qad, cub, tmp  7.1.1-7.1.4    output from R alone
  corr_pres2          7.3.3          pressure, corrected with the temperature of channel n0
  corr_cond           7.3.4          conductivity, corrected with temperature n0 and pressure n1
R = reading / 2**30, with readings signed 32-bit (section 5.3.2). Words 0xF6xxxxxx are error codes.

Memory layout (established 2026-09-25 on 16 duet/concerto .rsk files, Ruskin 2.26.1 as truth):
  - The deployment header is sectioned as in section 5.3.1, but in the older version 1.xxx (1009 on the
    duets, 1014 on the concertos): section 1 = metadata (version at byte 3, total header length at byte 7),
    section 2 = logger and deployment settings at the offsets of Ruskin's L2HeaderDeploymentVersionOffsets
    (relative to the section start, byte 9), section 3 = channels. The header ends with a big-endian
    CRC-16/CCITT of the bytes before it. Its length is not a multiple of 4.
  - Channel details (section 3): type (6 bytes), status (uint16, the channelStatus bits), calibration
    date (uint32, s since 2000), coefficient count (1 byte), then 4 bytes per coefficient: float32 c's and x's,
    int32 n's (0 = the default 'value'), in the order the `calibration` command lists them. Pressure
    channels carry one extra word after n0 (0x1111 or 0x8111) whose meaning is not known.
  - The body is 32-bit words: readings (one per stored channel per sample set), 0xF6 error codes, and
    events: 0xF7 (8 bytes, as on the RBRsolo), 0xF5 (12 bytes: CRC, type, marker, seconds, ms), and 0xF3
    (4*N bytes, N at byte 10, byte 11 bit 0 = "the next sample set takes this event's time", section
    5.3.3). Each event's CRC-16/CCITT covers its bytes 2..end, stored big-endian in bytes 0-1.
  - Sample sets are timed from the last time-anchoring event (0xF3 with bit 0 set; 0xF5/0xF7 time sync or
    restart) plus n * period. This reproduces Ruskin's timestamps exactly, including twist-activation gaps.
  - Ruskin's values are reproduced to <1e-13 only with Ruskin's coefficient values: the float32 coefficient
    printed as its shortest decimal and read back as a double (ruskin_coefficient()). The exact float32
    values differ by up to ~4e-6 dbar, and the logger's own `calibration` replies round to 8 digits.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from .crc import crc16_ccitt
from .rawbin import (
    EPOCH2000_MS,
    FEATURE_AVERAGING,
    FEATURE_BURSTING,
    RESTART_EVENTS,
    TFLAG_NO_ANCHOR,
    TFLAG_RESET_CLOCK,
    Decoded,
    Event,
    Header,
)

RATIO_SCALE = float(1 << 30)
ERROR_MARKER = 0xF6
STATUS_NOT_STORED = 0x04

# Default parameters ('value' in an n coefficient); the logger's `settings` command reports the real ones.
DEFAULTS = {"temperature": 15.0, "pressure": 10.1325}


def ratio(raw: np.ndarray) -> np.ndarray:
    """R = signed 32-bit reading / 2**30 (L3 ref section 5.3.2)."""
    return np.asarray(raw, np.uint32).view(np.int32).astype(np.float64) / RATIO_SCALE


def ruskin_coefficient(x: float) -> float:
    """A float32 coefficient as Ruskin 2.26.1 uses it for L2 loggers: shortest decimal, read back as a double."""
    return float(str(np.float32(x)))


def _is_value_ref(n) -> bool:
    if isinstance(n, str):
        try:
            n = float(n)
        except ValueError:
            return True  # "value"
    return int(n) <= 0  # Ruskin stores 'value' as -1, the memory header as 0


class _Problem(Exception):
    pass


def evaluate(raw: np.ndarray, channels: list[dict], defaults: dict | None = None
             ) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """raw: (n, nstored) uint32 words for the STORED channels in channel order (channels whose status has bit
    0x04 clear). channels: the FULL channel list in channel order, each dict with keys: index (1-based int),
    type (e.g. 'pres21'), equation (e.g. 'corr_pres2'), status (int), coefficients (dict name -> float or str,
    as parsed from the logger's `calibration N` reply / Ruskin's coefficients table; 'nX' entries may be
    channel indices or the string 'value'). `defaults` gives the parameters used for 'value' references
    (keys 'temperature', 'pressure'; the logger's `settings`), else DEFAULTS.
    Returns (values (n, nstored) float64 engineering units, bad (n, nstored) bool, problems {stored column:
    reason})."""
    raw = np.asarray(raw, np.uint32)
    if raw.ndim == 1:
        raw = raw[:, None]
    defaults = {**DEFAULTS, **(defaults or {})}
    stored = [c for c in channels if not int(c.get("status", 0)) & STATUS_NOT_STORED]
    if raw.shape[1] != len(stored):
        raise ValueError(f"raw has {raw.shape[1]} columns for {len(stored)} stored channels")
    column = {int(c["index"]): k for k, c in enumerate(stored)}
    by_index = {int(c["index"]): c for c in channels}
    n = raw.shape[0]
    cache: dict[int, np.ndarray] = {}
    active: set[int] = set()

    def reference(ch: dict, name: str, default_key: str) -> np.ndarray:
        coeffs = ch.get("coefficients", {})
        if name not in coeffs:
            raise _Problem(f"missing coefficient {name}")
        ref = coeffs[name]
        if _is_value_ref(ref):
            return np.full(n, float(defaults[default_key]))
        idx = int(float(ref))
        if idx not in column:
            raise _Problem(f"{name} refers to channel {idx}, which is not stored")
        return value(idx)

    def value(idx: int) -> np.ndarray:
        if idx in cache:
            return cache[idx]
        if idx in active:
            raise _Problem(f"circular channel reference at channel {idx}")
        active.add(idx)
        try:
            ch = by_index[idx]
            words = raw[:, column[idx]]
            v = _equation(ch, ratio(words), lambda name, key: reference(ch, name, key))
            v[(words >> 24) == ERROR_MARKER] = math.nan
            v[~np.isfinite(v)] = math.nan  # e.g. a zero denominator: no infinities reach the NetCDF file
            cache[idx] = v
            return v
        finally:
            active.discard(idx)  # also when this channel fails, so later channels are not blamed for a cycle

    values = np.full(raw.shape, math.nan)
    problems: dict[int, str] = {}
    for k, ch in enumerate(stored):
        try:
            values[:, k] = value(int(ch["index"]))
        except _Problem as err:
            problems[k] = f"channel {ch['index']} ({ch.get('type')}, {ch.get('equation')}): {err}"
            cache.pop(int(ch["index"]), None)
    return values, ~np.isfinite(values), problems


def _coefficient(ch: dict, name: str) -> float:
    try:
        return float(ch["coefficients"][name])
    except KeyError:
        raise _Problem(f"missing coefficient {name}") from None


def _equation(ch: dict, r: np.ndarray, ref) -> np.ndarray:
    eq = ch.get("equation", "")
    c = lambda i: _coefficient(ch, f"c{i}")  # noqa: E731
    x = lambda i: _coefficient(ch, f"x{i}")  # noqa: E731
    with np.errstate(all="ignore"):
        if eq == "lin":
            return c(0) + c(1) * r
        if eq == "qad":
            return c(0) + c(1) * r + c(2) * r**2
        if eq == "cub":
            return c(0) + c(1) * r + c(2) * r**2 + c(3) * r**3
        if eq == "tmp":
            ok = (r > 0) & (r < 1)
            xx = np.log(1.0 / np.where(ok, r, 0.5) - 1.0)
            t = 1.0 / (c(0) + c(1) * xx + c(2) * xx**2 + c(3) * xx**3) - 273.15
            t[~ok] = math.nan
            return t
        if eq == "corr_pres2":
            praw = c(0) + c(1) * r + c(2) * r**2 + c(3) * r**3
            dt = ref("n0", "temperature") - x(5)
            return x(0) + (praw - x(0) - x(1) * dt - x(2) * dt**2 - x(3) * dt**3) / (1 + x(4) * dt)
        if eq == "corr_cond":
            craw = c(0) + c(1) * r
            dt = ref("n0", "temperature") - x(3)
            return (craw - x(0) * dt) / (1 + x(1) * dt + x(2) * (ref("n1", "pressure") - x(4)))
    raise _Problem(f"equation {eq!r} is not implemented")


# --- L2 memory with a sectioned deployment header (header versions 1.xxx) -------------------------------------

# Offsets in section 2, relative to its first byte (Ruskin 2.26.1 L2HeaderDeploymentVersionOffsets).
L2_DEPLOYMENT_OFFSETS = {
    "fw_version": 3, "serial": 7, "logger_time": 11, "start_time": 15, "end_time": 19, "period_ms": 23,
    "output_format": 27, "status": 31, "baud": 35, "features": 39, "avg_interval": 43, "avg_count": 47,
    "burst_interval": 51, "burst_count": 55,
}
FEATURE_TIDE = 0x40
FEATURE_WAVE = 0x80
L2_UNSUPPORTED_FEATURES = FEATURE_AVERAGING | FEATURE_BURSTING | FEATURE_TIDE | FEATURE_WAVE
TIME_SYNC = 0x01


def is_sectioned_header(image: bytes) -> bool:
    """True for the sectioned header of L2 duets/concertos (and Gen3 rawbin00): metadata section first."""
    return len(image) >= 11 and image[0] == 0x01 and struct.unpack_from("<H", image, 1)[0] == 9


def parse_l2_header(image: bytes) -> Header:
    """Sectioned deployment header -> rawbin.Header (fields also has 'channels': the deployment's channels)."""
    if not is_sectioned_header(image):
        raise ValueError("not a sectioned deployment header")
    version = struct.unpack_from("<I", image, 3)[0]
    length = struct.unpack_from("<H", image, 7)[0]
    if not 11 < length <= len(image):
        raise ValueError(f"implausible header length {length} (image is {len(image)} bytes)")
    if crc16_ccitt(image[: length - 2]) != struct.unpack_from(">H", image, length - 2)[0]:
        raise ValueError("deployment header CRC mismatch")
    fields: dict = {"length": length, "version": version}
    off = 9
    while off < length - 2:
        sid, slen = image[off], struct.unpack_from("<H", image, off + 1)[0]
        if slen < 3 or off + slen > length - 2:
            raise ValueError(f"bad header section {sid} at byte {off} (length {slen})")
        if sid == 2:
            if version >= 2000:
                raise NotImplementedError(f"deployment header version {version} (Gen3 2.xxx) is not supported here")
            fields.update({k: struct.unpack_from("<I", image, off + rel)[0]
                           for k, rel in L2_DEPLOYMENT_OFFSETS.items()})
        elif sid == 3 and version < 2000:
            fields["channels"] = _l2_channels(image[off : off + slen])
        off += slen
    if "period_ms" not in fields:
        raise ValueError("deployment header has no deployment section")
    return Header(fields=fields, raw=bytes(image[:length]), trailer_event=None)


def _l2_channels(sec: bytes) -> list[dict]:
    nch = sec[3]
    starts = [struct.unpack_from("<H", sec, 4 + 2 * k)[0] for k in range(nch)]
    out = []
    for k, a in enumerate(starts):
        ctype = sec[a : a + 6].decode("ascii", "replace").rstrip("\x00")
        status, caldate, ncoef = struct.unpack_from("<HIB", sec, a + 6)
        words = sec[a + 13 : a + 13 + 4 * ncoef]
        out.append({"index": k + 1, "type": ctype, "status": status, "calibration_s2000": caldate,
                    "coefficient_words": [words[4 * i : 4 * i + 4] for i in range(ncoef)]})
    return out


def header_coefficients(ch: dict, names: list[str]) -> dict[str, float | int]:
    """Name a header channel's coefficient words (names in the order the `calibration` command lists them:
    c's, x's, then n's). c/x -> Ruskin-style doubles (ruskin_coefficient), n -> channel index (0 = 'value')."""
    out: dict[str, float | int] = {}
    for name, word in zip(names, ch["coefficient_words"], strict=False):
        if name.startswith("n"):
            out[name] = struct.unpack("<i", word)[0]
        else:
            out[name] = ruskin_coefficient(struct.unpack("<f", word)[0])
    return out


def decode_l2(image: bytes, nstored: int) -> Decoded:
    """Split an L2 sectioned-header memory image into sample sets and events and time the sample sets.

    Same result type and time rules as rawbin.decode (TFLAG_NO_ANCHOR before the first anchor, TFLAG_RESET_CLOCK
    for anchors before the enable time), so rawbin.resolve_times and the NetCDF writer apply unchanged.
    Event times in Decoded.events are whole seconds (rawbin.Event has no ms); sample times keep the ms."""
    if nstored < 1:
        raise ValueError("nstored must be >= 1")
    hdr = parse_l2_header(image)
    if hdr.features & L2_UNSUPPORTED_FEATURES:
        raise NotImplementedError(f"sampling features 0x{hdr.features:X} (average/burst/tide/wave) are not supported")
    period = hdr.period_ms
    if not 0 < period < 86_400_000:
        raise ValueError(f"implausible sampling period {period} ms")
    body = image[hdr.length :]
    nwords = len(body) // 4
    words = np.frombuffer(body[: 4 * nwords], "<u4")
    top = words >> 24

    is_reading = np.ones(nwords, bool)
    events: list[Event] = []
    anchors: list[tuple[int, int, int]] = []  # (sample index, unix ms, seconds since 2000)
    i_prev = readings_before = bad_markers = 0
    for i in np.flatnonzero((top == 0xF3) | (top == 0xF5) | (top == 0xF7)):
        i = int(i)
        if i < i_prev:
            continue
        b = 4 * i
        marker = int(top[i])
        size = 8 if marker == 0xF7 else 12 if marker == 0xF5 else (4 * body[b + 10] if b + 10 < len(body) else 0)
        if size < 8 or b + size > len(body):
            if b + size > len(body):
                is_reading[i:] = False  # truncated event at the very end
                break
            continue
        rec = body[b : b + size]
        if struct.unpack_from(">H", rec, 0)[0] != crc16_ccitt(rec[2:]):
            bad_markers += 1  # signed readings can start 0xF3/0xF5/0xF7, so this may be a reading: keep it
            continue
        readings_before += int(is_reading[i_prev:i].sum())
        is_reading[i : i + size // 4] = False
        etype, seconds = rec[2], struct.unpack_from("<I", rec, 4)[0]
        ms = struct.unpack_from("<H", rec, 8)[0] if marker != 0xF7 else 0
        index = readings_before // nstored
        events.append(Event(offset=hdr.length + b, type=etype, seconds=seconds, crc_ok=True, sample_index=index))
        anchors_next = bool(rec[11] & 1) if marker == 0xF3 else (etype == TIME_SYNC or etype in RESTART_EVENTS)
        if anchors_next:
            anchors.append((index, EPOCH2000_MS + 1000 * seconds + ms, seconds))
        i_prev = i + size // 4

    readings = words[is_reading]
    nsets = readings.size // nstored
    trailing = 4 * (readings.size - nsets * nstored) + (len(body) - 4 * nwords)
    raw = readings[: nsets * nstored].reshape(nsets, nstored)
    flags = np.zeros(raw.shape, np.uint8)
    flags[(raw >> 24) == ERROR_MARKER] |= 1  # rawbin.FLAG_ERROR_CODE

    time_ms = np.empty(nsets, np.int64)
    time_flags = np.zeros(nsets, np.uint8)
    segment = np.full(nsets, -1, np.int32)
    first = anchors[0][0] if anchors else nsets
    if first > 0:
        t0 = EPOCH2000_MS + 1000 * max(hdr.start_time, hdr.logger_time)
        time_ms[:first] = t0 + period * np.arange(first, dtype=np.int64)
        time_flags[:first] |= TFLAG_NO_ANCHOR
    for k, (idx, t_ms, seconds) in enumerate(anchors):
        end = anchors[k + 1][0] if k + 1 < len(anchors) else nsets
        time_ms[idx:end] = t_ms + period * np.arange(end - idx, dtype=np.int64)
        segment[idx:end] = k
        if seconds < hdr.logger_time:
            time_flags[idx:end] |= TFLAG_RESET_CLOCK
    return Decoded(header=hdr, nchan=nstored, raw=raw, flags=flags, time_ms=time_ms, time_flags=time_flags,
                   segment=segment, events=events, trailing_bytes=trailing,
                   bad_event_words=bad_markers)
