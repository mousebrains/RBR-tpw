"""Decode the Standard ("rawbin00"-style) memory image of an RBRsolo, fwtype 9.

Layout, as established on 2026-09-25 from real downloads and cross-checked
against Ruskin 2.26.1 (header offsets from its SL2HeaderVersionOffsets class;
sample values reproduced to 6e-14 degC against Ruskin .rsk files):

  bytes 0..(length-1)  header, little-endian uint32 fields (see HEADER_FIELDS);
                       unused words are 0xFFFFFFFF. Bytes 504..511 hold an
                       event-like record that Ruskin does not treat as data.
  bytes length..       stream of 32-bit little-endian words:
                         sample   raw ADC reading, one word per channel
                         0xF6...  per-reading error code (L3 ref section 5.3.2)
                         0xF7...  8-byte event: <crc16 BE of bytes 2..7><type><0xF7><uint32 LE seconds since 2000>

Event type codes match the L3 command reference, section 5.3.3.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from .crc import crc16_ccitt

EPOCH2000_MS = 946_684_800_000  # 2000-01-01T00:00:00Z in Unix ms
RATIO_SCALE = float(1 << 30)  # raw reading / 2**30 = voltage ratio R

HEADER_FIELDS = {  # name: byte offset of a little-endian uint32
    "length": 0,
    "version": 4,  # firmware version * 1000
    "serial": 8,
    "logger_time": 12,  # logger clock when the deployment was enabled, s since 2000
    "start_time": 16,  # programmed start, s since 2000
    "end_time": 20,  # programmed end, s since 2000
    "period_ms": 24,
    "sampling_rate": 28,
    "status": 32,
    "baud": 36,
    "features": 40,
    "avg_interval": 44,
    "avg_count": 48,
    "burst_interval": 52,
    "burst_count": 56,
    "burst_altitude": 60,
}
FEATURE_AVERAGING = 16
FEATURE_BURSTING = 32
FEATURE_FAST_BURST = 1024
FEATURE_FAST_WAVE = 2048
UNSUPPORTED_FEATURES = FEATURE_AVERAGING | FEATURE_BURSTING | FEATURE_FAST_BURST | FEATURE_FAST_WAVE

EVENT_MARKER = 0xF7
ERROR_MARKER = 0xF6
EVENT_TIME_SYNC = 0x01

EVENT_NAMES = {  # L3 command reference rev L, section 5.3.3
    0x00: "unknown_event",
    0x01: "time_synchronization_marker",
    0x02: "disable_command_received",
    0x03: "runtime_error",
    0x04: "cpu_reset_detected",
    0x05: "parameters_recovered_after_reset",
    0x06: "restart_failed_rtc_invalid",
    0x07: "restart_failed_logger_status_invalid",
    0x08: "restart_failed_schedule_not_recovered",
    0x09: "unable_to_load_alarm_time",
    0x0A: "sampling_restarted_after_rtc_reset",
    0x0B: "parameters_recovered_sampling_restarted_after_rtc_reset",
    0x0C: "deployment_end_time_reached",
    0x0D: "start_of_recorded_burst",
    0x0E: "start_of_wave_burst",
    0x0F: "power_source_switched_to_usb",
    0x10: "streaming_off_both_ports",
    0x11: "streaming_on_usb_off_serial",
    0x12: "streaming_off_usb_on_serial",
    0x13: "streaming_on_both_ports",
    0x14: "sampling_started_threshold_condition_satisfied",
    0x15: "sampling_paused_threshold_condition_not_met",
    0x16: "power_source_switched_to_internal_battery",
    0x17: "power_source_switched_to_external_battery",
    0x18: "twist_activation_started_sampling",
    0x19: "twist_activation_paused_sampling",
    0x1A: "wifi_module_activated",
    0x1B: "wifi_module_deactivated",
    0x1C: "regimes_enabled_not_yet_in_regime",
    0x1D: "entered_regime_1",
    0x1E: "entered_regime_2",
    0x1F: "entered_regime_3",
    0x20: "start_of_regime_bin",
    0x21: "begin_profiling_up_cast",
    0x22: "begin_profiling_down_cast",
    0x23: "end_of_profiling_cast",
    0x24: "battery_failed_schedule_finished",
    0x25: "directional_sampling_fast_mode",
    0x26: "directional_sampling_slow_mode",
    0x27: "energy_used_marker_internal_battery",
    0x28: "energy_used_marker_external_power",
    0x29: "device_control_action_result",
}
RTC_RESET_EVENTS = {0x06, 0x0A, 0x0B}
RESTART_EVENTS = {0x0A, 0x0B}  # sampling restarted on a reset clock: the next sample set is re-anchored here
EVENT_CPU_RESET = 0x04
# Events whose timestamp is the time of the next sample set (8-byte events carry no "affects the next sample" flag,
# so this is by type): the time-sync marker, the restarts on a reset clock, and a CPU reset with the clock intact.
# The last: RBRsolo SN076315 (fwtype 0), bench 2026-09-26, browned out at 2.4 V on 2026-09-23, stopped sampling,
# and on USB power 3.5 days later wrote `CPU reset detected` (RTC still running, +5 s) and resumed. Counted from
# the previous sync marker, the samples after it would be dated 3.5 days early. Ruskin's convention for this event
# is unverified (no .rsk in hand has one); it is applied by analogy with 0x0A/0x0B. Not a clock-segment boundary.
ANCHOR_EVENTS = {EVENT_TIME_SYNC, EVENT_CPU_RESET} | RESTART_EVENTS

# per-reading flag bits
FLAG_ERROR_CODE = 1  # 0xF6 error code from the logger
FLAG_OUT_OF_RANGE = 2  # raw value cannot be converted (R outside (0, 1))

# per-sample-set time flag bits
TFLAG_NO_ANCHOR = 1  # precedes any time anchor; time taken from the header start time
TFLAG_RESET_CLOCK = 2  # anchored on a clock that had been reset (anchor earlier than the deployment's enable time)
TFLAG_SKEW_CORRECTED = 4  # set by the NetCDF writer: reset-clock time shifted by the skew measured at offload


def event_name(code: int) -> str:
    """CF-safe name for an event type code; codes above 0xFF are Ruskin's own, not the logger's."""
    if code in EVENT_NAMES:
        return EVENT_NAMES[code]
    return f"ruskin_event_{code}" if code > 0xFF else f"type_0x{code:02X}"


@dataclass
class Header:
    fields: dict
    raw: bytes
    trailer_event: tuple | None  # (type, seconds since 2000) from bytes 504..511, if present

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass
class Event:
    offset: int  # byte offset in the memory image
    type: int
    seconds: int  # seconds since 2000-01-01
    crc_ok: bool
    sample_index: int  # index of the next sample set
    size: int = 8  # bytes of the event record in the image

    @property
    def name(self) -> str:
        return event_name(self.type)

    @property
    def unix_ms(self) -> int:
        return EPOCH2000_MS + 1000 * self.seconds


@dataclass
class Decoded:
    header: Header
    nchan: int
    raw: np.ndarray  # (n, nchan) uint32 words as stored
    flags: np.ndarray  # (n, nchan) uint8, FLAG_* bits
    time_ms: np.ndarray  # (n,) int64 Unix ms on the logger's clock
    time_flags: np.ndarray  # (n,) uint8, TFLAG_* bits
    segment: np.ndarray  # (n,) int32 index of the time anchor each sample set hangs off; -1 = none
    events: list[Event] = field(default_factory=list)
    trailing_bytes: int = 0  # bytes after the last whole word or partial sample set
    bad_event_words: int = 0  # event-marker words whose record failed its CRC
    # (n,) int64 byte offset just past each sample set's last word: an offload whose image held `b` bytes had seen
    # sample set i iff set_end_byte[i] <= b
    set_end_byte: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def rtc_reset(self) -> bool:
        return (any(e.type in RTC_RESET_EVENTS for e in self.events)
                or bool((self.time_flags & TFLAG_RESET_CLOCK).any()))


def parse_header(image: bytes) -> Header:
    if len(image) < 64:
        raise ValueError(f"memory image too short for a header: {len(image)} bytes")
    fields = {k: struct.unpack_from("<I", image, off)[0] for k, off in HEADER_FIELDS.items()}
    length = fields["length"]
    if not 64 <= length <= len(image) or length % 4:
        raise ValueError(f"implausible header length {length} (image is {len(image)} bytes)")
    trailer = None
    if length >= 512:
        w0, ts = struct.unpack_from("<II", image, 504)
        if w0 >> 24 == EVENT_MARKER:
            trailer = ((w0 >> 16) & 0xFF, ts)
    return Header(fields=fields, raw=bytes(image[:length]), trailer_event=trailer)


def _event_crc_ok(rec: bytes) -> bool:
    # CRC of bytes 2..7, stored big-endian in bytes 0..1 (verified on 5 real events)
    return struct.unpack_from(">H", rec, 0)[0] == crc16_ccitt(rec[2:8])


def decode(image: bytes, nchan: int) -> Decoded:
    """Split a memory image into sample sets and events and assign sample times.

    Timing: each time-sync event (type 0x01), and each "sampling restarted
    after RTC reset" event (0x0A/0x0B) or "CPU reset detected" event (0x04), sets the time of the next sample set;
    later sets follow at the header's sampling period. This reproduces
    Ruskin's timestamps exactly in the normal case. Times are the logger's
    clock. Anchors earlier than the deployment's enable time are on a clock
    that was reset (power loss) and are flagged TFLAG_RESET_CLOCK.
    """
    if nchan < 1:
        raise ValueError("nchan must be >= 1")
    hdr = parse_header(image)
    if hdr.features & UNSUPPORTED_FEATURES:
        raise NotImplementedError(f"sampling features 0x{hdr.features:X} (averaging/burst/wave) are not supported yet")
    period = hdr.period_ms
    if not 0 < period < 86_400_000:
        raise ValueError(f"implausible sampling period {period} ms")

    body = image[hdr.length:]
    nwords = len(body) // 4
    words = np.frombuffer(body[: 4 * nwords], "<u4")
    top = words >> 24

    # Walk only the candidate event words; everything else is a reading.
    is_reading = np.ones(nwords, bool)
    events: list[Event] = []
    bad_event_words = []
    i_prev_end = 0
    readings_before = 0
    for i in np.flatnonzero(top == EVENT_MARKER):
        i = int(i)
        if i < i_prev_end:  # timestamp word of the previous event
            continue
        if i + 1 >= nwords:
            is_reading[i:] = False  # truncated event at the very end
            break
        rec = body[4 * i : 4 * i + 8]
        if not _event_crc_ok(rec):
            # An RBRsolo reading is R * 2**30 with 0 < R < 1 (top byte <= 0x3F), so a 0xF7 word is an event
            # whose record is corrupt. Drop both of its words: kept as readings they would add two false
            # samples (the second, a timestamp, converts to a plausible temperature) and shift later times.
            bad_event_words.append(i)
            readings_before += int(is_reading[i_prev_end:i].sum())
            is_reading[i : i + 2] = False
            i_prev_end = i + 2
            continue
        readings_before += int(is_reading[i_prev_end:i].sum())
        is_reading[i : i + 2] = False
        events.append(Event(offset=hdr.length + 4 * i, type=rec[2], seconds=struct.unpack_from("<I", rec, 4)[0],
                            crc_ok=True, sample_index=readings_before // nchan))
        i_prev_end = i + 2

    readings = words[is_reading]
    nsets = readings.size // nchan
    trailing = 4 * (readings.size - nsets * nchan) + (len(body) - 4 * nwords)
    raw = readings[: nsets * nchan].reshape(nsets, nchan)
    set_end_byte = set_end_bytes(hdr.length, is_reading, nchan, nsets)

    flags = np.zeros(raw.shape, np.uint8)
    flags[(raw >> 24) == ERROR_MARKER] |= FLAG_ERROR_CODE

    # Sample times: anchor at each time-sync or restart event.
    time_ms = np.empty(nsets, np.int64)
    time_flags = np.zeros(nsets, np.uint8)
    segment = np.full(nsets, -1, np.int32)
    anchors = [e for e in events if e.type in ANCHOR_EVENTS]
    first = anchors[0].sample_index if anchors else nsets
    if first > 0:  # sets before any anchor: fall back to the programmed start
        t0 = EPOCH2000_MS + 1000 * max(hdr.start_time, hdr.logger_time)
        time_ms[:first] = t0 + period * np.arange(first, dtype=np.int64)
        time_flags[:first] |= TFLAG_NO_ANCHOR
    for k, e in enumerate(anchors):
        idx = e.sample_index
        end = anchors[k + 1].sample_index if k + 1 < len(anchors) else nsets
        time_ms[idx:end] = e.unix_ms + period * np.arange(end - idx, dtype=np.int64)
        segment[idx:end] = k
        if e.seconds < hdr.logger_time:
            time_flags[idx:end] |= TFLAG_RESET_CLOCK

    return Decoded(header=hdr, nchan=nchan, raw=raw, flags=flags, time_ms=time_ms, time_flags=time_flags,
                   segment=segment, events=events, trailing_bytes=trailing, bad_event_words=len(bad_event_words),
                   set_end_byte=set_end_byte)


def scan_events(image: bytes, start: int = 0) -> list[tuple[int, int, int]]:
    """The valid 8-byte event records in `image` from byte `start` on, as (offset, type, seconds since 2000),
    without decoding the samples (a quick look at what an offload's new bytes hold). Assumes word alignment."""
    out = []
    lo = max(start, 0) - max(start, 0) % 4
    body = image[lo : len(image) - (len(image) - lo) % 4]
    words = np.frombuffer(body, "<u4")
    i = 0
    for i in np.flatnonzero((words >> 24) == EVENT_MARKER).tolist():
        rec = body[4 * i : 4 * i + 8]
        if len(rec) == 8 and _event_crc_ok(rec):
            out.append((lo + 4 * i, rec[2], struct.unpack_from("<I", rec, 4)[0]))
    return out


def set_end_bytes(body_offset: int, is_reading: np.ndarray, nchan: int, nsets: int) -> np.ndarray:
    """Byte offset just past the last word of each of the `nsets` sample sets, given which words are readings."""
    last = np.flatnonzero(is_reading)[nchan - 1 :: nchan][:nsets]
    return body_offset + 4 * (last.astype(np.int64) + 1)


def tmp_equation(raw: np.ndarray, c: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """RBR 'tmp' (Steinhart-Hart) equation, L3 ref section 7.1.4. Returns (degC, out_of_range mask)."""
    r = raw.astype(np.int64).astype(np.float64) / RATIO_SCALE
    bad = ~((r > 0) & (r < 1)) | ((raw >> 24) == ERROR_MARKER)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(1.0 / np.where(bad, 0.5, r) - 1.0)
        y = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
        t = 1.0 / y - 273.15
    bad |= ~np.isfinite(t)  # e.g. the polynomial passing through zero near R -> 1
    t[bad] = math.nan
    return t, bad


EQUATIONS = {"tmp": tmp_equation}


def hexfloat(s: str) -> float:
    """Decode the logger's 8-hex-digit IEEE-754 single-precision coefficient format."""
    return struct.unpack(">f", bytes.fromhex(s.strip()))[0]


RESET_CLOCK_BEFORE_MS = 978_307_200_000  # 2001-01-01T00:00:00Z: an RBR clock restarts at 2000-01-01 after a power loss


def clock_runs(time_ms: np.ndarray) -> np.ndarray:
    """Start index of each run of samples between backward steps of a per-sample logger clock (a restart)."""
    return np.concatenate(([0], np.flatnonzero(np.diff(time_ms) < 0) + 1)).astype(np.int64)


EVENT_JITTER_MS = 60_000  # events in acquisition order may step back a little: 1.7 s seen in a Gen3 .rsk


def event_indices(time_ms: np.ndarray, event_ms: list[int]) -> list[int]:
    """Index of the sample each event precedes, for events listed in acquisition order (Gen3 dataset 0, Gen4
    events, .rsk rowid order). np.searchsorted over the whole record is wrong once the clock has gone back: each
    event is placed within one run of increasing sample times. It moves to a later run when it is on the reset
    clock and either follows samples on a set clock or steps back by more than EVENT_JITTER_MS (a restart);
    smaller steps back are ordinary event jitter."""
    n = time_ms.size
    if not n:
        return [0] * len(event_ms)
    starts = [int(i) for i in clock_runs(time_ms)] + [n]
    last = len(starts) - 2
    r, prev, out = 0, None, []
    for e in event_ms:
        if prev is not None and e < min(prev - EVENT_JITTER_MS, RESET_CLOCK_BEFORE_MS) and r < last:
            r += 1
        while e < RESET_CLOCK_BEFORE_MS <= time_ms[starts[r]] and r < last:
            r += 1
        a, b = starts[r], starts[r + 1]
        if time_ms[a] < RESET_CLOCK_BEFORE_MS <= e:
            i = a  # a set-clock event before a run on the reset clock
        elif e < RESET_CLOCK_BEFORE_MS <= time_ms[a]:
            i = b  # the clock restarted after this run and no sample followed
        else:
            i = a + int(np.searchsorted(time_ms[a:b], e))
        out.append(i)  # not forced to increase: an event's own time places it, and events jitter (see above)
        prev = e
    return out


@dataclass
class SegmentStart:
    """Where a clock segment (a run of sample sets between clock restarts) begins."""

    sample_index: int
    event_index: int | None = None  # position in the event list of the restart event that starts it, if any


@dataclass
class OffloadView:
    """What one offload saw of the record: the sample sets and events its image held, when it happened, and the
    clock skew it measured (logger minus UTC, s; None when not measured)."""

    samples_seen: int
    events_seen: int
    unix_ms: int
    skew_s: float | None
    clock_set_before: bool = False  # the skew jumped since the previous offload with no reset event to explain it

    def saw(self, start: SegmentStart) -> bool:
        if start.event_index is not None:
            return start.event_index < self.events_seen
        return start.sample_index < self.samples_seen


def segments_from_starts(n: int, starts: list[SegmentStart]) -> np.ndarray:
    """Per sample set, how many segment starts precede it (starts at 0 or beyond the end do not count)."""
    seg = np.zeros(n, np.int32)
    for s in starts:
        if 0 < s.sample_index < n:
            seg[s.sample_index :] += 1
    return seg


def _starts(n: int, time_ms: np.ndarray, event_starts: list[tuple[int, int]]) -> list[SegmentStart]:
    """Segment starts from restart events (sample index, event index) and from backward steps of the clock."""
    starts: dict[int, SegmentStart] = {}
    for index, k in event_starts:
        if 0 < index < n and index not in starts:
            starts[index] = SegmentStart(index, k)
    if n:
        for i in clock_runs(time_ms)[1:].tolist():
            starts.setdefault(int(i), SegmentStart(int(i)))
    return [starts[i] for i in sorted(starts)]


def reset_segment_starts(time_ms: np.ndarray, events: list[tuple[int, int, int]]) -> list[SegmentStart]:
    """For loggers that timestamp every sample: a new clock segment after each restart event (`events` are
    (ms, type, sample index)) or backward step of the clock."""
    return _starts(time_ms.size, time_ms,
                   [(index, k) for k, (_, etype, index) in enumerate(events) if etype in RESTART_EVENTS])


def reset_segments(time_ms: np.ndarray, events: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """For loggers that timestamp every sample (Ruskin values, Gen3 EasyParse, Gen4): TFLAG_RESET_CLOCK on
    samples dated before 2001, and a new clock segment after each restart event or backward step of the clock
    (so a missing or misplaced restart event cannot join two reset runs). `events` are (ms, type, sample
    index)."""
    tflags = np.where(time_ms < RESET_CLOCK_BEFORE_MS, TFLAG_RESET_CLOCK, 0).astype(np.uint8)
    return tflags, segments_from_starts(time_ms.size, reset_segment_starts(time_ms, events))


def resolve_times(
    d: Decoded, skew_s: float | None = None, offload_unix_ms: int | None = None,
    views: list[OffloadView] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Best-estimate UTC sample times.

    Sample sets on a reset clock (TFLAG_RESET_CLOCK) are shifted by the logger-minus-UTC skew measured at an
    offload made while that clock ran unbroken: the latest offload whose current clock segment it was (`views`,
    one per offload of the deployment; a single offload is `skew_s` at `offload_unix_ms`). The result is checked
    to land before that offload, after the preceding samples and after 2001. Reset-clock sets that no offload
    can time are dropped (the raw image keeps them).

    "Segment" here means the samples between clock restarts (events 0x06/0x0A/0x0B, or a backward step of the
    sample times), not between time anchors: an ordinary anchor (a time sync, or a twist activation on an
    RBRconcerto) does not reset the clock, so samples on either side of it share one clock and one correction.

    Returns (time_ms, time_flags, keep mask, notes).
    """
    return resolve_time_arrays(d.time_ms, d.time_flags, None, skew_s, offload_unix_ms, views=views,
                               starts=clock_segment_starts(d))


def clock_segment_starts(d: Decoded) -> list[SegmentStart]:
    """Clock segment starts of a decoded L2 image: each RTC reset event, and each backward step of the times."""
    return _starts(d.time_ms.size, d.time_ms,
                   [(e.sample_index, k) for k, e in enumerate(d.events) if e.type in RTC_RESET_EVENTS])


def clock_segments(d: Decoded) -> np.ndarray:
    """Per sample set, how many clock restarts (RTC_RESET_EVENTS, or a backward step of the sample times)
    precede it."""
    return segments_from_starts(d.time_ms.size, clock_segment_starts(d))


def resolve_time_arrays(
    time_ms: np.ndarray, time_flags: np.ndarray, segment: np.ndarray | None, skew_s: float | None = None,
    offload_unix_ms: int | None = None, *, views: list[OffloadView] | None = None,
    starts: list[SegmentStart] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """resolve_times() on plain arrays: logger-clock times, TFLAG_* bits, and either the clock-segment numbers
    or the segment starts. One offload (`skew_s`, `offload_unix_ms`) or several (`views`)."""
    t = time_ms.copy()
    tf = time_flags.copy()
    n = t.size
    keep = np.ones(n, bool)
    notes: list[str] = []
    if views is None:
        views = [OffloadView(samples_seen=n, events_seen=1 << 62, unix_ms=int(offload_unix_ms or 0), skew_s=skew_s)]
    if starts is None:  # from the segment numbers: sample-index markers only
        starts = [SegmentStart(int(i)) for i in (np.flatnonzero(np.diff(segment) != 0) + 1)] if n > 1 else []
    # A clock set between two offloads (a sync in Ruskin, only possible while stopped) ends every clock run at what
    # the earlier offload saw: its skew is the last one measured on that clock.
    by_index = {s.sample_index: s for s in starts}
    for before, v in zip(views, views[1:], strict=False):
        if v.clock_set_before and 0 < before.samples_seen < n:
            by_index.setdefault(before.samples_seen, SegmentStart(before.samples_seen))
    starts = [by_index[i] for i in sorted(by_index) if 0 < i < n]
    bounds = [0, *[s.sample_index for s in starts], n]
    # each offload's current clock segment: the last one whose start it had seen
    current = [max([j for j, s in enumerate(starts, 1) if v.saw(s)], default=0) for v in views]
    reset = (tf & TFLAG_RESET_CLOCK) != 0
    dropped = 0
    several = len(views) > 1
    for s_idx in range(len(bounds) - 1):
        i0, i1 = bounds[s_idx], bounds[s_idx + 1]
        fix = np.zeros(n, bool)
        fix[i0:i1] = reset[i0:i1]
        if not fix.any():
            continue
        idx = np.flatnonzero(fix)
        j0 = int(idx[0])
        cands = [k for k, c in enumerate(current) if c == s_idx]
        applied = inconsistent = False
        for k in reversed(cands):  # the latest offload on this clock first, then fall back
            v = views[k]
            if v.skew_s is None or not math.isfinite(v.skew_s):
                continue
            shift = int(round(1000 * v.skew_s))
            shifted = t[fix] - shift
            prev_ok = j0 == 0 or reset[j0 - 1] or shifted[0] > t[j0 - 1]
            # the last of these samples that offload saw must land before the offload (plus 5 s of slack)
            seen = idx[idx < v.samples_seen]
            seen_ok = not seen.size or int(t[seen[-1]]) - shift <= v.unix_ms + 5000
            # still before 2001 after the shift: the skew was measured on a clock set since the reset
            if seen_ok and prev_ok and shifted[0] >= RESET_CLOCK_BEFORE_MS:
                t[fix] = shifted
                tf[fix] |= TFLAG_SKEW_CORRECTED
                notes.append(f"{int(fix.sum())} samples after a clock reset were re-timed by subtracting the "
                             f"offload clock skew ({v.skew_s:+.3f} s"
                             + (f", offload {k + 1} of {len(views)})" if several else ")"))
                applied = True
                break
            inconsistent = True
        if not applied:
            keep &= ~fix
            dropped += int(fix.sum())
            if inconsistent:
                notes.append(f"{int(fix.sum())} samples after a clock reset dropped: correcting them with the "
                             "offload skew does not give a consistent time")
            elif cands:
                notes.append(f"{int(fix.sum())} samples after a clock reset dropped: no clock skew was measured")
    if dropped:
        notes.append(f"{dropped} samples on a reset clock with no recoverable time are omitted "
                     "(still present in the raw file)")
    tk = t[keep]
    bad = np.flatnonzero(np.diff(tk) <= 0) if tk.size > 1 else []
    if len(bad):  # a CF time coordinate must be strictly monotonic: do not publish duplicate or reversed times
        i = int(bad[0])
        raise ValueError(f"sample times are not strictly increasing after clock corrections ({len(bad)} step(s); "
                         f"first after sample {i}: {int(tk[i])} -> {int(tk[i + 1])} ms); not writing a NetCDF "
                         "(the raw data is unchanged)")
    return t, tf, keep, notes
