"""Write a decoded logger download as a CF-1.13 NetCDF-4 file.

Built from two inputs so it can be re-run offline: the raw memory image and
the JSON offload record (logger config, clock skew, memory, power).
write_values_netcdf() writes the same layout from engineering values computed
elsewhere (Ruskin's .rsk `data` table), without the raw readings.
"""

from __future__ import annotations

import datetime as dt
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import netCDF4
import numpy as np

from . import __version__
from .equations import decode_l2, evaluate, header_coefficients, is_sectioned_header, parse_l2_header
from .rawbin import (
    EQUATIONS,
    EVENT_NAMES,
    FLAG_ERROR_CODE,
    FLAG_OUT_OF_RANGE,
    RESET_CLOCK_BEFORE_MS,
    RESTART_EVENTS,
    RTC_RESET_EVENTS,
    TFLAG_NO_ANCHOR,
    TFLAG_RESET_CLOCK,
    TFLAG_SKEW_CORRECTED,
    Decoded,
    OffloadView,
    TimeCorrection,
    decode,
    event_name,
    reset_segment_starts,
    resolve_time_arrays,
    resolve_times,
)

TIME_UNITS = "milliseconds since 1970-01-01 00:00:00"

# RBR channel type prefix -> (variable name, CF standard_name, units, long_name); first match wins
CHANNEL_KINDS = {
    "temp": ("temperature", "sea_water_temperature", "degree_Celsius", "Temperature"),
    "cond": ("conductivity", "sea_water_electrical_conductivity", "mS cm-1", "Conductivity"),
    "pres08": ("sea_pressure", "sea_water_pressure_due_to_sea_water", "dbar", "Sea pressure"),
    "pres": ("pressure", "sea_water_pressure", "dbar", "Pressure"),
    "dpth": ("depth", "depth", "m", "Depth"),
    "sal_": ("salinity", "sea_water_practical_salinity", "1", "Practical salinity"),
    "sos_": ("speed_of_sound", "speed_of_sound_in_sea_water", "m s-1", "Speed of sound"),
    "scon": ("specific_conductivity", None, "uS cm-1", "Specific conductivity"),
}

SOLO_BATTERY_COMMENT = ("From `powerstatus` after the download. battery_voltage_V = int / 1000 (mV). "
                        "battery_energy_remaining_J = hex(remaining) / 1000 (mJ); this is the logger's own "
                        "energy counter, meaningful only if reset (Ruskin 'Fresh battery') when the cell "
                        "was replaced. Nominal = one AA 3.6 V 2.6 Ah Li-SOCl2 cell (Ruskin's reset value). "
                        "Voltage units verified against a meter; energy-counter units inferred from Ruskin "
                        "2.26.1 behaviour, not from RBR documentation.")

TIME_COMMENT = ("Logger clock, except samples flagged time_corrected_by_offload_skew in time_flag (taken after a "
                "clock reset), which are logger clock minus the clock skew measured at an offload of this "
                "deployment made while that clock ran: time_correction_offload names the offload for each re-timed "
                "run (an index along offload_time; its skew is clock_skew there, or clock_skew_vs_host when the "
                "host clock was not referenced to UTC). Not otherwise corrected for clock skew.")


def _iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _s2000_iso(seconds: int) -> str:
    return _iso(946_684_800_000 + 1000 * int(seconds))


def _parse_iso_ms(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def skew_vs_utc(record: dict) -> float | None:
    """Logger minus UTC (s) from the record; falls back to logger minus host if NTP failed."""
    skew = record.get("clock_skew", {})
    if not skew.get("n"):
        return None
    return skew["skew_vs_host_s"] - (record.get("host_ntp", {}).get("offset_s") or 0.0)


def records_of(record) -> list[dict]:
    """The offload records of a deployment in order: a list as given, or one record."""
    return list(record) if isinstance(record, list | tuple) else [record]


def image_bytes_of(record: dict) -> int | None:
    """How many bytes of the (primary) memory image an offload's record describes; None if it does not say."""
    dep = record.get("deployment") or {}
    if dep.get("image_bytes") is not None:
        return int(dep["image_bytes"])
    raw = record.get("raw") or {}
    return int(raw["bytes"]) if raw.get("bytes") is not None else None


def _offload_ms(record: dict) -> int:
    return _parse_iso_ms(record.get("offload_finished") or record["offload_started"])


SKEW_JUMP_S = 2.0  # a skew change between offloads beyond this plus the drift allowance means the clock was set
SKEW_DRIFT_PPM = 50.0


def skew_jump(previous: dict, record: dict) -> tuple[float, float, float] | None:
    """(skew change s, tolerance s, days apart) between two offload records' clock skews, or None when they
    cannot be compared: a skew not measured, or one against UTC and the other against the host clock (their
    difference would be the host clock's offset, not the logger's). The tolerance is SKEW_JUMP_S plus
    SKEW_DRIFT_PPM of the time between them plus both measurements' uncertainties."""
    def skew_of(rec):
        sk, ntp = rec.get("clock_skew") or {}, rec.get("host_ntp") or {}
        if not sk.get("n"):
            return None
        s = skew_vs_utc(rec)
        if s is None or not math.isfinite(s):
            return None
        unc = _num(sk.get("uncertainty_s"), 0.0) + (_num(ntp.get("uncertainty_s"), 0.0) if ntp.get("offset_s")
                                                    is not None else 0.0)
        return s, ntp.get("offset_s") is not None, unc if math.isfinite(unc) else 0.0
    a, b = skew_of(previous), skew_of(record)
    if a is None or b is None or a[1] != b[1]:
        return None
    dt_s = max(0.0, (_offload_ms(record) - _offload_ms(previous)) / 1000)
    return b[0] - a[0], SKEW_JUMP_S + SKEW_DRIFT_PPM * 1e-6 * dt_s + a[2] + b[2], dt_s / 86400


def offload_views(records: list[dict], seen: list[tuple[int, int]], reset_between) -> list[OffloadView]:
    """One OffloadView per record. `seen[k]` = (sample sets, events) offload k's image held; `reset_between(a, b)`
    says whether a clock-reset event sits among events a..b-1, which explains a skew jump."""
    views: list[OffloadView] = []
    for k, (rec, (samples, events)) in enumerate(zip(records, seen, strict=True)):
        clock_set = False
        if k:
            jump = skew_jump(records[k - 1], rec)
            clock_set = (jump is not None and abs(jump[0]) > jump[1]
                         and not reset_between(views[-1].events_seen, events))
        views.append(OffloadView(samples, events, _offload_ms(rec), skew_vs_utc(rec), clock_set))
    return views


def _l2_seen(records: list[dict], d: Decoded) -> list[tuple[int, int]]:
    out = []
    for rec in records:
        b = image_bytes_of(rec)
        if b is None:
            out.append((d.time_ms.size, len(d.events)))
        else:
            out.append((int(np.searchsorted(d.set_end_byte, b, side="right")),
                        sum(1 for e in d.events if e.offset + e.size <= b)))
    return out


def _num(x, default=math.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def channel_kind(ctype: str, order: int) -> tuple[str, str | None, str | None, str]:
    """(variable name, CF standard_name, CF units, long_name) for an RBR channel type such as temp09."""
    kind = next((v for p, v in CHANNEL_KINDS.items() if ctype.startswith(p)), None)
    return kind or (f"channel{order:02d}", None, None, f"Channel {order} ({ctype})")


def _unique_name(name: str, order: int, used: set[str]) -> str:
    if name in used:
        name = f"{name}{order:02d}"
    used.add(name)
    return name


@contextmanager
def _atomic_dataset(path: Path) -> Iterator[netCDF4.Dataset]:
    """A NETCDF4 file written under a temporary name, fsynced, then renamed onto `path`."""
    tmp = path.with_name(path.name + ".tmp")
    nc = netCDF4.Dataset(tmp, "w", format="NETCDF4")
    try:
        yield nc
    except BaseException:
        nc.close()
        tmp.unlink(missing_ok=True)
        raise
    nc.close()
    with open(tmp, "r+b") as f:  # fsync needs a writable handle on Windows
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _time_variables(nc: netCDF4.Dataset, t_utc: np.ndarray, logger_ms: np.ndarray, tflags: np.ndarray,
                    timing: str) -> tuple[int]:
    """Create the time dimension and time, logger_time, time_flag; returns the chunk shape."""
    n = t_utc.size
    nc.createDimension("time", n)
    chunk = (max(1, min(n, 1 << 18)),)
    tv = nc.createVariable("time", "i8", ("time",), zlib=True, complevel=4, chunksizes=chunk)
    tv.setncatts({"standard_name": "time", "long_name": "sample time (UTC)",
                  "units": TIME_UNITS, "calendar": "standard", "units_metadata": "leap_seconds: none",
                  "axis": "T", "comment": f"{TIME_COMMENT} {timing}"})
    tv[:] = t_utc
    lt = nc.createVariable("logger_time", "i8", ("time",), zlib=True, complevel=4, chunksizes=chunk)
    lt.setncatts({"long_name": "sample time on the logger's clock, as recorded", "units": TIME_UNITS,
                  "calendar": "standard", "units_metadata": "leap_seconds: none"})
    lt[:] = logger_ms
    tq = nc.createVariable("time_flag", "u1", ("time",), zlib=True, complevel=4, chunksizes=chunk)
    tq.setncatts({"long_name": "sample time quality flags",
                  "flag_masks": np.array([TFLAG_NO_ANCHOR, TFLAG_RESET_CLOCK, TFLAG_SKEW_CORRECTED], "u1"),
                  "flag_meanings": "no_time_anchor_before_sample logger_clock_had_been_reset "
                                   "time_corrected_by_offload_skew"})
    tq[:] = tflags
    return chunk


def _flag_variable(nc: netCDF4.Dataset, name: str, long: str, flags: np.ndarray, chunk: tuple[int],
                   comment: str | None = None):
    fv = nc.createVariable(f"{name}_flag", "u1", ("time",), zlib=True, complevel=4, chunksizes=chunk)
    atts = {"long_name": f"{long} quality flags",
            "flag_masks": np.array([FLAG_ERROR_CODE, FLAG_OUT_OF_RANGE], "u1"),
            "flag_meanings": "logger_error_code conversion_out_of_range"}
    if comment:
        atts["comment"] = comment
    fv.setncatts(atts)
    fv[:] = flags


ENERGY_MARKER_EVENTS = {0x27, 0x28}


def _event_variables(nc: netCDF4.Dataset, times_ms: list[int], types: list[int], index: list[int],
                     payloads: list[int] | None = None):
    """Event list; `index` is the position along time of the first sample after each event. `payloads` are the
    event records' 32-bit payloads (Gen3 EasyParse, Gen4), from which the energy markers' joules are decoded."""
    nc.createDimension("event", len(times_ms))
    codes = np.array(sorted(set(EVENT_NAMES) | set(types)), "u2")
    et = nc.createVariable("event_time", "i8", ("event",))
    et.setncatts({"long_name": "event time on the logger's clock", "units": TIME_UNITS, "calendar": "standard",
                  "units_metadata": "leap_seconds: none"})
    ey = nc.createVariable("event_type", "u2", ("event",))
    ey.setncatts({"long_name": "logger event type", "flag_values": codes,
                  "flag_meanings": " ".join(event_name(int(c)) for c in codes),
                  "comment": "RBR event type codes, L3 command reference section 5.3.3. Codes above 255 are "
                             "Ruskin's own annotations, not logger events."})
    ei = nc.createVariable("event_sample_index", "i8", ("event",))
    ei.setncatts({"long_name": "index along time of the first sample after the event", "units": "1"})
    if times_ms:
        et[:] = times_ms
        ey[:] = types
        ei[:] = index
    if payloads is not None:
        ep = nc.createVariable("event_payload", "u8", ("event",))
        ep.setncatts({"long_name": "event record payload, as stored", "units": "1",
                      "comment": "The payload of the event record: 32 bits on Gen3 EasyParse (L3 command "
                                 "reference section 5.2.2), e.g. a sample address for cast events or the energy "
                                 "accumulator for energy markers (see energy_used_marker); the 8-byte auxiliary "
                                 "word on Gen4."})
        em = nc.createVariable("energy_used_marker", "f8", ("event",), fill_value=np.nan)
        em.setncatts({"long_name": "energy used from the power source since the accumulator was last reset, "
                                   "as the logger recorded it in an energy-used marker event", "units": "J",
                      "comment": "Set on energy_used_marker_internal_battery (0x27) and _external_power (0x28) "
                                 "events only: the payload read as an IEEE-754 single. Equal to `powerinternal "
                                 "used` on 2 RBRconcerto3 loggers (2026-09-28); written at enable and then about "
                                 "every 33.8 h."})
        if times_ms:
            pl = np.array(payloads, np.uint64)
            ep[:] = pl
            joules = (pl & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32).astype(np.float64)
            marker = np.isin(np.array(types), list(ENERGY_MARKER_EVENTS)) & (pl < np.uint64(1 << 32))
            em[:] = np.where(marker, joules, np.nan)


def _string_variable(nc: netCDF4.Dataset, name: str, values: list[str], long: str):
    v = nc.createVariable(name, str, ("offload_time",))
    v.long_name = long
    for i, x in enumerate(values):
        v[i] = str(x)


def _offload_series(nc: netCDF4.Dataset, records: list[dict], views: list[OffloadView]):
    """Per-offload state of the logger along `offload_time`: what the global attributes say for the latest
    offload, for every offload of the deployment."""
    n = len(records)
    nc.createDimension("offload_time", n)

    def var(name, dtype, values, **atts):
        v = nc.createVariable(name, dtype, ("offload_time",), fill_value=np.nan if dtype == "f8" else None)
        v.setncatts(atts)
        v[:] = values
        return v

    after = [r.get("after") or {} for r in records]
    snap = [r.get("snapshot_before") or {} for r in records]
    mem = [a.get("meminfo") or s.get("meminfo") or {} for a, s in zip(after, snap, strict=True)]
    pwr = [a.get("power") or s.get("power") or {} for a, s in zip(after, snap, strict=True)]
    rem = [r.get("remaining_time") or {} for r in records]
    skew = [r.get("clock_skew") or {} for r in records]
    ntp = [r.get("host_ntp") or {} for r in records]
    dep = [r.get("deployment") or {} for r in records]
    vs_utc = [sk["skew_vs_host_s"] - nt["offset_s"] if sk.get("n") and nt.get("offset_s") is not None else math.nan
              for sk, nt in zip(skew, ntp, strict=True)]
    unc = [sk["uncertainty_s"] + (nt.get("uncertainty_s") or 0.0)
           if sk.get("n") and nt.get("offset_s") is not None else math.nan for sk, nt in zip(skew, ntp, strict=True)]
    held = [image_bytes_of(r) or 0 for r in records]
    var("offload_time", "i8", [_parse_iso_ms(r["offload_started"]) for r in records], standard_name="time",
        long_name="offload start time (UTC)", units=TIME_UNITS, calendar="standard",
        units_metadata="leap_seconds: none")
    var("offload_image_bytes", "i8", held, long_name="bytes of logger memory held after this offload", units="byte")
    var("offload_segment_bytes", "i8", [int((d.get("segment") or {}).get("bytes", h)) for d, h in
                                        zip(dep, held, strict=True)],
        long_name="bytes added to the saved memory image at this offload", units="byte")
    var("offload_bytes_read", "i8", [int(d.get("bytes_read", (d.get("segment") or {}).get("bytes", h)))
                                     for d, h in zip(dep, held, strict=True)],
        long_name="bytes read from the logger at this offload (checks and full reads included)", units="byte")
    var("offload_samples", "i8", [v.samples_seen for v in views],
        long_name="sample sets held after this offload", units="1")
    var("battery_voltage", "f8", [_num(p.get("battery_voltage_V")) for p in pwr], units="V",
        long_name="internal battery voltage at offload (read on USB power: an unloaded cell)")
    var("battery_energy_remaining", "f8", [_num(p.get("energy_remaining_J")) for p in pwr], units="J",
        long_name="logger energy counter at offload (its own accounting, not a measurement; NaN if it has none)")
    var("memory_used", "i8", [int(m.get("used", 0) or 0) for m in mem], units="byte",
        long_name="logger memory used at offload")
    var("memory_remaining", "i8", [int(m.get("remaining", 0) or 0) for m in mem], units="byte",
        long_name="logger memory remaining at offload")
    var("clock_skew", "f8", vs_utc, units="s", long_name="logger clock minus UTC at offload",
        comment="UTC = host clock + host_ntp_offset; NaN when the host clock was not referenced to UTC")
    var("clock_skew_uncertainty", "f8", unc, units="s",
        long_name="clock skew uncertainty (worst tick half-bracket plus NTP uncertainty)")
    var("clock_skew_vs_host", "f8", [_num(sk.get("skew_vs_host_s")) if sk.get("n") else math.nan for sk in skew],
        units="s", long_name="logger clock minus host clock at offload")
    var("host_ntp_offset", "f8", [_num(nt.get("offset_s")) for nt in ntp], units="s",
        long_name="UTC minus host clock, from SNTP")
    var("host_ntp_uncertainty", "f8", [_num(nt.get("uncertainty_s")) for nt in ntp], units="s",
        long_name="SNTP offset uncertainty")
    var("sampling_days_remaining", "f8", [_num(x.get("days")) for x in rem], units="day",
        long_name="sampling time left at offload (lesser of memory-limited and modelled energy-limited)")
    var("energy_days_remaining_modelled", "f8", [_num(x.get("energy_days")) for x in rem], units="day",
        long_name="modelled energy-limited sampling days left at offload")
    var("clock_set_detected", "u1", [int(v.clock_set_before) for v in views], units="1",
        long_name="clock set between this offload and the previous one",
        flag_values=np.array([0, 1], "u1"), flag_meanings="no_jump skew_jump_without_reset_event",
        comment=f"1 when the skew changed by more than {SKEW_JUMP_S:g} s plus {SKEW_DRIFT_PPM:g} ppm of the "
                "elapsed time since the previous offload, with no clock-reset event in the new data to explain it.")
    var("header_changed", "i4", [len(d.get("header_changed") or []) for d in dep], units="1",
        long_name="memory header bytes that changed since the previous offload",
        comment="The saved image keeps the header as first downloaded (the logger writes it at enable); the "
                "record of an offload that saw a different header keeps that header as header_hex.")
    _string_variable(nc, "logger_status", [a.get("status") or s.get("status") or "" for a, s in
                                           zip(after, snap, strict=True)], "logger status after the offload")
    _string_variable(nc, "power_source", [p.get("source") or "" for p in pwr], "power source at offload")
    _string_variable(nc, "offload_port", [_port_name(r.get("port")) for r in records], "serial port")
    _string_variable(nc, "offload_tool_version", [(r.get("tool") or {}).get("version") or "" for r in records],
                     "rbr-tpw version")


def _port_name(port) -> str:
    """A short port name, as the console shows it: usbmodem101, ttyACM0, COM3."""
    return (str(port or "?")).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].removeprefix("cu.")


def _correction_variables(nc: netCDF4.Dataset, corrections: list[TimeCorrection], keep: np.ndarray):
    """Which offload's skew re-timed each run of reset-clock samples (provenance for time_flag bit 4)."""
    nc.createDimension("time_correction", len(corrections))
    specs = [("time_correction_start_index", "i8", "index along time of the first sample of the re-timed run"),
             ("time_correction_end_index", "i8", "index along time just past the last sample of the re-timed run"),
             ("time_correction_offload", "i4", "index along offload_time of the offload whose skew was subtracted")]
    out = {}
    for name, dtype, long in specs:
        out[name] = nc.createVariable(name, dtype, ("time_correction",))
        out[name].setncatts({"long_name": long, "units": "1"})
    sk = nc.createVariable("time_correction_skew", "f8", ("time_correction",))
    sk.setncatts({"long_name": "clock skew subtracted from the logger clock for the run (logger minus UTC)",
                  "units": "s"})
    if corrections:
        out["time_correction_start_index"][:] = _kept_index(keep, [c.start for c in corrections])
        out["time_correction_end_index"][:] = _kept_index(keep, [c.end for c in corrections])
        out["time_correction_offload"][:] = [c.offload for c in corrections]
        sk[:] = [c.skew_s for c in corrections]


def _kept_index(keep: np.ndarray, sample_index: list[int]) -> list[int]:
    """Map sample-set indices of the full record onto indices along the kept time axis."""
    kept_before = np.concatenate([[0], np.cumsum(keep)])
    return [int(kept_before[min(max(i, 0), keep.size)]) for i in sample_index]


def write_netcdf(image: bytes, record: dict | list[dict], path: Path) -> tuple[Decoded, list[str], np.ndarray]:
    """Decode `image` using `record` and write `path` atomically.

    `record` is one offload record, or the deployment's records in offload order: the latest gives the
    metadata, all of them give the offload series and the clock skews that time reset-clock samples.
    Returns (decoded, warnings, UTC times in ms of the samples written).
    """
    records = records_of(record)
    record = records[-1]
    snap = record["snapshot_before"]
    channels = snap["channel_list"]
    warnings = list(record.get("warnings", []))
    evaluated = None  # (values, bad, problems, channels with the coefficients used) for sectioned L2 images
    if is_sectioned_header(image):  # RBRduet / RBRconcerto (L3 ref 5.3.1, header versions 1.xxx)
        all_channels = _header_completed_channels(image, snap, warnings)
        channels = [c for c in all_channels if not int(c.get("status", 0)) & 0x04]  # the stored ones, in order
        d = decode_l2(image, len(channels))
        evaluated = _evaluate_l2(d, snap, all_channels)
    else:
        d = decode(image, len(channels))
    # Also a reset clock: a logger enabled after its clock had restarted at 2000-01-01. The decoders' own
    # rule (an anchor earlier than the enable time) misses it, since the enable time is then in 2000 too.
    d.time_flags[d.time_ms < RESET_CLOCK_BEFORE_MS] |= TFLAG_RESET_CLOCK
    views = offload_views(records, _l2_seen(records, d),
                          lambda a, b: any(e.type in RTC_RESET_EVENTS for e in d.events[a:b]))
    corrections: list[TimeCorrection] = []
    t_utc, tflags, keep, notes = resolve_times(d, views=views, corrections=corrections)
    if d.rtc_reset:
        warnings.append("logger real-time clock was reset during this deployment (power loss)")
    warnings.extend(notes)
    if d.trailing_bytes:
        warnings.append(f"{d.trailing_bytes} trailing bytes did not form a complete sample set and were ignored")
    if d.bad_event_words:
        warnings.append(f"{d.bad_event_words} words looked like event markers but did not start a valid event record "
                        "(bad CRC or length)" + (
            " and were kept as readings (on this logger a reading can look like an event)"
            if evaluated is not None else "; they were dropped and sample times after them may be off"))

    with _atomic_dataset(path) as nc:
        chunk = _time_variables(nc, t_utc[keep], d.time_ms[keep], tflags[keep],
                                "Each sample set is timed from the preceding time-synchronization or restart "
                                "event plus n * sampling period.")
        used_names: set[str] = set()
        for k, ch in enumerate(channels):
            ctype = ch.get("type", "")
            name, std, units, long = channel_kind(ctype, k + 1)
            name = _unique_name(name, k + 1, used_names)
            raw = d.raw[keep, k]
            flags = d.flags[keep, k].copy()

            rv = nc.createVariable(f"{name}_raw", "u4", ("time",), zlib=True, complevel=4, shuffle=True,
                                   chunksizes=chunk)
            rv.setncatts({"long_name": f"{long} raw reading as stored by the logger", "units": "1",
                          "comment": "32-bit word from logger memory. Voltage ratio R = raw / 2**30. "
                                     "Words 0xF6xxxxxx are logger error codes (L3 ref section 5.3.2).",
                          "rbr_channel_type": ctype})
            rv[:] = raw

            coeffs = ch.get("coefficients", {})
            eq = EQUATIONS.get(ch.get("equation", ""))
            varnames = [f"{name}_raw", f"{name}_flag", "time_flag"]
            computed = None  # (values, bad) for the kept samples
            if evaluated is not None:
                all_values, all_bad, problems, used = evaluated
                coeffs = used[k].get("coefficients", coeffs)
                if k in problems:
                    warnings.append(f"{problems[k]}; raw readings only")
                else:
                    computed = (all_values[keep, k], all_bad[keep, k])
            elif eq is not None:
                computed = eq(raw, tuple(coeffs.get(f"c{i}", math.nan) for i in range(4)))
            else:
                warnings.append(f"channel {k + 1} ({ctype}, equation {ch.get('equation')!r}): no converter; "
                                "raw readings only")
            if computed is not None:
                values, bad = computed
                flags[bad & ((raw >> 24) != 0xF6)] |= FLAG_OUT_OF_RANGE
                vv = nc.createVariable(name, "f8", ("time",), zlib=True, complevel=4, chunksizes=chunk,
                                       fill_value=np.nan)
                atts = {"long_name": long, "units": units or ch.get("userunits", "1"),
                        "ancillary_variables": " ".join(varnames),
                        "rbr_channel_type": ctype, "calibration_equation": ch.get("equation", ""),
                        "calibration_datetime": ch.get("calibration_datetime", ""),
                        "comment": "Computed from the raw reading with the logger's own calibration coefficients "
                                   "(RBR 'tmp' Steinhart-Hart form, L3 command reference section 7.1.4). "
                                   "Matches Ruskin 2.26.1 to <1e-12 degC on test files."}
                if evaluated is not None:
                    atts["comment"] = ("Computed from the raw reading with the calibration stored in the logger's "
                                       f"deployment header (equation {ch.get('equation')}, L3 command reference "
                                       "section 7), including readings of the channels it references. Matched "
                                       "Ruskin 2.26.1 to <=1.1e-13 on 17 RBRduet/RBRconcerto files.")
                hidden = int(ch.get("status", 0) or 0) & 0x01  # e.g. a pressure sensor's compensation thermistor
                if std and not hidden:
                    atts["standard_name"] = std
                if std == "depth":
                    atts["positive"] = "down"
                if hidden:
                    atts["rbr_channel_status"] = np.int32(ch["status"])
                    atts["comment"] += (" The logger marks this channel hidden (channelStatus bit 0x01; Ruskin does "
                                        "not show it); it is not labelled as a sea-water quantity.")
                if units and units.startswith("degree_C"):
                    atts["units_metadata"] = "temperature: on_scale"
                for key, val in coeffs.items():
                    atts[f"calibration_{key}"] = val if isinstance(val, str) else float(val)
                vv.setncatts(atts)
                vv[:] = values
            _flag_variable(nc, name, long, flags, chunk)

        _event_variables(nc, [e.unix_ms for e in d.events], [e.type for e in d.events],
                         _kept_index(keep, [e.sample_index for e in d.events]))
        _offload_series(nc, records, views)
        _correction_variables(nc, corrections, keep)
        hdr = d.header
        deployment = {"deployment_enabled_logger_time": _s2000_iso(hdr.logger_time),
                      "deployment_start_time": _s2000_iso(hdr.start_time),
                      "deployment_end_time": _s2000_iso(hdr.end_time)}
        nc.setncatts(_global_attributes(records, warnings, t_utc[keep], period_ms=hdr.period_ms, nchan=d.nchan,
                                        rtc_reset=d.rtc_reset, deployment=deployment))
    return d, warnings, t_utc[keep]


def _header_completed_channels(image: bytes, snap: dict, warnings: list[str]) -> list[dict]:
    """The logger's channel list, completed from the deployment header when the logger listed fewer channels.

    Locked, a duet or concerto leaves its hidden channels out of `channels` (bench 2026-09-26: SN060275 lists 8
    of 9; the ninth, the pressure-compensation thermistor, is stored in every sample set). The header lists them
    all, with type, status and coefficients but no equation name: a temp* channel is decoded as `tmp`, anything
    else keeps its raw readings. Hidden channels come last on every logger seen, so positions still line up."""
    listed = [dict(c) for c in (snap.get("channels_all") or snap["channel_list"])]
    try:
        header_channels = parse_l2_header(image).fields.get("channels", [])
    except Exception:  # noqa: BLE001  (an unparseable header is reported by decode_l2 itself)
        return listed
    # The header describes the data as it was recorded; the logger's list describes the logger now. A sensor that
    # failed since (RBRduet SN081015, bench 2026-09-27: its compensation thermistor went from status 9, stored, to
    # 31, not stored, unresponsive) must still be decoded as stored, or every sample set is misread.
    for ch, hc in zip(listed, header_channels, strict=False):
        if str(hc.get("type", "")) != str(ch.get("type", "")):
            break  # a different channel table: leave the logger's statuses alone
        now, then = int(ch.get("status", 0)), int(hc.get("status", 0))
        if (now ^ then) & (0x04 | 0x01):
            ch["status_at_offload"] = now
            ch["status"] = then
            warnings.append(f"channel {ch.get('index')} ({ch.get('type')}): the logger now reports status {now}, the "
                            f"deployment header status {then}; decoded as recorded (header)")
    for hc in header_channels[len(listed):]:
        ctype = str(hc.get("type", ""))
        equation = "tmp" if ctype.startswith("temp") else ""
        names = ["c0", "c1", "c2", "c3"] if equation else []
        listed.append({"index": int(hc["index"]), "type": ctype, "status": int(hc.get("status", 0)),
                       "equation": equation, "userunits": "C" if equation else "",
                       "coefficients": header_coefficients(hc, names) if names else {},
                       "calibration_datetime": "", "from_deployment_header": True})
        warnings.append(f"channel {hc['index']} ({ctype}, status {hc.get('status')}) is in the deployment header "
                        "but the logger did not list it (a locked logger hides its hidden channels); "
                        + ("decoded as `tmp` with the header's coefficients" if equation else "raw readings only"))
    return listed


def _evaluate_l2(d: Decoded, snap: dict, channels: list[dict] | None = None):
    """Engineering values for a sectioned L2 image: coefficients from the memory header (the calibration in
    force when the deployment was enabled), named in the order the logger's `calibration N` reply lists them."""
    header_channels = d.header.fields.get("channels", [])
    used = []
    for ch in channels if channels is not None else (snap.get("channels_all") or snap["channel_list"]):
        i = int(ch.get("index", len(used) + 1))
        hc = header_channels[i - 1] if i <= len(header_channels) else None
        if hc is not None and hc.get("coefficient_words"):
            # header words are stored c0.., x0.., n0.. (the order of the `calibration N` reply); sort the names so
            # a differently ordered source (e.g. an .rsk coefficients table) pairs them the same way
            names = sorted(ch.get("coefficients", {}), key=lambda n: ("cxn".find(n[:1]) % 4, int(n[1:] or 0)
                                                                      if n[1:].isdigit() else 0))
            used.append({**ch, "coefficients": header_coefficients(hc, names)})
        else:
            used.append(ch)
    settings = snap.get("settings") or {}
    defaults = {k: float(settings[k]) for k in ("temperature", "pressure") if k in settings}
    values, bad, problems = evaluate(d.raw, used, defaults)
    stored = [c for c in used if not int(c.get("status", 0)) & 0x04]
    return values, bad, problems, stored


@dataclass
class ChannelValues:
    """One channel of engineering values computed outside rbr-tpw (e.g. by Ruskin)."""

    ctype: str  # RBR channel type, e.g. temp09
    order: int  # 1-based channel order on the logger
    values: np.ndarray  # float64 per sample set, NaN where there is no value
    flags: np.ndarray  # uint8 FLAG_* bits per sample set
    attrs: dict = field(default_factory=dict)  # extra variable attributes (calibration, sensor serial, ...)
    hidden: bool = False  # hidden in Ruskin: what it measures is not documented, so no CF standard_name


def write_values_netcdf(time_ms: np.ndarray, time_flags: np.ndarray, segment: np.ndarray,
                        channels: list[ChannelValues], events: list[tuple[int, int, int]],
                        record: dict | list[dict], path: Path, *, period_ms: int, deployment: dict,
                        values_comment: str, flag_comment: str, seen: list[tuple[int, int]] | None = None,
                        event_payloads: list[int] | None = None) -> tuple[list[str], np.ndarray]:
    """Write engineering values in the write_netcdf() layout, without the raw readings.

    `time_ms` is the logger clock per sample set; `time_flags`/`segment` mark reset-clock samples and
    clock segments as in rawbin.decode(); `events` are (logger-clock ms, type code, sample-set index), with
    their stored payloads in `event_payloads` when the format has them. `record` is one offload record or the
    deployment's records in order, and `seen[k]` = (sample sets, events) offload k's image held (default: all).
    Returns (warnings, UTC times in ms of the samples written).
    """
    records = records_of(record)
    record = records[-1]
    warnings = list(record.get("warnings", []))
    if seen is None:
        seen = [(time_ms.size, len(events))] * len(records)
    views = offload_views(records, seen, lambda a, b: any(e[1] in RESTART_EVENTS for e in events[a:b]))
    corrections: list[TimeCorrection] = []
    t_utc, tflags, keep, notes = resolve_time_arrays(time_ms, time_flags, segment, views=views,
                                                     starts=reset_segment_starts(time_ms, events),
                                                     corrections=corrections)
    rtc_reset = bool((time_flags & TFLAG_RESET_CLOCK).any())
    if rtc_reset:
        warnings.append("logger real-time clock had been reset (power loss): sample times start at 2000-01-01")
    warnings.extend(notes)

    with _atomic_dataset(path) as nc:
        chunk = _time_variables(nc, t_utc[keep], time_ms[keep], tflags[keep],
                                "Logger-clock times as stored in the source file.")
        used_names: set[str] = set()
        # visible channels first, so a hidden channel never takes the plain name (e.g. "temperature")
        for ch in sorted(channels, key=lambda c: (c.hidden, c.order)):
            name, std, units, long = channel_kind(ch.ctype, ch.order)
            name = _unique_name(name, ch.order, used_names)
            vv = nc.createVariable(name, "f8", ("time",), zlib=True, complevel=4, chunksizes=chunk,
                                   fill_value=np.nan)
            atts = {"long_name": long, "units": units or ch.attrs.get("rbr_units", "1"),
                    "ancillary_variables": f"{name}_flag time_flag", "rbr_channel_type": ch.ctype,
                    "rbr_channel_order": np.int32(ch.order)}
            if std and not ch.hidden:
                atts["standard_name"] = std
            if std == "depth":
                atts["positive"] = "down"
            if (units or "").startswith("degree_C"):
                atts["units_metadata"] = "temperature: on_scale"
            atts.update(ch.attrs)
            comment = values_comment
            if ch.hidden:
                comment += (" Ruskin hides this channel by default (channelStatus bit 0x01); what it measures is "
                            "not documented in the file, so no standard_name is given.")
            atts["comment"] = comment
            vv.setncatts(atts)
            vv[:] = ch.values[keep]
            _flag_variable(nc, name, long, ch.flags[keep], chunk, flag_comment)

        _event_variables(nc, [e[0] for e in events], [e[1] for e in events],
                         _kept_index(keep, [e[2] for e in events]), payloads=event_payloads)
        _offload_series(nc, records, views)
        _correction_variables(nc, corrections, keep)
        nc.setncatts(_global_attributes(records, warnings, t_utc[keep], period_ms=period_ms, nchan=len(channels),
                                        rtc_reset=rtc_reset, deployment=deployment))
    return warnings, t_utc[keep]


def _remaining_attrs(rem: dict | None) -> dict:
    if not rem:
        return {}
    return {
        "sampling_days_remaining": float(rem["days"]),
        "sampling_limited_by": rem["limited_by"],
        "energy_days_remaining_modelled": float(rem["energy_days"]),
        "energy_per_day_modelled_J": float(rem["energy_per_day_J"]),
        "energy_used_this_deployment_modelled_J": float(rem["energy_used_this_deployment_J"]),
        "energy_model_comment": "Energy-limited days = (" + (
            "energy counter x 0.9 derating - modelled use for the samples in memory"
            if rem.get("derating") == "proportional" else  # records written before 2026-09-25
            "energy counter - modelled use for the samples in memory - 3,370 J derating"
        ) + ") / modelled J per day, computed at offload, with 3.6 V x (0.69 mA while sampling for "
                                "latency + read time, 0.0055 mA asleep) from Ruskin 2.26.1 constants. Not yet "
                                "checked against Ruskin's own estimate.",
    }


def _history(records: list[dict], raw_name: str, now: str) -> str:
    """One line per offload of the deployment, oldest first, then the line for this file."""
    lines = []
    for k, rec in enumerate(records):
        if rec.get("history"):  # e.g. rbr-rsk2nc: the record says where it came from
            lines.append(f"{now} rbr-tpw {__version__}: {rec['history']}")
            continue
        ver = (rec.get("tool") or {}).get("version", "?")
        port = _port_name(rec.get("port"))
        dep = rec.get("deployment") or {}
        if dep:
            seg = dep.get("segment") or {}
            lines.append(f"{rec['offload_started']} rbr-tpw {ver}: offload {dep.get('offload_index', k)} from {port}, "
                         f"{dep.get('download', '?')}, {seg.get('bytes', '?')} bytes added, "
                         f"{dep.get('bytes_read', seg.get('bytes', '?'))} read, {dep.get('image_bytes', '?')} held")
        else:
            lines.append(f"{rec['offload_started']} rbr-tpw {ver}: offloaded from {port}")
    lines.append(f"{now} rbr-tpw {__version__}: NetCDF written from {raw_name}")
    return "\n".join(lines)


def _global_attributes(records: list[dict] | dict, warnings: list[str], t_ms: np.ndarray, *, period_ms: int,
                       nchan: int, rtc_reset: bool, deployment: dict) -> dict:
    records = records_of(records)
    record = records[-1]
    ident = record["id"]
    snap = record["snapshot_before"]
    after = record.get("after", {})
    skew = record.get("clock_skew", {})
    ntp = record.get("host_ntp", {})
    mem = after.get("meminfo") or snap.get("meminfo")
    pwr = after.get("power") or snap.get("power")
    raw = record.get("raw")
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    sn = ident["serial"]
    raw_name = raw["file"] if raw else "?"

    a = {
        "Conventions": "CF-1.13",
        "title": record.get("title") or f"{ident['model']} SN{sn} data offloaded {record['offload_started']}",
        "source": record.get("source")
        or f"{ident['model']} SN{sn} memory download (read data), decoded by rbr-tpw {__version__}",
        "history": _history(records, raw_name, now),
        "date_created": now,
        "instrument": ident["model"],
        "instrument_serial_number": sn,
        "instrument_firmware_version": ident["version"],
        "instrument_firmware_type": int(ident["fwtype"]),
        "logger_status_at_offload": snap.get("status") or "",
        **deployment,
        "sampling_mode": snap.get("sampling", {}).get("mode", ""),
        "sampling_period_ms": int(period_ms),
        "offload_time_utc": record["offload_started"],
    }
    if raw:
        a.update({"raw_file": raw["file"], "raw_bytes": int(raw["bytes"]), "raw_sha256": raw["sha256"]})
    a["clock_reset_detected"] = "yes" if rtc_reset else "no"
    if len(t_ms):
        a["time_coverage_start"] = _iso(int(t_ms[0]))
        a["time_coverage_end"] = _iso(int(t_ms[-1]))

    # Clock skew: logger minus UTC.  UTC = host + ntp_offset.
    if skew.get("n"):
        off = ntp.get("offset_s")
        vs_utc = skew["skew_vs_host_s"] - off if off is not None else math.nan
        unc = skew["uncertainty_s"] + (ntp.get("uncertainty_s") or 0.0) if off is not None else math.nan
        a.update({
            "clock_skew_s": vs_utc,
            "clock_skew_uncertainty_s": unc,
            "clock_skew_vs_host_s": skew["skew_vs_host_s"],
            "clock_skew_vs_host_uncertainty_s": _num(skew["uncertainty_s"]),
            "clock_skew_spread_s": _num(skew["spread_s"]),
            "clock_skew_n": int(skew["n"]),
            "clock_skew_measured_at": skew["measured_at"],
            "clock_skew_method": skew.get("method")
            or "Logger clock minus UTC (positive = logger ahead). Polled the logger's `now` "
               "(whole seconds) until it ticked, bracketed the tick between request/reply host "
               "times, median of clock_skew_n ticks; UTC = host clock + host_ntp_offset_s. "
               "Uncertainty = worst tick half-bracket + NTP uncertainty. Measured before the "
               "download.",
        })
    else:
        a["clock_skew_s"] = math.nan
        warnings.append(skew.get("error") or "clock skew could not be measured")
    a["host_ntp_server"] = ntp.get("server", "")
    a["host_ntp_offset_s"] = _num(ntp.get("offset_s"))
    a["host_ntp_uncertainty_s"] = _num(ntp.get("uncertainty_s"))
    if "error" in ntp:
        a["host_ntp_error"] = ntp["error"]

    if mem:
        samples_per_day = 86_400_000 / period_ms if period_ms else math.nan
        bytes_per_sample = record.get("bytes_per_sample") or 4 * nchan
        bytes_per_day = bytes_per_sample * samples_per_day
        a.update({
            "memory_size_bytes": int(mem["size"]),
            "memory_used_bytes": int(mem["used"]),
            "memory_remaining_bytes": int(mem["remaining"]),
            "memory_remaining_fraction": mem["remaining"] / mem["size"] if mem["size"] else math.nan,
            "memory_remaining_days": mem["remaining"] / bytes_per_day
            if snap["sampling"].get("mode") == "continuous" else math.nan,
            "memory_comment": "From `meminfo` after the download. memory_remaining_days assumes continuous "
                              f"sampling at sampling_period_ms with {bytes_per_sample} bytes per sample set (events "
                              "ignored).",
        })
    a.update(_remaining_attrs(record.get("remaining_time")))
    if pwr:
        a.update({
            "power_source_at_offload": pwr.get("source", ""),
            "battery_voltage_V": _num(pwr.get("battery_voltage_V")),
            "battery_energy_remaining_J": _num(pwr.get("energy_remaining_J")),
            "battery_energy_nominal_J": _num(pwr.get("energy_nominal_J")),
            "battery_energy_remaining_fraction": _num(pwr.get("energy_remaining_fraction")),
            "battery_powerstatus_raw": f"int = {pwr.get('int_raw')}, remaining = {pwr.get('remaining_raw')}",
            "battery_comment": pwr.get("comment") or SOLO_BATTERY_COMMENT,
        })
    dep = record.get("deployment") or {}
    if dep:
        a.update({"deployment_stem": dep.get("stem", ""), "deployment_offloads": int(len(records)),
                  "deployment_header_sha256": dep.get("header_sha256", "")})
    a["offload_series_comment"] = ("The battery, memory, clock-skew and remaining-time attributes describe the "
                                   "latest offload; the variables along offload_time hold the same quantities "
                                   "for every offload of this deployment.")
    a.update(record.get("extra_attributes", {}))
    if warnings:
        a["warnings"] = "; ".join(warnings)
    return a
