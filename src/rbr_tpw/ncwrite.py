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
from .equations import decode_l2, evaluate, header_coefficients, is_sectioned_header
from .rawbin import (
    EQUATIONS,
    EVENT_NAMES,
    FLAG_ERROR_CODE,
    FLAG_OUT_OF_RANGE,
    RESET_CLOCK_BEFORE_MS,
    TFLAG_NO_ANCHOR,
    TFLAG_RESET_CLOCK,
    TFLAG_SKEW_CORRECTED,
    Decoded,
    decode,
    event_name,
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
                "clock reset), which are logger clock minus the clock skew measured at offload (clock_skew_s, or "
                "clock_skew_vs_host_s when the host clock was not referenced to UTC). Not otherwise corrected for "
                "clock skew.")


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
    with open(tmp, "rb") as f:
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


def _event_variables(nc: netCDF4.Dataset, times_ms: list[int], types: list[int], index: list[int]):
    """Event list; `index` is the position along time of the first sample after each event."""
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


def _kept_index(keep: np.ndarray, sample_index: list[int]) -> list[int]:
    """Map sample-set indices of the full record onto indices along the kept time axis."""
    kept_before = np.concatenate([[0], np.cumsum(keep)])
    return [int(kept_before[min(max(i, 0), keep.size)]) for i in sample_index]


def write_netcdf(image: bytes, record: dict, path: Path) -> tuple[Decoded, list[str], np.ndarray]:
    """Decode `image` using `record` and write `path` atomically.

    Returns (decoded, warnings, UTC times in ms of the samples written).
    """
    snap = record["snapshot_before"]
    channels = snap["channel_list"]
    warnings = list(record.get("warnings", []))
    evaluated = None  # (values, bad, problems, channels with the coefficients used) for sectioned L2 images
    if is_sectioned_header(image):  # RBRduet / RBRconcerto (L3 ref 5.3.1, header versions 1.xxx)
        d = decode_l2(image, len(channels))
        evaluated = _evaluate_l2(d, snap)
        # also a reset clock: a logger enabled after its clock had restarted at 2000-01-01 (rawbin rule misses it)
        d.time_flags[d.time_ms < RESET_CLOCK_BEFORE_MS] |= TFLAG_RESET_CLOCK
    else:
        d = decode(image, len(channels))
    offload_ms = _parse_iso_ms(record.get("offload_finished") or record["offload_started"])
    t_utc, tflags, keep, notes = resolve_times(d, skew_vs_utc(record), offload_ms)
    if d.rtc_reset:
        warnings.append("logger real-time clock was reset during this deployment (power loss)")
    warnings.extend(notes)
    if d.trailing_bytes:
        warnings.append(f"{d.trailing_bytes} trailing bytes did not form a complete sample set and were ignored")
    if d.bad_event_words:
        warnings.append(f"{d.bad_event_words} event records failed their CRC" + (
            " and were kept as readings (on this logger a reading can look like an event)"
            if evaluated is not None else " and were dropped; sample times after them may be off"))

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
        hdr = d.header
        deployment = {"deployment_enabled_logger_time": _s2000_iso(hdr.logger_time),
                      "deployment_start_time": _s2000_iso(hdr.start_time),
                      "deployment_end_time": _s2000_iso(hdr.end_time)}
        nc.setncatts(_global_attributes(record, warnings, t_utc[keep], period_ms=hdr.period_ms, nchan=d.nchan,
                                        rtc_reset=d.rtc_reset, deployment=deployment))
    return d, warnings, t_utc[keep]


def _evaluate_l2(d: Decoded, snap: dict):
    """Engineering values for a sectioned L2 image: coefficients from the memory header (the calibration in
    force when the deployment was enabled), named in the order the logger's `calibration N` reply lists them."""
    header_channels = d.header.fields.get("channels", [])
    used = []
    for ch in snap.get("channels_all") or snap["channel_list"]:
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
                        channels: list[ChannelValues], events: list[tuple[int, int, int]], record: dict,
                        path: Path, *, period_ms: int, deployment: dict, values_comment: str,
                        flag_comment: str) -> tuple[list[str], np.ndarray]:
    """Write engineering values in the write_netcdf() layout, without the raw readings.

    `time_ms` is the logger clock per sample set; `time_flags`/`segment` mark reset-clock samples and
    clock segments as in rawbin.decode(); `events` are (logger-clock ms, type code, sample-set index).
    Returns (warnings, UTC times in ms of the samples written).
    """
    warnings = list(record.get("warnings", []))
    offload_ms = _parse_iso_ms(record.get("offload_finished") or record["offload_started"])
    t_utc, tflags, keep, notes = resolve_time_arrays(time_ms, time_flags, segment, skew_vs_utc(record), offload_ms)
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
                         _kept_index(keep, [e[2] for e in events]))
        nc.setncatts(_global_attributes(record, warnings, t_utc[keep], period_ms=period_ms, nchan=len(channels),
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


def _global_attributes(record: dict, warnings: list[str], t_ms: np.ndarray, *, period_ms: int, nchan: int,
                       rtc_reset: bool, deployment: dict) -> dict:
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
        "history": f"{now} rbr-tpw {__version__}: "
        + (record.get("history") or f"offloaded from {record.get('port', '?')} at {record['offload_started']}; "
                                    f"NetCDF written from {raw_name}"),
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
    a.update(record.get("extra_attributes", {}))
    if warnings:
        a["warnings"] = "; ".join(warnings)
    return a
