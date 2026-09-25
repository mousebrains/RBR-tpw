"""Write a decoded logger download as a CF-1.13 NetCDF-4 file.

Built from two inputs so it can be re-run offline: the raw memory image and
the JSON offload record (logger config, clock skew, memory, power).
"""

from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path

import netCDF4
import numpy as np

from . import __version__
from .rawbin import (
    EQUATIONS,
    EVENT_NAMES,
    FLAG_ERROR_CODE,
    FLAG_OUT_OF_RANGE,
    TFLAG_NO_ANCHOR,
    TFLAG_RESET_CLOCK,
    TFLAG_SKEW_CORRECTED,
    Decoded,
    decode,
    resolve_times,
)

TIME_UNITS = "milliseconds since 1970-01-01 00:00:00"

# RBR channel type prefix -> (variable name, CF standard_name, units, long_name)
CHANNEL_KINDS = {
    "temp": ("temperature", "sea_water_temperature", "degree_Celsius", "Temperature"),
}


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


def write_netcdf(image: bytes, record: dict, path: Path) -> tuple[Decoded, list[str], np.ndarray]:
    """Decode `image` using `record` and write `path` atomically.

    Returns (decoded, warnings, UTC times in ms of the samples written).
    """
    snap = record["snapshot_before"]
    channels = snap["channel_list"]
    d = decode(image, len(channels))
    warnings = list(record.get("warnings", []))
    offload_ms = _parse_iso_ms(record.get("offload_finished") or record["offload_started"])
    t_utc, tflags, keep, notes = resolve_times(d, skew_vs_utc(record), offload_ms)
    if d.rtc_reset:
        warnings.append("logger real-time clock was reset during this deployment (power loss)")
    warnings.extend(notes)
    if d.trailing_bytes:
        warnings.append(f"{d.trailing_bytes} trailing bytes did not form a complete sample set and were ignored")

    tmp = path.with_name(path.name + ".tmp")
    nc = netCDF4.Dataset(tmp, "w", format="NETCDF4")
    try:
        n = int(keep.sum())
        nc.createDimension("time", n)
        chunk = (max(1, min(n, 1 << 18)),)
        tv = nc.createVariable("time", "i8", ("time",), zlib=True, complevel=4, chunksizes=chunk)
        tv.setncatts({"standard_name": "time", "long_name": "sample time (UTC)",
                      "units": TIME_UNITS, "calendar": "standard", "units_metadata": "leap_seconds: none",
                      "axis": "T",
                      "comment": "Logger clock, except samples flagged time_corrected_by_offload_skew in time_flag "
                                 "(taken after a clock reset), which are logger clock minus clock_skew_s. Not "
                                 "otherwise corrected for clock skew. Each sample set is timed from the preceding "
                                 "time-synchronization or restart event plus n * sampling period."})
        tv[:] = t_utc[keep]
        lt = nc.createVariable("logger_time", "i8", ("time",), zlib=True, complevel=4, chunksizes=chunk)
        lt.setncatts({"long_name": "sample time on the logger's clock, as recorded", "units": TIME_UNITS,
                      "calendar": "standard", "units_metadata": "leap_seconds: none"})
        lt[:] = d.time_ms[keep]
        tq = nc.createVariable("time_flag", "u1", ("time",), zlib=True, complevel=4, chunksizes=chunk)
        tq.setncatts({"long_name": "sample time quality flags",
                      "flag_masks": np.array([TFLAG_NO_ANCHOR, TFLAG_RESET_CLOCK, TFLAG_SKEW_CORRECTED], "u1"),
                      "flag_meanings": "no_time_anchor_before_sample logger_clock_had_been_reset "
                                       "time_corrected_by_offload_skew"})
        tq[:] = tflags[keep]

        used_names = set()
        for k, ch in enumerate(channels):
            ctype = ch.get("type", "")
            kind = next((v for p, v in CHANNEL_KINDS.items() if ctype.startswith(p)), None)
            name, std, units, long = kind or (f"channel{k + 1:02d}", None, None, f"Channel {k + 1} ({ctype})")
            if name in used_names:
                name = f"{name}{k + 1:02d}"
            used_names.add(name)
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
            if eq is not None:
                c = tuple(coeffs.get(f"c{i}", math.nan) for i in range(4))
                values, bad = eq(raw, c)
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
                if std:
                    atts["standard_name"] = std
                if units and units.startswith("degree_C"):
                    atts["units_metadata"] = "temperature: on_scale"
                for key, val in coeffs.items():
                    atts[f"calibration_{key}"] = float(val)
                vv.setncatts(atts)
                vv[:] = values
            else:
                warnings.append(f"channel {k + 1} ({ctype}, equation {ch.get('equation')!r}): no converter; "
                                "raw readings only")

            fv = nc.createVariable(f"{name}_flag", "u1", ("time",), zlib=True, complevel=4, chunksizes=chunk)
            fv.setncatts({"long_name": f"{long} quality flags",
                          "flag_masks": np.array([FLAG_ERROR_CODE, FLAG_OUT_OF_RANGE], "u1"),
                          "flag_meanings": "logger_error_code conversion_out_of_range"})
            fv[:] = flags

        nc.createDimension("event", len(d.events))
        codes = np.array(sorted(EVENT_NAMES), "u1")
        et = nc.createVariable("event_time", "i8", ("event",))
        et.setncatts({"long_name": "event time on the logger's clock", "units": TIME_UNITS, "calendar": "standard",
                      "units_metadata": "leap_seconds: none"})
        ey = nc.createVariable("event_type", "u1", ("event",))
        ey.setncatts({"long_name": "logger event type", "flag_values": codes,
                      "flag_meanings": " ".join(EVENT_NAMES[int(c)] for c in codes),
                      "comment": "RBR event type codes, L3 command reference section 5.3.3"})
        ei = nc.createVariable("event_sample_index", "i8", ("event",))
        ei.setncatts({"long_name": "index along time of the first sample after the event", "units": "1"})
        if d.events:
            kept_before = np.concatenate([[0], np.cumsum(keep)])
            et[:] = [e.unix_ms for e in d.events]
            ey[:] = [e.type for e in d.events]
            ei[:] = [int(kept_before[min(e.sample_index, keep.size)]) for e in d.events]

        nc.setncatts(_global_attributes(d, record, warnings, t_utc[keep]))
    except BaseException:
        nc.close()
        tmp.unlink(missing_ok=True)
        raise
    nc.close()
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return d, warnings, t_utc[keep]


def _remaining_attrs(rem: dict | None) -> dict:
    if not rem:
        return {}
    return {
        "sampling_days_remaining": float(rem["days"]),
        "sampling_limited_by": rem["limited_by"],
        "energy_days_remaining_modelled": float(rem["energy_days"]),
        "energy_per_day_modelled_J": float(rem["energy_per_day_J"]),
        "energy_used_this_deployment_modelled_J": float(rem["energy_used_this_deployment_J"]),
        "energy_model_comment": "Energy-limited days = (energy counter - modelled use for the samples in memory - "
                                "10% derating) / modelled J per day, with 3.6 V x (0.69 mA while sampling for "
                                "latency + read time, 0.0055 mA asleep) from Ruskin 2.26.1 constants. Not yet "
                                "checked against Ruskin's own estimate.",
    }


def _global_attributes(d: Decoded, record: dict, warnings: list[str], t_ms: np.ndarray) -> dict:
    ident = record["id"]
    snap = record["snapshot_before"]
    after = record.get("after", {})
    skew = record.get("clock_skew", {})
    ntp = record.get("host_ntp", {})
    mem = after.get("meminfo") or snap["meminfo"]
    pwr = after.get("power") or snap["power"]
    hdr = d.header
    period = hdr.period_ms
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    sn = ident["serial"]

    a = {
        "Conventions": "CF-1.13",
        "title": f"{ident['model']} SN{sn} data offloaded {record['offload_started']}",
        "source": f"{ident['model']} SN{sn} memory download (read data), decoded by rbr-tpw {__version__}",
        "history": f"{now} rbr-tpw {__version__}: offloaded from {record.get('port', '?')} at "
                   f"{record['offload_started']}; NetCDF written from {record['raw']['file']}",
        "date_created": now,
        "instrument": ident["model"],
        "instrument_serial_number": sn,
        "instrument_firmware_version": ident["version"],
        "instrument_firmware_type": int(ident["fwtype"]),
        "logger_status_at_offload": snap["status"],
        "deployment_enabled_logger_time": _s2000_iso(hdr.logger_time),
        "deployment_start_time": _s2000_iso(hdr.start_time),
        "deployment_end_time": _s2000_iso(hdr.end_time),
        "sampling_mode": snap["sampling"].get("mode", ""),
        "sampling_period_ms": int(period),
        "offload_time_utc": record["offload_started"],
        "raw_file": record["raw"]["file"],
        "raw_bytes": int(record["raw"]["bytes"]),
        "raw_sha256": record["raw"]["sha256"],
    }
    a["clock_reset_detected"] = "yes" if d.rtc_reset else "no"
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
            "clock_skew_vs_host_uncertainty_s": skew["uncertainty_s"],
            "clock_skew_spread_s": skew["spread_s"],
            "clock_skew_n": int(skew["n"]),
            "clock_skew_measured_at": skew["measured_at"],
            "clock_skew_method": "Logger clock minus UTC (positive = logger ahead). Polled the logger's `now` "
                                 "(whole seconds) until it ticked, bracketed the tick between request/reply host "
                                 "times, median of clock_skew_n ticks; UTC = host clock + host_ntp_offset_s. "
                                 "Uncertainty = worst tick half-bracket + NTP uncertainty. Measured before the "
                                 "download.",
        })
    else:
        a["clock_skew_s"] = math.nan
        warnings.append("clock skew could not be measured")
    a["host_ntp_server"] = ntp.get("server", "")
    a["host_ntp_offset_s"] = _num(ntp.get("offset_s"))
    a["host_ntp_uncertainty_s"] = _num(ntp.get("uncertainty_s"))
    if "error" in ntp:
        a["host_ntp_error"] = ntp["error"]

    samples_per_day = 86_400_000 / period if period else math.nan
    bytes_per_day = 4 * d.nchan * samples_per_day
    a.update({
        "memory_size_bytes": int(mem["size"]),
        "memory_used_bytes": int(mem["used"]),
        "memory_remaining_bytes": int(mem["remaining"]),
        "memory_remaining_fraction": mem["remaining"] / mem["size"] if mem["size"] else math.nan,
        "memory_remaining_days": mem["remaining"] / bytes_per_day
        if snap["sampling"].get("mode") == "continuous" else math.nan,
        "memory_comment": "From `meminfo` after the download. memory_remaining_days assumes continuous sampling "
                          f"at sampling_period_ms with {4 * d.nchan} bytes per sample set (events ignored).",
        **_remaining_attrs(record.get("remaining_time")),
        "power_source_at_offload": pwr.get("source", ""),
        "battery_voltage_V": _num(pwr.get("battery_voltage_V")),
        "battery_energy_remaining_J": _num(pwr.get("energy_remaining_J")),
        "battery_energy_nominal_J": _num(pwr.get("energy_nominal_J")),
        "battery_energy_remaining_fraction": _num(pwr.get("energy_remaining_fraction")),
        "battery_powerstatus_raw": f"int = {pwr.get('int_raw')}, remaining = {pwr.get('remaining_raw')}",
        "battery_comment": "From `powerstatus` after the download. battery_voltage_V = int / 1000 (mV). "
                           "battery_energy_remaining_J = hex(remaining) / 1000 (mJ); this is the logger's own "
                           "energy counter, meaningful only if reset (Ruskin 'Fresh battery') when the cell was "
                           "replaced. Nominal = one AA 3.6 V 2.6 Ah Li-SOCl2 cell (Ruskin's reset value). "
                           "Voltage units verified against a meter; energy-counter units inferred from Ruskin "
                           "2.26.1 behaviour, not from RBR documentation.",
    })
    if warnings:
        a["warnings"] = "; ".join(warnings)
    return a
