"""rbr-rsk2nc: convert Ruskin .rsk files to rbr-tpw's CF NetCDF (the rbr-offload format).

Each file takes one of two routes:

  decode  RBRsolo fwtype 0 and 9. Ruskin keeps the logger's memory image in the `downloads`
          table; rbr-tpw decodes it exactly as rbr-offload does (same variables, raw words, time
          flags and clock-reset handling) and compares the result with Ruskin's values.
  values  Other loggers (RBRduet, RBRconcerto, RBRconcerto3, ...), whose memory formats rbr-tpw
          cannot decode. Ruskin's computed values from the `data` table are written in the same
          layout, without the raw words.

The .rsk file is opened read-only and never modified.

RSK facts relied on (Ruskin 2.18-2.26 files with RSK schema 2.18.2/2.19.0; checked 2026-09-25 on
141 files):
  - deployments.loggerTimeDrift = logger clock minus Ruskin host clock (ms) and timeOfDownload =
    logger clock at download (ms): timeOfDownload - loggerTimeDrift matches the file times.
    EasyParse ("EPdesktop") files leave loggerTimeDrift empty.
  - data.tstamp is the logger clock in ms since 1970. Columns are channelNN, NN = channelOrder,
    only for stored channels.
  - instrumentChannels.channelStatus bits, as used by RBR's pyRSKtools 1.3.0: 0x01 hidden,
    0x04 not stored, 0x08 not streamed.
  - events.sampleIndex is 1-based (it equals rbr-tpw's 0-based index + 1 on 86 solo files), and
    -1 in EasyParse files.
  - fwtype 0 and 9 memory images decode with rawbin.decode to Ruskin's values within 1.2e-13 degC,
    with identical sample times and events, on 86 files (all but the two SN100689 files whose
    clock reset, where Ruskin times the samples from the deployment start time instead).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .ncwrite import ChannelValues, write_netcdf, write_values_netcdf
from .rawbin import (
    EQUATIONS,
    FLAG_ERROR_CODE,
    RESTART_EVENTS,
    TFLAG_RESET_CLOCK,
    Decoded,
    decode,
)

DECODABLE_FWTYPES = {0, 9}
STATUS_HIDDEN = 0x01
STATUS_NOT_STORED = 0x04
# An RBR clock restarts at 2000-01-01 after a power loss; no deployment samples in 2000 on purpose.
RESET_CLOCK_BEFORE_MS = 978_307_200_000  # 2001-01-01T00:00:00Z
VALUE_TOLERANCE = 1e-9  # decode vs Ruskin, engineering units
READ_BLOCK = 200_000  # data-table rows per fetch


class RskError(Exception):
    pass


@dataclass
class RskChannel:
    order: int
    short_name: str
    long_name: str
    units: str
    units_plain: str
    status: int
    derived: bool
    column: str | None  # data-table column, None if Ruskin did not store the channel
    equation: str = ""
    calibration_ms: int | None = None
    coefficients: dict[str, str] = field(default_factory=dict)
    sensor_serial: str = ""

    @property
    def hidden(self) -> bool:
        return bool(self.status & STATUS_HIDDEN)


@dataclass
class Rsk:
    path: Path
    schema: str
    kind: str  # "full" (Ruskin kept the raw download) or "EPdesktop" (EasyParse)
    ruskin_version: str
    model: str
    serial: int
    firmware: str
    fwtype: int
    part_number: str
    status: str
    drift_ms: int | None
    download_logger_ms: int | None
    mode: str
    gate: str
    period_ms: int
    start_ms: int | None
    end_ms: int | None
    channels: list[RskChannel]
    parameters: dict[str, str]
    nrows: int

    @property
    def serial_str(self) -> str:
        return f"{self.serial:06d}"

    @property
    def download_host_ms(self) -> int | None:
        """Host clock at download; logger clock if Ruskin recorded no drift (unverified which, then)."""
        if self.download_logger_ms is None:
            return None
        return self.download_logger_ms - (self.drift_ms or 0)

    @property
    def stored(self) -> list[RskChannel]:
        return [c for c in self.channels if c.column]


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)  # read-only: never touch the .rsk
    con.row_factory = sqlite3.Row
    return con


def _rows(con: sqlite3.Connection, sql: str, *args) -> list[dict]:
    try:
        return [dict(r) for r in con.execute(sql, args)]
    except sqlite3.OperationalError as err:
        if "no such table" in str(err):
            return []
        raise


def _one(con: sqlite3.Connection, table: str) -> dict:
    rows = _rows(con, f"select * from {table}")
    if len(rows) != 1:
        raise RskError(f"expected one row in `{table}`, found {len(rows)} (multi-deployment files are not supported)")
    return rows[0]


def read_rsk(path: Path, con: sqlite3.Connection) -> Rsk:
    db = _one(con, "dbInfo")
    if int(str(db["version"]).split(".")[0]) >= 3:
        raise RskError(f"RSK schema {db['version']} is not supported yet (only 2.x)")
    inst = _one(con, "instruments")
    dep = _one(con, "deployments")
    sched = _one(con, "schedules")
    period = None
    for table in ("continuous", "average", "burst", "tide", "wave"):
        rows = _rows(con, f"select samplingPeriod from {table} where scheduleID = ?", sched["scheduleID"])
        if rows:
            period = int(rows[0]["samplingPeriod"])
            break
    epochs = _rows(con, "select startTime, endTime from epochs")
    app = _rows(con, "select ruskinVersion from appSettings")

    columns = {r["name"] for r in _rows(con, "pragma table_info(data)")}
    linked = {r["calibrationID"] for r in _rows(con, "select calibrationID from deploymentCalibrations")}
    cals: dict[int, dict] = {}
    for c in _rows(con, "select * from calibrations order by calibrationID"):
        if not linked or c["calibrationID"] in linked:
            cals[c["channelOrder"]] = c  # the deployment's calibration; the last one if none is linked
    sensors = {r["channelOrder"]: str(r["serialID"]) for r in _rows(con, "select * from instrumentSensors")}
    channels = []
    for r in _rows(con, "select c.*, ic.channelOrder, ic.channelStatus from channels c join instrumentChannels ic "
                        "using (channelID) order by ic.channelOrder"):
        order = int(r["channelOrder"])
        col = f"channel{order:02d}"
        cal = cals.get(order, {})
        coeffs = {k: v for k, v in (
            (x["key"], x["value"]) for x in _rows(con, "select key, value from coefficients where calibrationID = ?",
                                                   cal.get("calibrationID")))}
        channels.append(RskChannel(
            order=order, short_name=r["shortName"], long_name=r["longName"], units=r["units"] or "",
            units_plain=r["unitsPlainText"] or "", status=int(r["channelStatus"] or 0), derived=bool(r["isDerived"]),
            column=col if col in columns else None, equation=cal.get("equation") or "",
            calibration_ms=cal.get("tstamp"), coefficients=coeffs, sensor_serial=sensors.get(order, "")))
    nrows = _rows(con, "select count(*) as n from data")[0]["n"] if "tstamp" in columns else 0
    orphans = sorted(c for c in columns if c.startswith("channel") and c not in {ch.column for ch in channels})
    if orphans and nrows:  # seen only on empty tables (SN081015, Ruskin 2.18.2); never guess a mapping
        raise RskError(f"data-table columns {orphans} match no channel in instrumentChannels")
    drift = dep.get("loggerTimeDrift")
    return Rsk(
        path=Path(path), schema=str(db["version"]), kind=str(db["type"]),
        ruskin_version=app[0]["ruskinVersion"] if app else "", model=inst["model"], serial=int(inst["serialID"]),
        firmware=str(inst["firmwareVersion"]), fwtype=int(inst["firmwareType"]),
        part_number=inst.get("partNumber") or "", status=dep.get("loggerStatus") or "",
        drift_ms=int(drift) if drift not in (None, "") else None,
        download_logger_ms=int(dep["timeOfDownload"]) if dep.get("timeOfDownload") not in (None, "") else None,
        mode=sched["mode"], gate=sched.get("gate") or "", period_ms=period or 0,
        start_ms=epochs[0]["startTime"] if epochs else None, end_ms=epochs[0]["endTime"] if epochs else None,
        channels=channels, nrows=int(nrows),
        parameters={r["key"]: r["value"] for r in _rows(con, "select key, value from parameterKeys")})


def read_download(con: sqlite3.Connection) -> bytes | None:
    """The logger memory image Ruskin downloaded, reassembled from its parts; None if absent."""
    out = bytearray()
    for r in _rows(con, "select part, offset, data from downloads order by deploymentID, part"):
        if r["offset"] != len(out):
            raise RskError(f"downloads part {r['part']} starts at byte {r['offset']}, expected {len(out)}")
        out += r["data"]
    return bytes(out) if out else None


def read_values(con: sqlite3.Connection, rsk: Rsk) -> tuple[np.ndarray, np.ndarray]:
    """Ruskin's data table: (logger-clock ms, values[n, stored channel]); NULL -> NaN."""
    cols = [c.column for c in rsk.stored]
    t = np.empty(rsk.nrows, np.int64)
    v = np.empty((rsk.nrows, len(cols)))
    cur = con.execute(f"select tstamp, {', '.join(cols)} from data order by tstamp")
    i = 0
    while rows := cur.fetchmany(READ_BLOCK):
        a = np.array(rows, dtype=np.float64)  # tstamp < 2**53, so exact
        t[i : i + len(a)] = a[:, 0].astype(np.int64)
        v[i : i + len(a)] = a[:, 1:]
        i += len(a)
    if i != rsk.nrows:
        raise RskError(f"data table changed while reading ({i} of {rsk.nrows} rows)")
    return t, v


def _iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _compact(ms: int | None) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y%m%d%H%M%S") if ms is not None else ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def _float_or_str(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def make_record(rsk: Rsk, route: str, image: bytes | None) -> dict:
    """An rbr-offload-style record (see cli.offload) built from what Ruskin stored."""
    host_ms = rsk.download_host_ms
    if rsk.drift_ms is not None:
        skew = {"n": 1, "skew_vs_host_s": rsk.drift_ms / 1000, "uncertainty_s": math.nan, "spread_s": math.nan,
                "measured_at": _iso(host_ms),
                "method": f"Ruskin {rsk.ruskin_version} deployments.loggerTimeDrift: logger clock minus the Ruskin "
                          "computer's clock (positive = logger ahead), measured by Ruskin at download. That "
                          "computer's offset from UTC is unknown, hence clock_skew_s = NaN; Ruskin's method and "
                          "uncertainty are not documented."}
    else:
        skew = {"n": 0, "error": f"Ruskin recorded no clock drift for this download ({rsk.kind} file)"}
    stored = rsk.stored
    missing = [f"{c.short_name} ({c.long_name}, {'derived' if c.derived else 'measured'})"
               for c in rsk.channels if c.status & STATUS_NOT_STORED]
    extra = {
        "conversion_route": route,
        "rsk_file": rsk.path.name,
        "rsk_file_sha256": _sha256_file(rsk.path),
        "rsk_schema_version": rsk.schema,
        "rsk_type": rsk.kind,
        "ruskin_version": rsk.ruskin_version,
        "logger_time_at_download": _iso(rsk.download_logger_ms),
        "sampling_gate": rsk.gate,
        "ruskin_parameters": "; ".join(f"{k} = {v}" for k, v in sorted(rsk.parameters.items())),
    }
    if rsk.part_number:
        extra["instrument_part_number"] = rsk.part_number
    if missing:
        extra["ruskin_channels_not_stored"] = ", ".join(missing)
        extra["ruskin_channels_not_stored_comment"] = (
            "Channels the .rsk lists without values (channelStatus bit 0x04): measured channels the logger did not "
            "log, and derived channels Ruskin computes from the stored ones and ruskin_parameters. rbr-tpw does not "
            "recompute them.")
    how = ("memory image kept in the .rsk, decoded by rbr-tpw" if route == "decode"
           else f"values computed by Ruskin {rsk.ruskin_version} (.rsk data table)")
    return {
        "tool": {"name": "rbr-tpw", "version": __version__},
        "offload_started": _iso(host_ms if host_ms is not None else 0),
        "id": {"model": rsk.model, "version": rsk.firmware, "serial": rsk.serial_str, "fwtype": rsk.fwtype},
        "host_ntp": {},
        "clock_skew": skew,
        "snapshot_before": {
            "status": rsk.status,
            "sampling": {"mode": rsk.mode, "period": str(rsk.period_ms)},
            "channel_list": [{"type": c.short_name, "equation": c.equation,
                              "calibration_datetime": _compact(c.calibration_ms),
                              "coefficients": {k: _float_or_str(v) for k, v in c.coefficients.items()}}
                             for c in stored],
        },
        "raw": ({"file": f"{rsk.path.name} (downloads table)", "bytes": len(image),
                 "sha256": hashlib.sha256(image).hexdigest()} if image is not None else None),
        "warnings": [],
        "title": f"{rsk.model} SN{rsk.serial_str} data from Ruskin file {rsk.path.name}",
        "source": f"{rsk.model} SN{rsk.serial_str} downloaded by Ruskin {rsk.ruskin_version}; {how}",
        "history": f"converted from {rsk.path.name} by rbr-rsk2nc",
        "extra_attributes": extra,
    }


def compare_with_ruskin(d: Decoded, rsk: Rsk, t_ruskin: np.ndarray, v_ruskin: np.ndarray) -> tuple[str, list[str]]:
    """Compare a decoded memory image with Ruskin's values and logger times. Returns (summary, warnings)."""
    n = min(len(d.time_ms), len(t_ruskin))
    parts, warnings = [], []
    if len(d.time_ms) != len(t_ruskin):
        warnings.append(f"rbr-tpw decoded {len(d.time_ms)} sample sets, Ruskin stored {len(t_ruskin)}")
    for k, ch in enumerate(rsk.stored):
        eq = EQUATIONS.get(ch.equation)
        if eq is None:
            continue
        c = tuple(float(ch.coefficients.get(f"c{i}", "nan")) for i in range(4))
        ours, _ = eq(d.raw[:n, k], c)
        theirs = v_ruskin[:n, k]
        both = np.isfinite(ours) & np.isfinite(theirs)
        worst = float(np.max(np.abs(ours[both] - theirs[both]))) if both.any() else 0.0
        nan_mismatch = int((np.isfinite(ours) != np.isfinite(theirs)).sum())
        parts.append(f"{ch.short_name} max |difference| {worst:.1e}")
        if worst > VALUE_TOLERANCE or nan_mismatch:
            warnings.append(f"{ch.short_name}: decoded values differ from Ruskin's by up to {worst:.3g} "
                            f"({nan_mismatch} missing in one but not the other)")
    dt_ms = d.time_ms[:n] - t_ruskin[:n]
    ndiff = int(np.count_nonzero(dt_ms))
    if ndiff == 0:
        parts.append("logger times identical")
    else:
        median_s = np.median(dt_ms[dt_ms != 0]) / 1e3
        parts.append(f"logger times differ for {ndiff} sample sets (median {median_s:+.3f} s)")
        if not d.rtc_reset:
            warnings.append(f"{ndiff} sample times differ from Ruskin's without a clock reset")
        else:
            parts.append("expected after a clock reset: Ruskin times such samples from the deployment start")
    summary = (f"rbr-tpw decode vs Ruskin {rsk.ruskin_version} data table: {len(d.time_ms)} vs {len(t_ruskin)} "
               f"sample sets; " + "; ".join(parts))
    return summary, warnings


def _values_channels(rsk: Rsk, con: sqlite3.Connection, t: np.ndarray, v: np.ndarray) -> list[ChannelValues]:
    flags = np.zeros(v.shape, np.uint8)
    flags[~np.isfinite(v)] |= FLAG_ERROR_CODE
    col_of = {c.order: k for k, c in enumerate(rsk.stored)}
    for e in _rows(con, "select tstamp, sampleIndex, channelOrder from errors"):
        k = col_of.get(e["channelOrder"])
        i = int(np.searchsorted(t, e["tstamp"]))
        if k is None:
            continue
        if not (i < t.size and t[i] == e["tstamp"]):
            i = int(e["sampleIndex"]) - 1
        if 0 <= i < t.size:
            flags[i, k] |= FLAG_ERROR_CODE
    out = []
    for k, c in enumerate(rsk.stored):
        attrs = {"rbr_channel_long_name": c.long_name, "rbr_units": c.units_plain or c.units,
                 "rbr_channel_status": np.int32(c.status), "calibration_equation": c.equation,
                 "calibration_datetime": _compact(c.calibration_ms)}
        if c.sensor_serial:
            attrs["sensor_serial_number"] = c.sensor_serial
        attrs.update({f"calibration_{key}": _float_or_str(val) for key, val in c.coefficients.items()})
        out.append(ChannelValues(ctype=c.short_name, order=c.order, values=v[:, k], flags=flags[:, k], attrs=attrs,
                                 hidden=c.hidden))
    return out


def _values_events(con: sqlite3.Connection, t: np.ndarray) -> list[tuple[int, int, int]]:
    events = []
    for e in _rows(con, "select tstamp, type, sampleIndex from events order by tstamp, rowid"):
        si = int(e["sampleIndex"])
        index = si - 1 if si >= 1 else int(np.searchsorted(t, e["tstamp"]))  # EasyParse files store -1
        events.append((int(e["tstamp"]), int(e["type"]), index))
    return events


def _time_segments(t: np.ndarray, events: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """TFLAG_RESET_CLOCK on samples dated before 2001, and a new clock segment at each restart event."""
    tflags = np.where(t < RESET_CLOCK_BEFORE_MS, TFLAG_RESET_CLOCK, 0).astype(np.uint8)
    segment = np.zeros(t.size, np.int32)
    for _, etype, index in events:
        if etype in RESTART_EVENTS and 0 <= index < t.size:
            segment[index:] += 1
    return tflags, segment


@dataclass
class Result:
    route: str
    samples: int = 0
    t_first: int | None = None
    t_last: int | None = None
    warnings: list[str] = field(default_factory=list)
    note: str = ""


def convert(path: Path, out: Path, force_values: bool = False) -> Result:
    """Convert one .rsk file to `out` (written atomically). No file is written if there are no samples."""
    con = connect(path)
    try:
        rsk = read_rsk(path, con)
        image = read_download(con) if rsk.kind == "full" else None
        route = "decode" if (image is not None and rsk.fwtype in DECODABLE_FWTYPES and not force_values) else "values"
        t, v = read_values(con, rsk) if rsk.nrows else (np.empty(0, np.int64), np.empty((0, len(rsk.stored))))
        record = make_record(rsk, route, image)

        if route == "decode":
            d = decode(image, len(rsk.stored))
            summary, warns = compare_with_ruskin(d, rsk, t, v)
            record["extra_attributes"]["ruskin_comparison"] = summary
            record["warnings"].extend(warns)
            if d.header.serial != rsk.serial:
                record["warnings"].append(f"memory header serial {d.header.serial} differs from the .rsk's "
                                          f"{rsk.serial}")
            if not len(d.time_ms):
                return Result(route, note="no samples in the logger memory; nothing written")
            _, warnings, t_out = write_netcdf(image, record, out)
            return Result(route, len(t_out), int(t_out[0]) if len(t_out) else None,
                          int(t_out[-1]) if len(t_out) else None, warnings, summary)

        if not rsk.nrows:
            why = (f"; the raw download ({len(image)} bytes) is not decodable by rbr-tpw for fwtype {rsk.fwtype}"
                   if image is not None and rsk.fwtype not in DECODABLE_FWTYPES else "")
            return Result(route, note=f"Ruskin's data table is empty{why}; nothing written")
        events = _values_events(con, t)
        tflags, segment = _time_segments(t, events)
        channels = _values_channels(rsk, con, t, v)
        deployment = {"deployment_start_time": _iso(rsk.start_ms), "deployment_end_time": _iso(rsk.end_ms)}
        warnings, t_out = write_values_netcdf(
            t, tflags, segment, channels, events, record, out, period_ms=rsk.period_ms, deployment=deployment,
            values_comment=f"Value as computed by Ruskin {rsk.ruskin_version} and stored in the .rsk data table "
                           "(rbr-tpw does not decode this logger's memory format).",
            flag_comment="logger_error_code: Ruskin lists an error for this reading (errors table) or stored no "
                         "value. conversion_out_of_range is not used for Ruskin values.")
        return Result(route, len(t_out), int(t_out[0]) if len(t_out) else None,
                      int(t_out[-1]) if len(t_out) else None, warnings)
    finally:
        con.close()


def _inputs(paths: list[Path], outdir: Path) -> list[tuple[Path, Path]]:
    """(rsk, nc) pairs; a directory's sub-folders are reproduced under outdir."""
    pairs = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*.rsk")):
                if not f.name.startswith("."):  # skip macOS ._ resource-fork files
                    pairs.append((f, outdir / f.relative_to(p).with_suffix(".nc")))
        elif p.suffix.lower() == ".rsk":
            pairs.append((p, outdir / p.with_suffix(".nc").name))
        else:
            raise SystemExit(f"{p}: not a .rsk file or a directory")
    seen: dict[Path, Path] = {}
    for f, nc in pairs:
        if nc in seen:
            raise SystemExit(f"{f} and {seen[nc]} would both be written to {nc}")
        seen[nc] = f
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rbr-rsk2nc", description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path, help="directory for the NetCDF files")
    ap.add_argument("inputs", nargs="+", type=Path,
                    help=".rsk files, or directories searched recursively (their sub-folders are kept)")
    ap.add_argument("--force", action="store_true", help="overwrite existing NetCDF files (default: skip them)")
    ap.add_argument("--ruskin-values", action="store_true",
                    help="use Ruskin's computed values even for loggers rbr-tpw can decode")
    args = ap.parse_args(argv)

    pairs = _inputs(args.inputs, args.outdir)
    failed = 0
    for i, (f, nc) in enumerate(pairs, 1):
        tag = f"[{i}/{len(pairs)}] {f}"
        if nc.exists() and not args.force:
            print(f"{tag}: {nc} exists, skipped")
            continue
        nc.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = convert(f, nc, force_values=args.ruskin_values)
        except Exception as err:  # keep going; report at the end
            failed += 1
            print(f"{tag}: FAILED: {type(err).__name__}: {err}", flush=True)
            continue
        if not r.samples:
            print(f"{tag}: {r.route}: {r.note}", flush=True)
            continue
        print(f"{tag}: {r.route}: {r.samples} samples, {_iso(r.t_first)} to {_iso(r.t_last)} -> {nc}", flush=True)
        if r.note:
            print(f"    {r.note}")
        for w in r.warnings:
            print(f"    WARNING: {w}")
    if failed:
        sys.exit(f"{failed} of {len(pairs)} files failed")


if __name__ == "__main__":
    main()
