"""Per-family logger drivers: read the clock, the settings and the memory (all read-only toward the
logger), and turn a download into NetCDF.

Families by the `id` fwtype, and what each rests on:

  L2    0 and 9 (RBRsolo), 102 (RBRduet), 103 (RBRconcerto). The commands Ruskin 2.26.1 sends these loggers
        (~/Ruskin/logs/ruskin_serial.log*, 2026-09-10..25): `now`, `status`, `starttime`, `endtime`,
        `sampling`, `channels`, `channel N`, `calibration N`, `meminfo`, `powerstatus`, and
        `read data 1 <size> <offset>` in 68000-byte blocks. Run live on SN100689 (fwtype 9) only.
  Gen3  104 (RBRconcerto3 and the other L3 loggers): `clock`, `deployment`, `sampling`, `channels`,
        `channel N`, `calibration N`, `memformat`, `meminfo dataset N`, `power`, `powerinternal`, and
        `readdata size = <s>, offset = <o>, dataset = <d>` for datasets 2 (deployment header), 0 (events)
        and 1 (samples). L3 command reference, and the same commands in Ruskin's logs for SN233442 and
        SN243188. Not yet run live.
  Gen4  120: L3.5 command reference only (gen4.py); no Gen4 logger has been tested.

Only fwtype 9 can be configured (--configure); the others are offloaded read-only.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np

from . import solo
from .link import Link, LoggerError
from .ncwrite import ChannelValues, write_netcdf, write_values_netcdf
from .rawbin import FLAG_ERROR_CODE, reset_segments

STATUS_HIDDEN = 0x01  # channelStatus bits (L3 ref; RBR's pyRSKtools)
STATUS_NOT_STORED = 0x04


class DecodeUnavailable(Exception):
    """No decoder for this memory yet. The raw download is saved; `rbr-offload --rebuild` can convert it later."""


def _q(link: Link, cmd: str) -> dict:
    """A query this model may not have: {} (and a transcript note) on a logger error."""
    try:
        return link.query(cmd)
    except LoggerError as err:
        link.note(f"{cmd!r} not available on this logger: {err}")
        return {}


def _iso(compact: str | None) -> str:
    """'YYYYMMDDhhmmss' -> ISO-8601 UTC ('' if absent or unparsable)."""
    try:
        t = dt.datetime.strptime(str(compact).strip(), "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
    except ValueError:
        return ""
    return t.isoformat().replace("+00:00", "Z")


def _channels(link: Link, count: int) -> list[dict]:
    out = []
    for i in range(1, count + 1):
        ch = link.query(f"channel {i}")
        cal = _q(link, f"calibration {i}")
        ch["index"] = i
        ch["status"] = int(ch.get("status", "0") or 0)
        ch["calibration_datetime"] = cal.pop("datetime", "")
        cal.pop("type", None)
        cal.pop("label", None)
        ch["coefficients"] = {k: solo._coefficient(v) for k, v in cal.items()}
        ch["coefficients_as_reported"] = cal
        out.append(ch)
    return out


def _no_energy(pwr: dict, comment: str) -> dict:
    pwr.update({"energy_remaining_J": math.nan, "energy_nominal_J": math.nan, "energy_remaining_fraction": math.nan,
                "comment": comment})
    return pwr


def write_engineering(time_ms: np.ndarray, values: np.ndarray, error_codes: np.ndarray,
                      events: list[tuple[int, int, int]], record: dict, path: Path, values_comment: str,
                      flag_comment: str) -> tuple[list[str], np.ndarray]:
    """NetCDF from engineering values the logger stored with a timestamp per sample (Gen3 EasyParse, Gen4).
    `events` are (logger-clock ms, type, payload)."""
    snap = record["snapshot_before"]
    chans = snap["channel_list"]
    ev = [(int(ms), int(typ), int(np.searchsorted(time_ms, ms))) for ms, typ, _ in events]
    tflags, segment = reset_segments(time_ms, ev)
    cvs = []
    for k, c in enumerate(chans):
        flags = np.zeros(time_ms.size, np.uint8)
        flags[(error_codes[:, k] != 0) | ~np.isfinite(values[:, k])] |= FLAG_ERROR_CODE
        attrs = {"rbr_channel_status": np.int32(c["status"]), "rbr_units": c.get("userunits", ""),
                 "calibration_equation": c.get("equation", ""),
                 "calibration_datetime": c.get("calibration_datetime", "")}
        if c.get("label"):
            attrs["rbr_channel_label"] = c["label"]
        attrs.update({f"calibration_{k2}": v for k2, v in c.get("coefficients", {}).items()})
        cvs.append(ChannelValues(ctype=c.get("type", ""), order=c["index"], values=values[:, k], flags=flags,
                                 attrs=attrs, hidden=bool(c["status"] & STATUS_HIDDEN)))
    deployment = {"deployment_start_time": _iso(snap.get("starttime")),
                  "deployment_end_time": _iso(snap.get("endtime"))}
    return write_values_netcdf(time_ms, tflags, segment, cvs, ev, record, path,
                               period_ms=int(snap.get("sampling", {}).get("period", 0) or 0), deployment=deployment,
                               values_comment=values_comment, flag_comment=flag_comment)


class Driver:
    family = "?"
    configurable = False  # --configure (writes to the logger) is implemented and tested for fwtype 9 only
    energy_model = False  # power.py's RBRsolo T model applies
    l3 = False  # Gen3 `readdata` transfers

    def __init__(self, fwtype: int):
        self.fwtype = fwtype

    def clock_now(self, link: Link) -> str:
        raise NotImplementedError

    def snapshot(self, link: Link) -> dict:
        raise NotImplementedError

    def after(self, link: Link) -> dict:
        raise NotImplementedError

    def datasets(self, snap: dict) -> list[tuple[str, int, int]]:
        """(name, dataset number, bytes) to download, in download order."""
        raise NotImplementedError

    def part_name(self, sn: str, number: int) -> str:
        return f"{sn}.dataset{number}.part"

    def part_files(self, part_dir: Path, sn: str) -> list[Path]:
        """Partial downloads of this logger, removed once the raw files are saved."""
        return sorted(part_dir.glob(f"{sn}.*part"))

    def bytes_per_sample(self, snap: dict) -> int | None:
        return None

    def download(self, link: Link, snap: dict, part_dir: Path, sn: str, progress=None) -> dict[str, bytes]:
        plan = self.datasets(snap)
        grand = sum(n for _, _, n in plan)
        out: dict[str, bytes] = {}
        base = 0
        for name, number, size in plan:
            if size <= 0:
                out[name] = b""
                continue

            def prog(done, total, rate, base=base):
                if progress:
                    progress(base + done, grand, rate)

            out[name] = solo.download(link, size, part_dir / self.part_name(sn, number), prog, dataset=number,
                                      l3=self.l3)
            base += size
        return out

    def write_netcdf(self, data: dict[str, bytes], record: dict, path: Path) -> tuple[list[str], np.ndarray]:
        raise DecodeUnavailable(f"no decoder for fwtype {self.fwtype}")


class L2Driver(Driver):
    family = "L2"

    def __init__(self, fwtype: int):
        super().__init__(fwtype)
        self.configurable = fwtype == 9
        self.energy_model = fwtype in (0, 9)  # RBRsolo T constants; fwtype 0 has no energy counter

    def clock_now(self, link: Link) -> str:
        return link.query("now")["now"]

    def _power(self, link: Link, pwr: dict | None = None) -> dict:
        """solo.power(), plus what differs by model; `pwr` is a powerstatus already read."""
        pwr = pwr if pwr is not None else solo.power(link)
        if self.fwtype == 9:
            return pwr
        if self.fwtype == 102 and not pwr.get("remaining_raw"):  # Ruskin asks the duet separately
            pwr["remaining_raw"] = _q(link, "powerstatus remaining").get("remaining", "")
        return _no_energy(pwr, "From `powerstatus`. battery_voltage_V = int (volts when it has a decimal point, else "
                               "millivolts, as on the RBRsolo). The energy counter and capacity are recorded as "
                               "reported (battery_powerstatus_raw); their units are not verified for this model and "
                               "no energy estimate is made.")

    def snapshot(self, link: Link) -> dict:
        snap = solo.snapshot(link)
        snap["power"] = self._power(link, snap["power"])
        snap["channels_all"] = snap["channel_list"]
        snap["channel_list"] = [c for c in snap["channels_all"] if not c["status"] & STATUS_NOT_STORED]
        if self.fwtype in (102, 103):
            snap["memformat"] = _q(link, "memformat")
            snap["settings"] = _q(link, "settings")
        return snap

    def after(self, link: Link) -> dict:
        return {"meminfo": solo.memory(link), "power": self._power(link), "status": link.query("status")["status"]}

    def datasets(self, snap: dict) -> list[tuple[str, int, int]]:
        return [("dataset1", 1, int(snap["meminfo"]["used"]))]

    def part_name(self, sn: str, number: int) -> str:
        return f"{sn}.bin.part"  # the name rbr-offload has always used, so older partial downloads resume

    def bytes_per_sample(self, snap: dict) -> int | None:
        return 4 * len(snap["channel_list"])

    def write_netcdf(self, data, record, path):
        fmt = (record["snapshot_before"].get("memformat") or {}).get("type", "rawbin00")
        if fmt != "rawbin00":
            raise DecodeUnavailable(f"memory format {fmt!r} is not decoded yet (only rawbin00)")
        _, warnings, t_ms = write_netcdf(data["dataset1"], record, path)
        return warnings, t_ms


class Gen3Driver(Driver):
    family = "Gen3"
    l3 = True

    def clock_now(self, link: Link) -> str:
        return link.query("clock")["datetime"]

    def _meminfo(self, link: Link, number: int) -> dict:
        m = _q(link, f"meminfo dataset {number}")
        return {k: int(m.get(k, 0) or 0) for k in ("used", "remaining", "size")}

    def _power(self, link: Link) -> dict:
        p = _q(link, "power")
        pi = _q(link, "powerinternal")
        pwr = {"source": p.get("source", ""), "int_raw": p.get("int", ""), "ext_raw": p.get("ext", ""),
               "remaining_raw": "", "battery_voltage_V": solo.parse_voltage(p.get("int", "")),
               "powerinternal_raw": ", ".join(f"{k} = {v}" for k, v in pi.items())}
        return _no_energy(pwr, "From `power` (int, V) and `powerinternal` (recorded as reported in "
                               "powerinternal_raw; L3 ref 4.9). No energy estimate is made for Gen3 loggers.")

    def snapshot(self, link: Link) -> dict:
        clock = link.query("clock")
        dep = link.query("deployment")
        snap = {"now": clock.get("datetime", ""), "clock": clock, "status": dep.get("status", ""),
                "starttime": dep.get("starttime", ""), "endtime": dep.get("endtime", ""),
                "sampling": _q(link, "sampling"), "channels": link.query("channels")}
        snap["channels_all"] = _channels(link, int(snap["channels"]["count"]))
        snap["channel_list"] = [c for c in snap["channels_all"] if not c["status"] & STATUS_NOT_STORED]
        snap["memformat"] = _q(link, "memformat")
        snap["dataset_meminfo"] = {str(d): self._meminfo(link, d) for d in (0, 1, 2)}
        snap["meminfo"] = snap["dataset_meminfo"]["1"]
        snap["power"] = self._power(link)
        snap["settings"] = _q(link, "settings")
        snap["info"] = _q(link, "info")
        off = clock.get("offsetfromutc", "")
        if off and float(off.replace("+", "") or 0) != 0:
            link.note(f"logger clock offsetfromutc = {off}: its clock may be local time, not UTC")
        return snap

    def after(self, link: Link) -> dict:
        return {"meminfo": self._meminfo(link, 1), "power": self._power(link),
                "status": _q(link, "deployment status").get("status", "")}

    def datasets(self, snap: dict) -> list[tuple[str, int, int]]:
        m = snap["dataset_meminfo"]
        return [("dataset2", 2, m["2"]["used"]), ("dataset0", 0, m["0"]["used"]), ("dataset1", 1, m["1"]["used"])]

    def bytes_per_sample(self, snap: dict) -> int | None:
        if (snap.get("memformat") or {}).get("type") == "calbin00":
            return 8 + 4 * len(snap["channel_list"])  # EasyParse: int64 ms + float32 per stored channel
        return None

    def write_netcdf(self, data, record, path):
        fmt = (record["snapshot_before"].get("memformat") or {}).get("type", "")
        if fmt != "calbin00":
            raise DecodeUnavailable(f"Gen3 memory format {fmt!r} is not decoded yet (only EasyParse, calbin00)")
        try:
            from .easyparse import decode_easyparse
        except ImportError as err:
            raise DecodeUnavailable("the EasyParse decoder is not installed yet") from err
        ep = decode_easyparse(data["dataset1"], len(record["snapshot_before"]["channel_list"]),
                              data.get("dataset0") or None)
        if ep.trailing_bytes:
            record.setdefault("warnings", []).append(f"{ep.trailing_bytes} trailing bytes in dataset 1 were ignored")
        return write_engineering(
            ep.time_ms, ep.values, ep.error_codes, ep.events, record, path,
            values_comment="Engineering value as stored by the logger (Gen3 EasyParse, L3 command reference 5.2).",
            flag_comment="logger_error_code: the logger stored a NaN error code (L3 ref 5.2.1) for this reading. "
                         "conversion_out_of_range is not used.")


class Gen4Driver(Driver):
    """Thin wrapper over gen4.py (L3.5 reference only; untested on a logger)."""

    family = "Gen4"

    def __init__(self, fwtype: int):
        super().__init__(fwtype)
        from . import gen4  # ImportError -> not supported

        self.g = gen4

    def clock_now(self, link):
        return self.g.clock_now(link)

    def snapshot(self, link):
        snap = self.g.snapshot(link)
        snap.setdefault("channels_all", snap["channel_list"])
        snap["channel_list"] = [c for c in snap["channels_all"] if not int(c.get("status", 0)) & STATUS_NOT_STORED]
        _no_energy(snap["power"], "From the L3.5 `instrument power` commands; no energy estimate is made for Gen4.")
        return snap

    def after(self, link):
        s = self.snapshot(link)
        return {"meminfo": s["meminfo"], "power": s["power"], "status": s.get("status_l2", s["status"])}

    def datasets(self, snap):
        return [("datasets", 0, int(snap["meminfo"]["used"]))]

    def part_files(self, part_dir: Path, sn: str) -> list[Path]:
        return sorted(part_dir.glob(f"{sn}__*.part"))

    def download(self, link, snap, part_dir, sn, progress=None):
        return self.g.download(link, part_dir, sn, progress)

    def write_netcdf(self, data, record, path):
        time_ms, values, error_codes, events = self.g.decode(data, record["snapshot_before"])
        return write_engineering(time_ms, values, error_codes, events, record, path,
                                 values_comment="Engineering value as stored by the logger (Gen4, L3.5 reference "
                                                "section 4.2; decoder untested on a real logger).",
                                 flag_comment="logger_error_code: the logger stored a NaN error code for this reading.")


def driver_for(fwtype: int) -> Driver | None:
    if fwtype in (0, 9, 102, 103):
        return L2Driver(fwtype)
    if fwtype == 104:
        return Gen3Driver(fwtype)
    if fwtype == 120:
        try:
            return Gen4Driver(fwtype)
        except ImportError:
            return None
    return None
