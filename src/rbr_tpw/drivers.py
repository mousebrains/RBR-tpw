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
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import solo
from .equations import is_sectioned_header, parse_l2_header
from .link import Link, LinkError, LoggerError
from .lock import unlocked_if_possible
from .ncwrite import ChannelValues, records_of, write_netcdf, write_values_netcdf
from .rawbin import FLAG_ERROR_CODE, event_indices, reset_segments
from .solo import STATUS_HIDDEN, STATUS_NOT_STORED, channel_status


class DecodeUnavailable(Exception):
    """No decoder for this memory yet. The raw download is saved; `rbr-offload --rebuild` can convert it later."""


class DeploymentMismatch(Exception):
    """The logger does not hold the deployment on disk: erased and enabled again (Ruskin or --configure), or a
    memory that cannot be explained. The offload starts a new deployment with a full download. `downloaded` is
    set when a full read was made before the mismatch was found, so it need not be repeated."""

    def __init__(self, reason: str, downloaded: Downloaded | None = None):
        super().__init__(reason)
        self.downloaded = downloaded


@dataclass
class Held:
    """A deployment's raw datasets on disk, verified against its latest record by the caller, and the header
    the logger reported at the latest offload when that differs from the saved image's (None: the same)."""

    stem: str
    data: dict[str, bytes]
    header: bytes | None = None


@dataclass
class Downloaded:
    """What one offload produced: every dataset complete (held bytes plus new), and what was read.

    The saved image only ever grows. A header the logger reports differently from the saved image's is not
    written into it: the logger writes its header at enable (L3 reference 5.3.1), so a change is an anomaly to
    record, and rewriting it would break every earlier record's checksum. It is kept in `header_now`."""

    data: dict[str, bytes]
    segments: dict[str, tuple[int, int]]  # dataset -> (offset, bytes) appended (offset 0: written whole)
    kind: str  # full | incremental | full-verified
    new_bytes: int = 0  # bytes the deployment grew by at this offload
    bytes_read: int = 0  # bytes read from the logger (header and tail checks, or the full image, included)
    tail_check: dict | None = None
    header_changed: list[int] = field(default_factory=list)  # header byte offsets changed since the previous offload
    header_now: bytes | None = None  # the logger's header, when it differs from the saved image's


def as_full(result: Downloaded) -> Downloaded:
    """A full read made before a mismatch was found, recast as a new deployment's download."""
    return Downloaded(result.data, {k: (0, len(v)) for k, v in result.data.items()}, "full",
                      new_bytes=sum(len(v) for v in result.data.values()), bytes_read=result.bytes_read)


def _q(link: Link, cmd: str) -> dict:
    """A query this model may not have: {} (and a transcript note) on a logger error."""
    try:
        return link.query(cmd)
    except LoggerError as err:
        link.note(f"{cmd!r} not available on this logger: {err}")
        return {}


def _iso_ms(ms: int | None) -> str:
    """Unix ms -> ISO-8601 UTC ('' if absent or zero)."""
    if not ms:
        return ""
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso(compact: str | None) -> str:
    """'YYYYMMDDhhmmss' -> ISO-8601 UTC ('' if absent or unparsable)."""
    try:
        t = dt.datetime.strptime(str(compact).strip(), "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
    except ValueError:
        return ""
    return t.isoformat().replace("+00:00", "Z")


def _channels(link: Link, count: int, easyparse: bool = True) -> list[dict]:
    out = []
    for i in range(1, count + 1):
        ch = link.query(f"channel {i}")
        cal = _q(link, f"calibration {i}")
        ch["index"] = i
        ch["status_as_reported"] = ch.get("status", "")
        derived = ch.get("derived", "").lower() == "on" or ch.get("equation", "").startswith("deri_")
        status = channel_status(ch["status_as_reported"], stored_if_on=easyparse or not derived)
        if status is None:  # not fatal: the download is still saved; a wrong channel count shows up in the decode
            link.note(f"channel {i}: status {ch['status_as_reported']!r} not understood; treated as a stored channel")
            status = 0
        ch["status"] = status
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
                      events: list[tuple[int, int, int]], record: dict | list[dict], path: Path,
                      values_comment: str, flag_comment: str, channels: list[dict] | None = None,
                      seen: list[tuple[int, int]] | None = None) -> tuple[list[str], np.ndarray]:
    """NetCDF from engineering values the logger stored with a timestamp per sample (Gen3 EasyParse, Gen4).
    `events` are (logger-clock ms, type, payload). `channels` describe the columns of `values` in order
    (default: the snapshot's channel_list). `record` is one offload record or the deployment's records in
    order; `seen[k]` = (sample sets, events) offload k's image held."""
    records = records_of(record)
    record = records[-1]
    snap = record["snapshot_before"]
    chans = snap["channel_list"] if channels is None else channels
    if values.shape[1] != len(chans):
        raise ValueError(f"{values.shape[1]} data columns but {len(chans)} channel descriptions")
    ev = [(int(ms), int(typ), i) for (ms, typ, _), i in
          zip(events, event_indices(time_ms, [int(e[0]) for e in events]), strict=True)]
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
    return write_values_netcdf(time_ms, tflags, segment, cvs, ev, records, path,
                               period_ms=int(snap.get("sampling", {}).get("period", 0) or 0), deployment=deployment,
                               values_comment=values_comment, flag_comment=flag_comment, seen=seen,
                               event_payloads=[int(e[2]) for e in events])


class Driver:
    family = "?"
    configurable = False  # --configure (writes to the logger): RBRsolo fwtype 9 and 0, both run on loggers
    energy_model = False  # power.py's RBRsolo T model applies
    l3 = False  # Gen3 `readdata` transfers

    def __init__(self, fwtype: int):
        self.fwtype = fwtype
        self.serial: int | None = None  # set once the logger is identified; lets the L2 snapshot unlock

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

    identity_dataset = "dataset1"  # the dataset (or its first bytes) that names a deployment
    identity_bytes: int | None = 512  # how many of its bytes; None = all of it
    growing = ("dataset1",)  # datasets a deployment extends; the others are rewritten whole at each offload

    def identity(self, data: dict[str, bytes]) -> bytes:
        blob = data.get(self.identity_dataset, b"")
        return blob if self.identity_bytes is None else blob[: self.identity_bytes]

    def identity_name(self, data: dict[str, bytes]) -> str:
        """Which dataset (or object) the identity comes from, for the record."""
        return self.identity_dataset

    def primary(self, data: dict[str, bytes]) -> str:
        """The dataset the record's `raw` and the NetCDF describe: the samples, or whatever is not empty."""
        if data.get("dataset1"):
            return "dataset1"
        names = [k for k, v in data.items() if v]
        return next((k for k in names if k.endswith("/data")), next(iter(names), ""))

    def grows(self, name: str) -> bool:
        """Does a deployment extend this dataset (as against rewriting it whole at each offload)?"""
        return name in self.growing

    def download(self, link: Link, snap: dict, part_dir: Path, sn: str, progress=None, held: Held | None = None,
                 full: bool = False) -> Downloaded:
        """Every dataset, whole. With `held`, the deployment on disk, the new image must extend it (each growing
        dataset starts with the old bytes, the identity dataset is unchanged), else DeploymentMismatch."""
        data = self._download_all(link, snap, part_dir, sn, progress)
        read = sum(len(v) for v in data.values())
        if held is None:
            return Downloaded(data, {k: (0, len(v)) for k, v in data.items()}, "full", new_bytes=read,
                              bytes_read=read)
        # a growing dataset is appended past what was held; the others are written whole
        segments = {k: (len(held.data[k]), len(v) - len(held.data[k])) if self.grows(k) and k in held.data
                    and len(v) >= len(held.data[k]) else (0, len(v)) for k, v in data.items()}
        grew = sum(n for k, (o, n) in segments.items() if o or (self.grows(k) and k not in held.data))
        result = Downloaded(data, segments, "full-verified", new_bytes=grew, bytes_read=read)
        reason = self.growth_mismatch(held.data, data)
        if reason:
            raise DeploymentMismatch(reason, result)
        return result

    def growth_mismatch(self, old: dict[str, bytes], new: dict[str, bytes]) -> str | None:
        if self.identity(old) != self.identity(new):
            return f"the {self.identity_dataset} header differs"
        for name, blob in old.items():
            if name not in new:
                return f"{name} is no longer on the logger"
            if self.grows(name) and not new[name].startswith(blob):
                first = next((i for i, (a, b) in enumerate(zip(blob, new[name], strict=False)) if a != b),
                             min(len(blob), len(new[name])))
                return f"{name} does not extend the deployment image (first difference at byte {first})"
        return None

    def _download_all(self, link: Link, snap: dict, part_dir: Path, sn: str, progress=None) -> dict[str, bytes]:
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
        self.configurable = fwtype in (0, 9)
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
        # duets and concertos hide their compensation thermistor while locked; solos have one channel and no lock
        # to speak of, and fwtype 0 is untested with `lock OFF`, so only 102/103 unlock
        snap = solo.snapshot(link, unlock_serial=self.serial if self.fwtype in (102, 103) else None)
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
        return f"{sn}.new.0.part"  # a new deployment, from byte 0

    def bytes_per_sample(self, snap: dict) -> int | None:
        return 4 * len(snap["channel_list"])

    @staticmethod
    def header_length(image: bytes) -> int:
        """The memory header's own length: 512 on the RBRsolo (fwtypes 0 and 9, 111 of 111 .rsk images); a
        sectioned header on the duet and concerto has its own, not a multiple of 4 (715, 939, 1037 seen)."""
        if len(image) < 64:
            return len(image)
        if is_sectioned_header(image):
            try:
                return int(parse_l2_header(image).length)
            except Exception:  # noqa: BLE001  (unparseable: treat the whole image as header, so it is compared)
                return len(image)
        return min(512, len(image))

    def identity(self, data):
        blob = data.get("dataset1", b"")
        return blob[: self.header_length(blob)]

    def download(self, link: Link, snap: dict, part_dir: Path, sn: str, progress=None, held: Held | None = None,
                 full: bool = False) -> Downloaded:
        """dataset 1: all of it for a new deployment or with `full`; otherwise only the bytes past the held image,
        after the identity check: the first and the last block of the held image are read back and compared (the
        first holds the header and the deployment's first time anchor, which equal data tails cannot vouch for),
        and a header that changed is accepted only after the whole image has been read and verified.

        Only the RBRsolo's 512-byte header reads incrementally. The duet and concerto (sectioned headers) are
        read in full and checked to extend the saved image, as a concerto3 is, until the incremental path has
        run on one (their .rsk downloads are append-only: 1 duet and 3 concerto pairs, 2026-09-28)."""
        total = int(snap["meminfo"]["used"])
        old = held.data.get("dataset1", b"") if held is not None else b""
        if held is not None and is_sectioned_header(old):
            return super().download(link, snap, part_dir, sn, progress, held=held, full=full)
        n_old = len(old)
        hl = self.header_length(old) if old else 512
        prev_header = held.header if held is not None and held.header is not None else old[:hl]

        def header_check(now: bytes) -> tuple[list[int], bytes | None]:
            changed = [i for i in range(min(hl, len(now), len(prev_header))) if now[i] != prev_header[i]]
            if changed and n_old <= hl:
                raise DeploymentMismatch("the header differs and there is no data to check")
            return changed, (now[:hl] if now[:hl] != old[:hl] else None)

        if held is None or full:
            image = solo.download(link, total, part_dir / self.part_name(sn, 1), progress) if total > 0 else b""
            if held is None:
                return Downloaded({"dataset1": image}, {"dataset1": (0, len(image))}, "full", new_bytes=len(image),
                                  bytes_read=len(image))
            result = Downloaded({"dataset1": old + image[n_old:]}, {"dataset1": (n_old, max(0, len(image) - n_old))},
                                "full-verified", new_bytes=max(0, len(image) - n_old), bytes_read=len(image))
            fresh = Downloaded({"dataset1": image}, {}, "full", bytes_read=len(image))  # if it is another deployment
            if len(image) < n_old:
                raise DeploymentMismatch(f"the logger holds {len(image)} bytes, the deployment image {n_old}", fresh)
            if image[hl:n_old] != old[hl:n_old]:
                first = next(i for i in range(hl, n_old) if image[i] != old[i])
                raise DeploymentMismatch(f"memory differs from the deployment image at byte {first}", fresh)
            try:
                result.header_changed, result.header_now = header_check(image)
            except DeploymentMismatch as err:
                raise DeploymentMismatch(str(err), fresh) from None
            return result
        if total < n_old:
            raise DeploymentMismatch(f"the logger holds {total} bytes, the deployment image {n_old}")
        head_len = min(solo.CHUNK, n_old)  # the header and the start of the data held
        head = link.read_data(1, head_len, 0)
        if len(head) != head_len:
            raise LinkError(f"short head read: {len(head)} of {head_len} bytes")
        n = min(solo.CHUNK, n_old - head_len)  # the end of the data held, unless the first block covered it
        tail = link.read_data(1, n, n_old - n) if n else b""
        if len(tail) != n:
            raise LinkError(f"short tail read: {len(tail)} of {n} bytes at {n_old - n}")
        tail_check = {"head_bytes": head_len, "offset": n_old - n, "bytes": n, "ok": True}
        for block, at in ((head, 0), (tail, n_old - n)):
            lo = max(hl, at)
            if block[lo - at:] != old[lo:at + len(block)]:
                first = next(i for i in range(lo, at + len(block)) if block[i - at] != old[i])
                raise DeploymentMismatch(f"memory differs from the deployment image at byte {first}")
        header_changed, header_now = header_check(head)
        if header_changed:  # never seen within a deployment: accept it only once every byte held is verified
            link.note(f"header changed at byte(s) {header_changed}: reading the whole image to verify it")
            result = self.download(link, snap, part_dir, sn, progress, held=held, full=True)
            result.bytes_read += head_len + n
            return result
        segment = b""
        if total > n_old:
            segment = solo.download(link, total, part_dir / f"{sn}.{held.stem}.{n_old}.part", progress, start=n_old)
        return Downloaded({"dataset1": old + segment}, {"dataset1": (n_old, len(segment))}, "incremental",
                          new_bytes=len(segment), bytes_read=head_len + n + len(segment), tail_check=tail_check,
                          header_changed=header_changed, header_now=header_now)

    def write_netcdf(self, data, record, path):
        fmt = (records_of(record)[-1]["snapshot_before"].get("memformat") or {}).get("type", "rawbin00")
        if fmt != "rawbin00":
            raise DecodeUnavailable(f"memory format {fmt!r} is not decoded yet (only rawbin00)")
        _, warnings, t_ms = write_netcdf(data["dataset1"], record, path)
        return warnings, t_ms


class Gen3Driver(Driver):
    family = "Gen3"
    l3 = True
    identity_dataset = "dataset2"  # the deployment header
    identity_bytes = None
    growing = ("dataset1", "dataset0")  # samples and events both grow; the header is rewritten

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
                "sampling": _q(link, "sampling")}
        snap["memformat"] = _q(link, "memformat")
        easyparse = snap["memformat"].get("type") == "calbin00"  # rawbin00 never stores derived channels (4.7.2)
        # unlocked, as Ruskin reads it: locked, the logger leaves hidden channels out and answers `status = on`
        with unlocked_if_possible(link, self.serial, clock_cmd="clock") as unlocked:
            snap["channels"] = link.query("channels")
            snap["channels_read_unlocked"] = unlocked
            snap["channels_all"] = _channels(link, int(snap["channels"]["count"]), easyparse=easyparse)
        snap["channel_list"] = [c for c in snap["channels_all"] if not c["status"] & STATUS_NOT_STORED]
        snap["dataset_meminfo"] = {str(d): self._meminfo(link, d) for d in (0, 1, 2)}
        snap["meminfo"] = snap["dataset_meminfo"]["1"]
        snap["power"] = self._power(link)
        snap["settings"] = _q(link, "settings")
        snap["info"] = _q(link, "info")
        off = clock.get("offsetfromutc", "")
        try:
            nonzero = float(off.replace("+", "") or 0) != 0
        except ValueError:  # "unknown": the default, and what setting the clock leaves (L3 ref 4.1.1)
            nonzero = False
        if nonzero:
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
        records = records_of(record)
        record = records[-1]
        fmt = (record["snapshot_before"].get("memformat") or {}).get("type", "")
        if fmt != "calbin00":
            raise DecodeUnavailable(f"Gen3 memory format {fmt!r} is not decoded yet (only EasyParse, calbin00)")
        try:
            from .easyparse import decode_easyparse
        except ImportError as err:
            raise DecodeUnavailable("the EasyParse decoder is not installed yet") from err
        # dataset 1 may be absent: events but no samples (a gated logger never activated). Live, download() gives
        # b""; --rebuild only has the datasets the record lists, which are the non-empty ones.
        nchan = len(record["snapshot_before"]["channel_list"])
        ep = decode_easyparse(data.get("dataset1", b""), nchan, data.get("dataset0") or None)
        # what each earlier offload's image held: dataset 1 records are 8 + 4 * nchan bytes, events 16
        seen = []
        for rec in records:
            ds = rec.get("datasets") or {}
            if rec is record or not ds:
                seen.append((ep.time_ms.size, len(ep.events)))
            else:
                b1 = int((ds.get("dataset1") or {}).get("bytes", 0) or 0)
                b0 = int((ds.get("dataset0") or {}).get("bytes", 0) or 0)
                seen.append((b1 // (8 + 4 * nchan), sum(1 for o in ep.event_offsets if o + 16 <= b0)))
        warns = []
        if ep.trailing_bytes:
            warns.append(f"{ep.trailing_bytes} trailing bytes in dataset 1 were ignored")
        if ep.event_trailing_bytes:
            warns.append(f"{ep.event_trailing_bytes} trailing bytes in dataset 0 (events) were ignored")
        if ep.bad_events:
            warns.append(f"{ep.bad_events} event records failed their CRC or marker check and were ignored")
        if ep.clock_resets:
            warns.append(f"the logger's clock restarted {ep.clock_resets} time(s) during the deployment")
        # a copy: the caller's record already went to the JSON file, and it prints what is new in the result
        record = {**record, "warnings": [*record.get("warnings", []), *warns]}
        return write_engineering(
            ep.time_ms, ep.values, ep.error_codes, ep.events, [*records[:-1], record], path,
            values_comment="Engineering value as stored by the logger (Gen3 EasyParse, L3 command reference 5.2).",
            flag_comment="logger_error_code: the logger stored a NaN error code (L3 ref 5.2.1) for this reading. "
                         "conversion_out_of_range is not used.", seen=seen)


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

    identity_bytes = None

    def selected(self, data: dict[str, bytes]) -> str | None:
        """The dataset write_netcdf converts (the latest; one dataset per deployment, L3.5 ref 2.1.6)."""
        try:
            return self.g._pick(data, None, None)
        except ValueError:
            return None

    def identity(self, data):
        ds = self.selected(data)
        return data.get(f"{ds}/meta", b"") if ds else b""

    def identity_name(self, data):
        ds = self.selected(data)
        return f"{ds}/meta" if ds else ""

    def primary(self, data):
        try:
            ds, sch, _, _, _ = self.g.columns(data, None)
            return f"{ds}/{sch}/data"
        except (ValueError, KeyError):
            return super().primary(data)

    def grows(self, name):  # every object but the metadata may grow
        return not name.endswith("/meta")

    def _download_all(self, link, snap, part_dir, sn, progress=None):
        return self.g.download(link, part_dir, sn, progress)

    def write_netcdf(self, data, record, path):
        records = records_of(record)
        record = records[-1]
        snap = record["snapshot_before"]
        ds, sch, cols, meta, fmt = self.g.columns(data, snap)
        time_ms, values, error_codes, events = self.g.decode(data, snap, ds, sch)
        # what each earlier offload's objects held: fixed-size sample records, and the events (of this schedule)
        # that ended within its events object
        rec_size = 8 + np.dtype(fmt).itemsize * len(cols)
        bit = 1 << (next(x["index"] for x in meta.schedules if x["label"] == sch) - 1)
        ev_ends, ev, pos = [], data.get(f"{ds}/events", b""), 0
        while pos + 24 <= len(ev):
            _, mask, size = struct.unpack_from("<QIH", ev, pos)
            step = size if size >= 24 else 24
            if mask == 0 or mask & bit:
                ev_ends.append(pos + step)
            pos += step
        seen = []
        for rec in records:
            objs = rec.get("datasets") or {}
            if rec is record or not objs:
                seen.append((time_ms.size, len(events)))
                continue
            b_data = int((objs.get(f"{ds}/{sch}/data") or {}).get("bytes", 0) or 0)
            b_ev = int((objs.get(f"{ds}/events") or {}).get("bytes", 0) or 0)
            seen.append((min(b_data // rec_size, time_ms.size), sum(1 for e in ev_ends if e <= b_ev)))
        # the columns as the dataset's own metadata describes them, not the logger's current channel list
        channels = [{"index": c["index"], "type": c.get("type", ""), "label": c.get("label", ""),
                     "userunits": c.get("userunits", ""), "status": STATUS_HIDDEN if c.get("hidden") else 0,
                     "equation": c.get("equation", ""), "coefficients": c.get("coefficients", {}),
                     "calibration_datetime": _iso_ms(c.get("calibration_ms"))} for c in cols]
        others = sorted(k[: -len("/data")] for k, v in data.items()
                        if k.endswith("/data") and v and k != f"{ds}/{sch}/data")
        warns = [f"this NetCDF holds dataset {ds} schedule {sch} only; also downloaded but NOT converted: "
                 f"{', '.join(others)} (saved in raw/)"] if others else []
        record = {**record, "warnings": [*record.get("warnings", []), *warns],
                  "raw": (record.get("datasets") or {}).get(f"{ds}/{sch}/data", record.get("raw"))}
        return write_engineering(time_ms, values, error_codes, events, [*records[:-1], record], path,
                                 values_comment="Engineering value as stored by the logger (Gen4, L3.5 reference "
                                                "section 4.2; decoder untested on a real logger).",
                                 flag_comment="logger_error_code: the logger stored a NaN error code for this reading.",
                                 channels=channels, seen=seen)


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
