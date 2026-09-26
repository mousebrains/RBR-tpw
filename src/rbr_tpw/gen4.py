"""RBR Gen4 loggers ("L3.5" command set, `id fwtype` 120): read-only queries, download, and decode.

Written from the RBR Gen4 L3.5 Command Reference, rev B (RBR#0014818revB, refs/) ALONE. No Gen4 logger
and no Gen4 data have been available, so nothing here has been checked against a real instrument; section
numbers below cite that reference. Where it is ambiguous the conservative, read-only reading is taken and
the choice is noted.

Everything here only reads from the logger. The commands used are `id`, `instrument`, `instrument power
...`, `clock`, `deployment`, `storage`, `dataset ...`, `schedule <label>`, `group <label>`, `channel ...`,
`calibration <label>`, `parameters`, and `download` (all Open, or read-only uses of Unsafe commands, which
are always allowed: 1.3 "Reading parameters is always available").

Memory model (2.1.6, 2.7, 3.5): memory holds up to `maxcount` datasets (one per deployment). Each dataset
has one `meta` object, one `events` object, and one `data` object per schedule, addressed as
`<dataset>/meta`, `<dataset>/events` and `<dataset>/<schedule>/data`. download() fetches every object of
every dataset, so nothing on the logger is left behind; decode() turns one (dataset, schedule) into arrays.
"""

from __future__ import annotations

import math
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .crc import crc16_ccitt
from .link import Link, LinkError, LoggerError, _show

FWTYPES = {120}  # 2.6.2, 3.3.3 "120 for L3.5 instruments". (3.3.1 examples: 130 RBRsolo4, 150 RBRoem; unverified)
CHUNK = 32_000  # bytes per `download`; the largest size used in the 3.5.2 examples
HEAD = 256  # bytes compared to decide whether a partial download can be resumed

_PROMPT = re.compile(r"^\s*(?:ready:\s*)+", re.I)  # 1.1.4.1 shows both "Ready:" and "ready:"
_ERRWRN = re.compile(r"^(ERR|WRN)-(\d+)\s*(.*)$", re.I)  # 7: "ERR-102 invalid command '...'", "WRN-408 ..."
_DL_HDR = re.compile(rb"download\s+(\S+)\s+([^\r\n]*)\r\n", re.I)  # 3.5.2 response line
_DL_ERR = re.compile(rb"(ERR-\d+[^\r\n]*)\r\n", re.I)
_PROMPT_B = re.compile(rb"(?:Ready|ready): ?")

# 4.3.7.3 sampling mode codes
MODES = {0: "continuous", 1: "average", 2: "tide", 3: "burst", 4: "wave", 5: "ddsampling", 6: "regimes", 7: "unknown"}
# 4.3.5.1 memory format codes -> numpy dtype of each stored value
DATA_FORMATS = {1: ("float32", "<f4"), 2: ("float64", "<f8"), 3: ("calfloat64", "<f8")}
# 4.4.2 event types (names for the documented ones; "reserved" codes are left unnamed)
EVENT_NAMES = {
    0x00: "unknown_event", 0x02: "disable_command_received", 0x03: "runtime_error", 0x04: "cpu_reset_detected",
    0x05: "parameters_recovered_after_reset", 0x06: "restart_failed_rtc_invalid",
    0x07: "restart_failed_logger_status_invalid", 0x08: "restart_failed_schedule_not_recovered",
    0x09: "unable_to_load_alarm_time", 0x0A: "sampling_restarted_after_rtc_reset",
    0x0B: "parameters_recovered_sampling_restarted_after_rtc_reset", 0x0C: "deployment_end_time_reached",
    0x0F: "power_source_switched_to_usb", 0x16: "power_source_switched_to_internal_battery",
    0x17: "power_source_switched_to_external_battery", 0x1C: "regimes_enabled_not_yet_in_regime",
    0x1D: "entered_regime_1", 0x1E: "entered_regime_2", 0x1F: "entered_regime_3", 0x20: "end_of_regime_bin",
    0x24: "battery_failed_schedule_finished", 0x2D: "regimes_passed_final_boundary",
}
# deployment status (3.4.5) in the L2 vocabulary the rest of rbr-tpw uses
STATUS_L2 = {"sampling": "logging", "gated": "pending", "paused": "paused", "inactive": "stopped"}


class UsbHostStorage(LinkError):
    """3.5.3: with `storage access=usbhost` the logger refuses downloads through the command interface."""


# --------------------------------------------------------------------------- replies


def parse_reply(line: str) -> tuple[list[str], dict[str, str]]:
    """Split a Gen4 reply into its echoed command words and its key=value pairs (1.1.5.1).

    Keys are matched case-insensitively and in any order, unknown keys are kept. The id command keeps the
    older "key = value, key = value" form (3.3.1); both forms are accepted.
    """
    text = _PROMPT.sub("", line).strip()
    text = re.sub(r"\s*=\s*", "=", text).replace(",", " ")
    words: list[str] = []
    pairs: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            pairs[k.lower()] = v
        elif not pairs:
            words.append(tok)
    return words, pairs


def _query(link: Link, cmd: str, raw: dict | None = None, timeout: float = 3.0) -> dict[str, str]:
    """Send a read-only command; raise LoggerError on ERR-nnn (7), note WRN-nnn and carry on."""
    line = _PROMPT.sub("", link.command(cmd, timeout)).strip()
    if raw is not None:
        raw[cmd] = line
    m = _ERRWRN.match(line)
    if m:
        if m.group(1).upper() == "ERR":
            raise LoggerError(cmd, m.group(2), m.group(3))
        link.note(f"warning from {cmd!r}: {line}")
    return parse_reply(line)[1]


def _try(link: Link, cmd: str, raw: dict) -> dict[str, str]:
    """_query() for optional information: an ERR reply is recorded in `raw` and gives {}."""
    try:
        return _query(link, cmd, raw)
    except LoggerError as err:
        raw[cmd] = f"ERR-{err.code} {err.text}".strip()
        return {}


def _labels(value: str | None) -> list[str]:
    """'a|b|c' -> [a, b, c]; 'none' or missing -> [] (3.5.1, 3.6.1, 3.6.4, 3.6.5)."""
    if not value or value.lower() == "none":
        return []
    return [x for x in value.split("|") if x]


def _float(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- queries


def clock_now(link: Link) -> str:
    """The logger clock as 'YYYYMMDDhhmmss', whole seconds (3.4.1 `clock datetime`).

    `clock` also reports offsetfromutc (hours, "+0.00" by default, "unknown" if never set); it is only a
    record of the local time zone and does not change `datetime` (3.4.1)."""
    return _query(link, "clock datetime")["datetime"]


def snapshot(link: Link) -> dict:
    """Configuration, state, memory and power as reported by the logger (read-only).

    Normalized keys as for the L2 loggers, plus Gen4 detail:
      status        deployment status (3.4.5): sampling | gated | paused | inactive
      status_l2     the same in the L2 vocabulary (logging | pending | paused | stopped)
      sampling      mode and period of the latest dataset's first schedule (regimes: period1)
      channel_list  every instrument channel in `channel list` order (3.6.1); status is always 0 (Gen4 has
                    no channel status; which channels are stored is decided per schedule, see decode())
      datasets      [{label, status, schedulelist, bytecount, datatype}] oldest first (3.5.1)
      schedules     live definitions of the latest dataset's schedules, with their stored channel order from
                    `schedule grouplist` + `group channellist` (4.2.2). Only guaranteed to match the data while
                    that dataset is open: schedules may be edited after a deployment (3.5.1). decode() uses the
                    dataset's own metadata instead.
    """
    raw: dict[str, str] = {}
    instrument = _query(link, "instrument", raw)
    clock = _try(link, "clock", raw)
    deployment = _try(link, "deployment", raw)
    storage = _query(link, "storage", raw)
    ds = _query(link, "dataset", raw)
    datasets = []
    for label in _labels(ds.get("list")):
        info = _try(link, f"dataset {label}", raw)
        datasets.append({"label": label, "status": info.get("status", ""),
                         "schedulelist": _labels(info.get("schedulelist")),
                         "bytecount": int(info.get("bytecount", 0) or 0), "datatype": info.get("datatype", "")})
    schedules = {}
    if datasets:
        latest = datasets[-1]["label"]
        for sch in datasets[-1]["schedulelist"]:
            s = _try(link, f"schedule {sch}", raw)
            data = _try(link, f"dataset {latest}/{sch}/data", raw)
            order = []
            for grp in _labels(s.get("grouplist")):
                order += _labels(_try(link, f"group {grp}", raw).get("channellist"))
            schedules[sch] = {"mode": s.get("mode", ""), "period": s.get("period") or s.get("period1", ""),
                              "grouplist": _labels(s.get("grouplist")), "channels": order,
                              "bytecount": int(data.get("bytecount", 0) or 0),
                              "samplecount": int(data.get("samplecount", 0) or 0), "detail": s}
    channel_list = []
    for i, label in enumerate(_labels(_query(link, "channel", raw).get("list")), 1):
        ch = _try(link, f"channel {label}", raw)
        cal = _try(link, f"calibration {label}", raw)
        coeffs = {k: _float(v, v) for k, v in cal.items() if re.fullmatch(r"[cxn]\d+", k)}
        channel_list.append({
            "index": i, "type": ch.get("type", ""), "label": label, "equation": cal.get("equation", ""),
            "status": 0, "userunits": ch.get("userunits", ""), "derived": ch.get("derived", "off") == "on",
            "calibration_datetime": cal.get("datetime", ""), "coefficients": coeffs,
            "user_offset": _float(cal.get("offset"), 0.0), "user_slope": _float(cal.get("slope"), 1.0),
            "sensor": ch.get("sensor", "")})
    first = next(iter(schedules.values()), {})
    power = _power(link, raw)
    status = deployment.get("status", "")
    return {
        "now": clock.get("datetime", ""), "offsetfromutc": clock.get("offsetfromutc", ""),
        "status": status, "status_l2": STATUS_L2.get(status, status),
        "instrument": instrument, "deployment": deployment,
        "sampling": {"mode": first.get("mode", ""), "period": str(first.get("period", ""))},
        "channel_list": channel_list,
        "datasets": datasets, "schedules": schedules,
        "meminfo": {"used": int(storage.get("used", 0) or 0), "remaining": int(storage.get("remaining", 0) or 0),
                    "size": int(storage.get("size", 0) or 0)},
        "storage_access": storage.get("access", ""),
        "datatype": instrument.get("datatype", ""),
        "parameters": _try(link, "parameters", raw),
        "power": power,
        "raw": raw,
    }


def _power(link: Link, raw: dict) -> dict:
    """3.3.3.3: `instrument power source`, `... internal`, `... external` (voltage V; capacity, used J)."""
    source = _try(link, "instrument power source", raw).get("source", "")
    internal = _try(link, "instrument power internal", raw)
    external = _try(link, "instrument power external", raw)
    cap, used = _float(internal.get("capacity")), _float(internal.get("used"))
    remaining = cap - used if cap > 0 else math.nan
    return {"source": source, "battery_voltage_V": _float(internal.get("voltage")),
            "external_voltage_V": _float(external.get("voltage")),
            "battery_type": internal.get("batterytype", ""), "energy_capacity_J": cap, "energy_used_J": used,
            "energy_remaining_J": remaining,
            "energy_remaining_fraction": remaining / cap if cap > 0 else math.nan,
            "internal": internal, "external": external,
            "int_raw": raw.get("instrument power internal", ""), "ext_raw": raw.get("instrument power external", "")}


# --------------------------------------------------------------------------- download


def _crc_mcrf4xx(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16 with the CCITT polynomial fed LSB first (reflected, 0x8408), seed 0xFFFF, no final XOR."""
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def crc_variant(data_with_crc: bytes) -> str | None:
    """Which reading of the 3.5.2 CRC checks out, or None.

    3.5.2: "CCITT polynomial, feeding bytes into the generator LSB first ... 0xFFFF seed ... bytes swapped
    before appending ... including them gives zero". The Gen3 L3 reference uses the same words, and on real
    L2 loggers (fwtype 9, 2026-09-25) that transfer CRC is CRC-16/CCITT-FALSE appended big-endian
    (crc.check_appended). Both that and the literal reflected reading are accepted; the transcript records
    which one matched.
    """
    if len(data_with_crc) < 2:
        return None
    if crc16_ccitt(data_with_crc) == 0:
        return "ccitt-false, big-endian (as L2/Gen3)"
    if _crc_mcrf4xx(data_with_crc) == 0:
        return "reflected ccitt, little-endian"
    return None


def _download_once(link: Link, obj: str, count: int, start: int, timeout: float) -> tuple[bytes, str]:
    """One `download <obj> bytecount=<count> bytestart=<start>` transfer (3.5.2). Returns (data, crc variant)."""
    leftover = _PROMPT_B.sub(b"", link._buf)
    if leftover.strip():
        link._record("DROP", _show(leftover))
    link._buf = b""
    cmd = f"download {obj} bytecount={count} bytestart={start}"  # all parameters every time: no tracking (ERR-302)
    link._send(cmd)
    deadline = time.monotonic() + timeout
    need = None
    while True:
        if need is None:
            m = _DL_HDR.search(link._buf)
            if m:
                before = _PROMPT_B.sub(b"", link._buf[: m.start()])
                if before.strip():
                    link._record("DROP", _show(before))
                header = m.group(0).decode("ascii", errors="replace").strip()
                link._record("RX", header)
                pairs = parse_reply(header)[1]
                if m.group(1).decode("ascii", errors="replace").lower() != obj.lower():
                    raise LinkError(f"reply is for {m.group(1)!r}, not {obj!r}: {header!r}")
                n, s = int(pairs.get("bytecount", -1)), int(pairs.get("bytestart", start))
                if s != start or not 0 <= n <= count:
                    raise LinkError(f"unexpected reply header {header!r} to {cmd!r}")
                link._buf = link._buf[m.end():]
                need = n + 2
            else:
                e = _DL_ERR.search(link._buf)
                if e:
                    text = e.group(1).decode("ascii", errors="replace").strip()
                    link._record("RX", text)
                    em = _ERRWRN.match(text)
                    raise LoggerError(cmd, em.group(2), em.group(3))
        if need is not None and len(link._buf) >= need:
            block, link._buf = link._buf[:need], link._buf[need:]
            variant = crc_variant(block)
            link._record("RX", f"<{need - 2} data bytes + CRC {block[-2:].hex()} {'OK' if variant else 'BAD'}>")
            if variant is None:
                raise LinkError(f"CRC mismatch on {cmd!r}")
            return block[:-2], variant
        if time.monotonic() > deadline:
            link.note(f"timeout after {timeout:g} s on {cmd!r}; have {len(link._buf)} of {need} bytes")
            raise LinkError(f"timeout on {cmd!r}: have {len(link._buf)} of {need} bytes")
        want = 65536 if need is None else max(1, need - len(link._buf))
        link._buf += link.ser.read(min(want, 65536))


def download_block(link: Link, obj: str, count: int, start: int, timeout: float = 10.0, retries: int = 5) -> bytes:
    """A CRC-checked block, retried on CRC failure or timeout (not on a logger error)."""
    last = None
    for attempt in range(1, retries + 1):
        try:
            data, variant = _download_once(link, obj, count, start, timeout)
            if getattr(link, "_gen4_crc_variant", None) != variant:
                link.note(f"download CRC matched as {variant}")
                link._gen4_crc_variant = variant
            return data
        except LoggerError:
            raise
        except LinkError as err:
            last = err
            link.note(f"download {obj} {start}+{count}: attempt {attempt} failed: {err}")
            link._drain(0.3)
    raise LinkError(f"download {obj} bytestart={start} failed after {retries} attempts: {last}")


def objects(link: Link) -> list[tuple[str, int]]:
    """Every storage object on the logger with its size in bytes: meta, events and each schedule's data."""
    out = []
    for ds in _labels(_query(link, "dataset").get("list")):
        schedules = _labels(_query(link, f"dataset {ds}").get("schedulelist"))
        for obj in [f"{ds}/meta", f"{ds}/events", *(f"{ds}/{s}/data" for s in schedules)]:
            out.append((obj, int(_query(link, f"dataset {obj}").get("bytecount", 0) or 0)))
    return out


def _part_path(part_dir: Path, sn: str, obj: str) -> Path:
    return Path(part_dir) / f"{sn}__{re.sub(r'[^A-Za-z0-9._-]', '_', obj.replace('/', '__'))}.part"


def download(link: Link, part_dir: Path, sn: str, progress=None) -> dict[str, bytes]:
    """Every object of every dataset, CRC-checked, keyed '<dataset>/meta', '<dataset>/events',
    '<dataset>/<schedule>/data'. Each object is appended to its own partial file under `part_dir`; a partial
    file is resumed only if its first bytes match the logger's (same deployment). Partial files are left in
    place for the caller to remove once the result is saved.

    An open dataset keeps growing while it downloads; its objects are read up to the size reported when the
    download started.
    """
    st = _query(link, "storage")
    if st.get("access", "").lower() == "usbhost":
        raise UsbHostStorage("this logger's storage is set to access=usbhost: it does not allow downloads "
                             "through commands. Copy its dataset files off it as a USB drive instead "
                             "(L3.5 ref 3.5.3).")
    todo = objects(link)
    total = sum(n for _, n in todo) or 1
    done = moved = 0
    t0 = time.monotonic()
    out: dict[str, bytes] = {}
    Path(part_dir).mkdir(parents=True, exist_ok=True)
    for obj, size in todo:
        part = _part_path(part_dir, sn, obj)
        head = download_block(link, obj, min(HEAD, size), 0) if size else b""
        if part.exists():
            with open(part, "rb") as f:
                old = f.read(len(head))
            if old != head or part.stat().st_size > size:
                link.note(f"partial file {part.name} does not match {obj}; restarting it")
                part.unlink()
            else:
                link.note(f"resuming {obj} from {part.stat().st_size} bytes in {part.name}")
        with open(part, "ab") as f:
            offset = f.tell()
            if offset == 0 and head:
                f.write(head)
                offset = len(head)
            while offset < size:
                n = min(CHUNK, size - offset)
                block = download_block(link, obj, n, offset)
                if not block:
                    link.note(f"{obj}: logger returned no bytes at {offset} of {size}; stopping there")
                    break
                f.write(block)
                f.flush()
                os.fsync(f.fileno())
                offset += len(block)
                moved += len(block)
                if progress:
                    progress(done + offset, total, moved / max(time.monotonic() - t0, 1e-6))
        data = part.read_bytes()
        if len(data) != size:
            link.note(f"{obj}: have {len(data)} bytes, logger reported {size}")
        out[obj] = data
        done += size
    return out


# --------------------------------------------------------------------------- metadata (4.3)


def _cstr(b: bytes) -> str:
    """NUL-terminated, 0xFF-padded string (4.3.x)."""
    return b.split(b"\x00", 1)[0].replace(b"\xff", b"").decode("ascii", errors="replace")


def _section_crc_ok(sec: bytes) -> bool:
    """4.3: "A standard 16b CRC of all Section N" — not further specified. CCITT-FALSE over the section
    without its last two bytes, stored either byte order, or the reflected reading, is accepted."""
    body, stored = sec[:-2], sec[-2:]
    c = crc16_ccitt(body)
    return stored in (struct.pack(">H", c), struct.pack("<H", c)) or crc_variant(sec) is not None


@dataclass
class Meta:
    """The parts of a dataset's metadata (4.3) needed to decode its sample data."""

    data_format: int = 0
    enable_time_ms: int = 0
    start_time_ms: int = 0
    end_time_ms: int = 0
    utc_offset_ms: int | None = None
    dataset_label: str = ""
    configuration_label: str = ""
    fw_type: int = 0
    fw_version: str = ""
    serial_number: int = 0
    model: str = ""
    schedules: list[dict] = field(default_factory=list)  # {index, label, mode, flags, groups, period, ...}
    groups: dict[int, dict] = field(default_factory=dict)  # index -> {label, channels: [(channel index, flags)]}
    channels: dict[int, dict] = field(default_factory=dict)  # index -> channel details
    warnings: list[str] = field(default_factory=list)


def _sections(meta: bytes, warnings: list[str]) -> dict[int, bytes]:
    """{section ID: bytes} via the map in Section 1 (4.3.2), falling back to walking the sections in order."""
    if meta[:4] != b"RBR\x00":  # 4.3.1 TAG 0x00524252
        raise ValueError(f"not Gen4 metadata: starts {meta[:4]!r}, expected b'RBR\\x00' (L3.5 ref 4.3.1)")
    found: dict[int, bytes] = {}
    try:
        sid, size, version, total = struct.unpack_from("<IHII", meta, 4)
        if sid != 0x01000000:
            raise ValueError(f"section 1 has ID 0x{sid:08X}")
        pos = 4 + 14 + (4 if version >= 0x00020000 else 0)  # hash code from metadata version 00.02.0000
        while pos + 10 <= 4 + size - 2:
            s_id, off, s_size = struct.unpack_from("<IIH", meta, pos)
            found[s_id] = meta[off : off + s_size]
            pos += 10
    except (struct.error, ValueError) as err:
        warnings.append(f"metadata map unreadable ({err}); walking sections in order")
    if len(found) <= 1:
        found, pos = {}, 4
        while pos + 6 <= len(meta):
            s_id, s_size = struct.unpack_from("<IH", meta, pos)
            if s_size < 8 or pos + s_size > len(meta):
                break
            found[s_id] = meta[pos : pos + s_size]
            pos += s_size
    for s_id, sec in found.items():
        if len(sec) < 8 or not _section_crc_ok(sec):
            warnings.append(f"metadata section 0x{s_id:08X}: CRC does not check")
    return found


def parse_meta(meta: bytes) -> Meta:
    """Decode the metadata object of a dataset (4.3): deployment, configuration, schedules, groups, channels."""
    m = Meta()
    secs = _sections(meta, m.warnings)
    if (s := secs.get(0x02000000)) is not None:  # 4.3.3 Logger
        m.fw_type = struct.unpack_from("<I", s, 6)[0]
        m.fw_version = _cstr(s[10:46])
        m.serial_number = struct.unpack_from("<I", s, 46)[0]
        m.model = _cstr(s[50:66])
    if (s := secs.get(0x04000000)) is not None:  # 4.3.5 Deployment
        m.data_format = struct.unpack_from("<I", s, 6)[0]
        m.enable_time_ms, m.start_time_ms, m.end_time_ms = struct.unpack_from("<QQQ", s, 16)
        utc = struct.unpack_from("<i", s, 40)[0]
        m.utc_offset_ms = None if utc == -2147483648 else utc
    if (s := secs.get(0x05000000)) is not None:  # 4.3.6 Configuration
        m.dataset_label, m.configuration_label = _cstr(s[6:38]), _cstr(s[38:70])
    for s_id, s in sorted(secs.items()):
        top = s_id >> 16
        if top == 0x0602:  # 4.3.7.2 schedule details
            index, = struct.unpack_from("<H", s, 6)
            label = _cstr(s[8:40])
            mode, flags, ngroups = struct.unpack_from("<BIB", s, 40)
            groups = [g for g in s[46 : 46 + 16][:ngroups] if g != 0xFF]
            params = s[46 + 16 + 1 + 16 : -2]  # after FE-group count (1) and FE-groups (16)
            sch = {"index": index, "label": label, "mode": MODES.get(mode, f"code {mode}"), "mode_code": mode,
                   "flags": flags, "stored": bool(flags & 1), "groups": groups}
            if mode == 0 and len(params) >= 5:  # 4.3.7.4.1 continuous
                sch["period"], cast = struct.unpack_from("<IB", params, 0)
                sch["castdetection"] = bool(cast)
            elif mode in (1, 2, 3, 4) and len(params) >= 12:  # 4.3.7.4.2
                sch["measurementperiod"], sch["period"], sch["measurementcount"] = struct.unpack_from("<III", params)
            elif mode == 5 and len(params) >= 18:  # 4.3.7.4.3
                d, c, sch["fastperiod"], sch["period"], ft, st = struct.unpack_from("<BBIIff", params)
                sch.update(direction="ascending" if d else "descending", castdetection=bool(c),
                           fastthreshold=ft, slowthreshold=st)
            elif mode == 6 and len(params) >= 31:  # 4.3.7.4.4
                d, count, ref, final = struct.unpack_from("<BBBH", params)
                regimes = [struct.unpack_from("<HHI", params, 5 + 8 * i) for i in range(3)]
                sch.update(direction="ascending" if d else "descending", count=count, reference=ref,
                           finalboundary=final, regimes=regimes, period=regimes[0][2])
            m.schedules.append(sch)
        elif top == 0x0702:  # 4.3.8.2 user group details
            index, = struct.unpack_from("<H", s, 6)
            label = _cstr(s[8:40])
            count, = struct.unpack_from("<H", s, 48)
            pairs = [(s[50 + 2 * i], s[51 + 2 * i]) for i in range(min(count, 32))]
            m.groups[index] = {"label": label, "channels": [(i, f) for i, f in pairs if i != 0xFF]}
        elif top == 0x0902:  # 4.3.10.2 channel details
            m.channels.update(_channel_details(s))
    m.schedules.sort(key=lambda x: x["index"])
    return m


def _channel_details(s: bytes) -> dict[int, dict]:
    index, address = struct.unpack_from("<HH", s, 6)
    type_key, label = _cstr(s[10:26]), _cstr(s[26:58])
    fw_size, = struct.unpack_from("<H", s, 58)
    # 4.3.10.2: fw_info_size "includes terminating NUL (may be NUL-padded to a multiple of four bytes)" - it is not
    # said whether the size counts the padding. Use the reading whose coefficient count fits the section size.
    fw_len = fw_size
    for cand in dict.fromkeys((fw_size, (fw_size + 3) // 4 * 4)):
        q = 60 + cand
        if q + 28 + 84 > len(s) - 2:
            continue
        spec_count_c, = struct.unpack_from("<H", s, q + 24)
        cc, = struct.unpack_from("<I", s, q + 28 + 80)
        parts = ((cc >> 8) & 0xFF) + ((cc >> 16) & 0xFF) + ((cc >> 24) & 0xFF)
        end = q + 28 + 84 + 4 * (cc & 0xFF)
        if parts == (cc & 0xFF) and end <= len(s) - 2 and (spec_count_c or end == len(s) - 2):
            fw_len = cand
            break
    fw_info = _cstr(s[60 : 60 + fw_len])
    p = 60 + fw_len
    user_groups, fe_groups, flags, settling, read, guard, spec_count, spec_off = struct.unpack_from("<IIIIIIHH", s, p)
    p += 28
    equation = _cstr(s[p : p + 32])
    cal_ms, user_offset, user_slope = struct.unpack_from("<Qff", s, p + 32)
    factory_units, user_units = _cstr(s[p + 48 : p + 64]), _cstr(s[p + 64 : p + 80])
    ccount, = struct.unpack_from("<I", s, p + 80)
    p += 84
    n_total, n_c, n_x, n_n = ccount & 0xFF, (ccount >> 8) & 0xFF, (ccount >> 16) & 0xFF, (ccount >> 24) & 0xFF
    coeffs: dict[str, float | int] = {}
    for i in range(n_c):
        coeffs[f"c{i}"] = struct.unpack_from("<f", s, p + 4 * i)[0]
    for i in range(n_x):
        coeffs[f"x{i}"] = struct.unpack_from("<f", s, p + 4 * (n_c + i))[0]
    for i in range(n_n):
        coeffs[f"n{i}"] = struct.unpack_from("<i", s, p + 4 * (n_c + n_x + i))[0]  # "signed 32-bit integers"
    return {index: {"index": index, "module_address": address, "type": type_key, "label": label,
                    "fw_info": fw_info, "user_groups": user_groups, "flags": flags,
                    "derived": bool(flags & (1 << 3)), "settlingtime": settling, "readtime": read,
                    "guardtime": guard, "equation": equation, "calibration_ms": cal_ms,
                    "user_offset": user_offset, "user_slope": user_slope, "factory_units": factory_units,
                    "userunits": user_units or factory_units, "coefficients": coeffs,
                    "coefficient_count": n_total}}


# --------------------------------------------------------------------------- decode


def _pick(datasets: dict[str, bytes], snap: dict | None, dataset: str | None) -> str:
    if dataset:
        return dataset
    labels = [d["label"] for d in (snap or {}).get("datasets", []) if f"{d['label']}/meta" in datasets]
    if labels:
        return labels[-1]
    metas = [k[: -len("/meta")] for k in datasets if k.endswith("/meta")]
    if not metas:
        raise ValueError("no <dataset>/meta object downloaded")
    return metas[-1]


def columns(datasets: dict[str, bytes], snap: dict | None = None, dataset: str | None = None,
            schedule: str | None = None) -> tuple[str, str, list[dict], Meta, str]:
    """(dataset, schedule, details of each stored column in order, metadata, value format) for decode().

    Column order: the schedule's user groups in order, then each group's channels in order (4.2.2), from
    the dataset's own metadata (4.3.7.2, 4.3.8.2), not the logger's current definitions. 4.2.2 does not say
    whether a group's channels flagged "not stored" (4.3.8.2.1 b0) take a column; the reading that makes the
    data a whole number of samples wins (and, for the open dataset, matches `bytecount / samplecount`).
    """
    ds = _pick(datasets, snap, dataset)
    meta = parse_meta(datasets[f"{ds}/meta"])
    schs = [s for s in meta.schedules if f"{ds}/{s['label']}/data" in datasets]
    if schedule:
        schs = [s for s in schs if s["label"] == schedule]
    if not schs:
        raise ValueError(f"dataset {ds!r}: no schedule with downloaded data (have {sorted(datasets)})")
    sch = schs[0]
    cols = []
    for g in sch["groups"]:
        for idx, flags in meta.groups.get(g, {}).get("channels", []):
            ch = dict(meta.channels.get(idx, {"index": idx, "label": f"channel_{idx:02d}"}))
            ch.update(group=meta.groups[g]["label"], group_flags=flags, stored=bool(flags & 1),
                      hidden=bool(flags & 0b1100))
            cols.append(ch)

    fmt = DATA_FORMATS.get(meta.data_format)
    if fmt is None:  # fall back to what the logger reported (3.5.1 dataset datatype, 3.3.3 instrument datatype)
        dtype = next((d["datatype"] for d in (snap or {}).get("datasets", []) if d["label"] == ds), "") or \
            (snap or {}).get("datatype", "float32")
        fmt = {"float32": DATA_FORMATS[1], "float64": DATA_FORMATS[2], "calfloat64": DATA_FORMATS[3]}.get(
            dtype, DATA_FORMATS[1])
        meta.warnings.append(f"data format not in the metadata; using {fmt[0]} from the logger")
    width = np.dtype(fmt[1]).itemsize
    size = len(datasets[f"{ds}/{sch['label']}/data"])
    s = (snap or {}).get("schedules", {}).get(sch["label"], {})
    per_sample = s["bytecount"] / s["samplecount"] if s.get("samplecount") and ds == _pick(datasets, snap, None) \
        else None
    stored = [c for c in cols if c["stored"]]
    for cand in ([stored, cols] if len(stored) != len(cols) else [cols]):
        rec = 8 + width * len(cand)
        if (per_sample is None and size % rec == 0) or per_sample == rec:
            return ds, sch["label"], cand, meta, fmt[1]
    raise ValueError(f"{ds}/{sch['label']}: {size} bytes is not a whole number of samples for {len(stored)} "
                     f"stored or {len(cols)} listed channels of {fmt[0]}")


def _error_codes(values: np.ndarray, width: int) -> np.ndarray:
    """uint32 error word per value: 0 for a number, the float32 bit pattern for NaN/inf (4.2.4).

    float64 NaNs are mapped to float32 by the 29-bit truncation of 4.2.4 (e.g. error 14 = 0xFFC0000E)."""
    bad = ~np.isfinite(values)
    if width == 4:
        bits = values.view("<u4").astype(np.uint32)
    else:
        b64 = values.view("<u8")
        bits = ((((b64 >> 63) & 1) << 31) | 0x7F800000 | ((b64 >> 29) & 0x7FFFFF)).astype(np.uint32)
    return np.where(bad, bits, np.uint32(0)).astype(np.uint32)


def decode(datasets: dict[str, bytes], snap: dict | None = None, dataset: str | None = None,
           schedule: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """(time_ms int64 (n,), values float64 (n, nchan), error words uint32 (n, nchan), events).

    Defaults to the latest dataset and its first schedule; columns() names the columns. Each stored sample
    is a uint64 ms timestamp (Unix epoch, no leap seconds) then one float32 or float64 per channel,
    little-endian (4.2.2). Errors are negative NaNs with the error number in the payload (4.2.4); they stay
    NaN in `values` and their float32 bit pattern goes in the error words. Events (4.4) are those whose
    schedule mask includes this schedule, or that name no schedule: (ms, type, 8-byte auxiliary as a
    little-endian integer).
    """
    ds, sch_label, cols, meta, fmt = columns(datasets, snap, dataset, schedule)
    width = np.dtype(fmt).itemsize
    n = len(cols)
    data = datasets[f"{ds}/{sch_label}/data"]
    rec = np.dtype([("t", "<u8"), ("v", fmt, (n,))])
    arr = np.frombuffer(data[: len(data) - len(data) % rec.itemsize], dtype=rec)
    raw_values = np.ascontiguousarray(arr["v"]).reshape(len(arr), n)
    errors = _error_codes(raw_values, width)
    time_ms = arr["t"].astype(np.int64)
    values = raw_values.astype(np.float64)
    bit = 1 << (next(x["index"] for x in meta.schedules if x["label"] == sch_label) - 1)
    events = []
    ev = datasets.get(f"{ds}/events", b"")
    pos = 0
    while pos + 24 <= len(ev):
        t, mask, size, etype, aux = struct.unpack_from("<QIHHQ", ev, pos)
        if mask == 0 or mask & bit:
            events.append((int(t), int(etype), int(aux)))
        pos += size if size >= 24 else 24
    return time_ms, values, errors, events


def event_name(code: int) -> str:
    return EVENT_NAMES.get(code, f"type_0x{code:02X}")
