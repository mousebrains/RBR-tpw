"""Gen4 (L3.5, fwtype 120) driver against a simulated logger built from the L3.5 Command Reference rev B.

No Gen4 logger or data has been available: these tests check consistency with the documentation (reply
formats copied from its examples, metadata laid out from the tables in section 4.3), not with hardware.
"""

import struct
import threading
import time

import numpy as np
import pytest

from rbr_tpw import gen4
from rbr_tpw import link as link_module
from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.link import Link, LinkError, LoggerError

ENABLE_MS = 1_727_740_800_000  # 2024-10-01T00:00:00Z
CHANNELS = [  # label, type, address, equation, units, coefficients (c, x, n), stored
    ("conductivity_00", "cond19", 32, "corr_cond3", "mS/cm", ([0.0300, 158.07, 1.0], [6.4e-4, -2.1e-5], [2, 3]), True),
    ("temperature_00", "temp14", 1, "tmp", "C", ([3.388e-3, -2.573e-4, 2.451e-6, -8.459e-8], [], []), True),
    ("pressure_00", "pres24", 2, "corr_pres2", "dbar", ([-32.83, 2368.2, 34.67, -31.74], [10.13, 0.0382], [2]), True),
    ("temperature_01", "temp22", 33, "tmp", "C", ([3.12e-3, -2.80e-4, 4.02e-6, -2.52e-7], [], []), False),
]


def _section(sid: int, body: bytes, crc: str = "be") -> bytes:
    head = struct.pack("<IH", sid, 4 + 2 + len(body) + 2) + body
    c = crc16_ccitt(head)
    return head + (struct.pack(">H", c) if crc == "be" else struct.pack("<H", c))


def _str(s: str, n: int, pad: bytes = b"\xff") -> bytes:
    b = s.encode() + b"\x00"
    return b + pad * (n - len(b))


def build_meta(data_format: int = 1, dataset="dataset_01", schedule="sch_ctd", group="gr_ctd", fw_pad=True,
               order=None) -> bytes:
    """Metadata laid out from L3.5 ref 4.3 (TAG, sections 1, 2, 4, 5, 6.1, 6.2.1, 7.1, 7.2.1, 9.1, 9.2.x).
    `order`: the group's channels as 0-based CHANNELS indices (default: CHANNELS order)."""
    secs = []
    pn = b"L3-M13-F15-BEC12-INT12-SCT16-SP11"
    secs.append(_section(0x02000000, struct.pack("<I", 120) + _str("2.1.0", 36, b"\x00") + struct.pack("<I", 210000)
                         + _str("RBRconcerto3", 16) + struct.pack("<IH", 0, 8) + _str("", 16)
                         + struct.pack("<I", 38400) + struct.pack("<H", len(pn) + 1) + pn
                         + struct.pack("<H", 1) + b""))
    deploy = struct.pack("<IIH", data_format, 0b100010, 2) + struct.pack("<QQQ", ENABLE_MS, ENABLE_MS,
                                                                            4102444799000)
    deploy += struct.pack("<iIBBII", 3_600_000, 600000, 0, 0, 0, 0) + struct.pack("<f", 10.1325)
    deploy += struct.pack("<HH", 6, 100) + struct.pack("<ffff", 138000.0, 0.0, 100100.0, 0.0)
    deploy += struct.pack("<ffffffff", 0.0191, 15.0, 10.1325, 10.1325, 1.0281, 35.0, 1550.744, 0.0)
    secs.append(_section(0x04000000, deploy))
    secs.append(_section(0x05000000, _str(dataset, 32) + _str("default_config", 32)))
    secs.append(_section(0x06010000, struct.pack("<H", 1) + struct.pack("<H", 1) + _str(schedule, 32)
                         + struct.pack("<H", 0)))
    sched = struct.pack("<H", 1) + _str(schedule, 32) + struct.pack("<BIB", 0, 1, 1) + bytes([1] + [0xFF] * 15)
    sched += struct.pack("<B", 0) + b"\xff" * 16 + struct.pack("<IB", 1000, 0)
    secs.append(_section(0x06020100, sched))
    secs.append(_section(0x07010000, struct.pack("<HH", 1, 1) + _str(group, 32) + struct.pack("<H", 0)))
    pairs = b"".join(bytes([i + 1, 0b011 if CHANNELS[i][-1] else 0b1100])
                     for i in (range(len(CHANNELS)) if order is None else order))
    secs.append(_section(0x07020100, struct.pack("<H", 1) + _str(group, 32) + struct.pack("<IIH", 1, 0, len(CHANNELS))
                         + pairs + b"\xff" * (64 - len(pairs))))
    cmap = struct.pack("<H", len(CHANNELS)) + b"".join(
        struct.pack("<H", i + 1) + _str(c[0], 32) + struct.pack("<H", 0) for i, c in enumerate(CHANNELS))
    secs.append(_section(0x09010000, cmap))
    for i, (label, ctype, addr, eq, units, (c, x, n), _) in enumerate(CHANNELS):
        fw = b"T = FE-cond3-cond, F = 7.041, r = 5373, H = 1, A = 32\x00"
        fw_field = fw + (b"\x00" * (-len(fw) % 4) if fw_pad else b"")
        fw_size = len(fw_field) if fw_pad else len(fw)
        body = struct.pack("<HH", i + 1, addr) + _str(ctype, 16) + _str(label, 32) + struct.pack("<H", fw_size)
        body += fw_field + struct.pack("<IIIIIIHH", 1, 0, 1 << 10 if ctype.startswith("temp") else 0, 60, 290, 20, 0, 0)
        body += _str(eq, 32, b"\x00") + struct.pack("<Qff", ENABLE_MS - 86_400_000, 0.0, 1.0)
        body += _str(units, 16) + _str(units, 16)
        body += struct.pack("<I", (len(c) + len(x) + len(n)) | len(c) << 8 | len(x) << 16 | len(n) << 24)
        body += struct.pack(f"<{len(c)}f", *c) + struct.pack(f"<{len(x)}f", *x) + struct.pack(f"<{len(n)}i", *n)
        secs.append(_section(0x09020000 | (i + 1) << 8, body))
    ids = [struct.unpack_from("<I", s)[0] for s in secs]
    s1_size = 4 + 2 + 4 + 4 + 4 + 10 * (len(secs) + 1) + 2
    offsets, off = [], 4 + s1_size
    for s in secs:
        offsets.append(off)
        off += len(s)
    rows = struct.pack("<IIH", 0x01000000, 4, s1_size) + b"".join(
        struct.pack("<IIH", sid, o, len(s)) for sid, o, s in zip(ids, offsets, secs, strict=True))
    s1 = _section(0x01000000, struct.pack("<III", 0x00020000, off, 0x1234ABCD) + rows)
    assert len(s1) == s1_size
    return b"RBR\x00" + s1 + b"".join(secs)


def build_data(n: int, data_format: int = 1) -> tuple[bytes, np.ndarray, np.ndarray]:
    """n samples of the 3 stored channels (4.2.2); sample 10, channel 2 carries error 14 (4.2.4)."""
    t = ENABLE_MS + 1000 * np.arange(n, dtype=np.int64)
    v = np.column_stack([np.linspace(40, 42, n), np.linspace(15, 16, n), np.linspace(10, 20, n)])
    fmt = "<f4" if data_format == 1 else "<f8"
    rec = np.zeros(n, np.dtype([("t", "<u8"), ("v", fmt, (3,))]))
    rec["t"], rec["v"] = t, v
    raw = bytearray(rec.tobytes())
    width = 4 if data_format == 1 else 8
    pos = 10 * (8 + 3 * width) + 8 + 1 * width
    raw[pos : pos + width] = struct.pack("<I", 0xFFC0000E) if width == 4 else struct.pack("<Q", 0xFFF80001C0000000)
    return bytes(raw), t, v


def build_events() -> bytes:
    """Two 24-byte events (4.4.1): USB power (no schedule mask) and one for schedule 2 only."""
    return (struct.pack("<QIHHQ", ENABLE_MS + 5000, 0, 24, 0x0F, 0xFFFFFFFFFFFFFFFF)
            + struct.pack("<QIHHQ", ENABLE_MS + 6000, 0b10, 24, 0x1D, 0xFFFFFFFFFFFFFFFF))


class FakeGen4:
    """A simulated L3.5 logger behind a pyserial-like port; replies follow the reference's examples."""

    def __init__(self, n_samples=5000, crc="be", access="instrument", fail_after_blocks=None, data_format=1,
                 bad_crc=False):
        self.meta = build_meta(data_format)
        self.data, self.t, self.v = build_data(n_samples, data_format)
        self.events = build_events()
        self.objects = {"dataset_01/meta": self.meta, "dataset_01/events": self.events,
                        "dataset_01/sch_ctd/data": self.data}
        self.crc, self.access, self.fail_after_blocks, self.bad_crc = crc, access, fail_after_blocks, bad_crc
        self.data_format = data_format
        self.timeout = 0.02
        self.commands: list[str] = []
        self.downloaded = 0
        self._out = bytearray()
        self._in = b""
        self._lock = threading.Lock()

    @property
    def in_waiting(self):
        return len(self._out)

    def reset_input_buffer(self):
        self._out.clear()

    def close(self):
        pass

    def write(self, data):
        self._in += data
        while b"\r" in self._in:
            line, self._in = self._in.split(b"\r", 1)
            self._handle(line.decode("ascii"))
        return len(data)

    def read(self, n=1):
        deadline = time.time() + self.timeout
        while True:
            with self._lock:
                if self._out:
                    chunk = bytes(self._out[:n])
                    del self._out[:n]
                    return chunk
            if time.time() >= deadline:
                return b""
            time.sleep(0.001)

    def _reply(self, text):
        with self._lock:
            self._out += text.encode() + b"\r\nReady: "

    def _crc(self, data):
        if self.bad_crc:
            return b"\x00\x00"
        if self.crc == "be":
            return struct.pack(">H", crc16_ccitt(data))
        return struct.pack("<H", gen4._crc_mcrf4xx(data))

    def _handle(self, cmd):
        self.commands.append(cmd)
        used = sum(len(b) for b in self.objects.values())
        dt = "float32" if self.data_format == 1 else "float64"
        cal = {"conductivity_00": "equation=corr_cond3 datetime=20250528165608 offset=0.0000000e+000 "
                                  "slope=1.0000000e+000 c0=30.056130e-003 c1=158.07142e+000 c2=1.0000000e+000 "
                                  "x0=641.78620e-006 x1=-20.795860e-006 n0=temperature_00 n1=pressure_00",
               "temperature_00": "equation=tmp datetime=20250526165351 offset=0.0000000e+000 slope=1.0000000e+000 "
                                 "c0=3.3883424e-003 c1=-257.27355e-006 c2=2.4507071e-006 c3=-84.591889e-009"}
        replies = {
            "id": "id model = RBRconcerto3, serial = 210000, version = 2.1.0, fwtype = 120",
            "instrument": "instrument state=enabled sn=210000 model=RBRconcerto3 pn=L3-M13-F15-BEC12-INT12-SCT16-SP11 "
                          f"fwversion=2.1.0 fwtype=120 fwlock=off datatype={dt}",
            "clock": f"clock datetime={time.strftime('%Y%m%d%H%M%S', time.gmtime())} offsetfromutc=+1.00",
            "clock datetime": f"clock datetime={time.strftime('%Y%m%d%H%M%S', time.gmtime())}",
            "deployment": "deployment status=sampling gate=none simulation=off",
            "storage": f"storage used={used} remaining={134217728 - used} size=134217728 access={self.access}",
            "dataset": "dataset count=1 maxcount=20 list=dataset_01",
            "dataset dataset_01": f"dataset dataset_01 status=open schedulelist=sch_ctd bytecount={used} datatype={dt}",
            "dataset dataset_01/meta": f"dataset dataset_01/meta bytecount={len(self.meta)}",
            "dataset dataset_01/events": f"dataset dataset_01/events bytecount={len(self.events)} eventcount=2",
            "dataset dataset_01/sch_ctd/data": f"dataset dataset_01/sch_ctd/data bytecount={len(self.data)} "
                                               f"samplecount={len(self.t)}",
            "schedule sch_ctd": "schedule sch_ctd grouplist=gr_ctd configlist=default_config stream=off storage=on "
                                "mode=continuous period=1000 castdetection=off",
            "group gr_ctd": "group gr_ctd channellist=conductivity_00|temperature_00|pressure_00|temperature_01 "
                            "schedulelist=sch_ctd",
            "channel": "channel count=4 list=conductivity_00|temperature_00|pressure_00|temperature_01",
            "instrument power source": "instrument power source=usb",
            "instrument power internal": "instrument power internal voltage=6.52 batterytype=nimh "
                                         "capacity=138.000e+003 used=100.100e+003",
            "instrument power external": "instrument power external voltage=0.01 batterytype=none "
                                         "capacity=0.000e+000 used=0.000e+000",
            "parameters": "parameters altitude=0.0000 atmosphere=10.1325 density=1.0281 salinity=35.0000",
        }
        for label, ctype, addr, _eq, units, _c, _st in CHANNELS:
            replies[f"channel {label}"] = (f"channel {label} type={ctype} address={addr} settlingtime=60 "
                                           f"readtime=290 guardtime=20 userunits={units} derived=off "
                                           "grouplist=gr_ctd sensor=none")
            replies[f"calibration {label}"] = f"calibration {label} " + cal.get(
                label, "equation=tmp datetime=20250526165413 c0=3.3859643e-003")
        if cmd == "":
            with self._lock:
                self._out += b"Ready: "
            return
        if cmd in replies:
            self._reply(replies[cmd])
            return
        if cmd.startswith("download "):
            _, obj, *kv = cmd.split()
            args = dict(x.split("=") for x in kv)
            if self.fail_after_blocks is not None and obj.endswith("/data") and int(args["bytestart"]) >= 256:
                if self.fail_after_blocks <= 0:
                    self._reply("ERR-111 command failed")
                    return
                self.fail_after_blocks -= 1
            data = self.objects[obj]
            start, count = int(args["bytestart"]), int(args["bytecount"])
            block = data[start : start + count]
            self.downloaded += len(block)
            with self._lock:
                self._out += f"download {obj} bytecount={len(block)} bytestart={start}\r\n".encode()
                self._out += block + self._crc(block) + b"Ready: "
            return
        self._reply(f"ERR-102 invalid command '{cmd.split()[0]}'")


@pytest.fixture
def port(monkeypatch):
    holder = {}

    def open_port(name, baudrate=115200):
        return holder["fake"]

    monkeypatch.setattr(link_module, "open_serial", open_port)

    def connect(fake, tmp_path=None):
        holder["fake"] = fake
        link = Link("/dev/cu.gen4test")
        if tmp_path:
            link.start_transcript(tmp_path / "t.log")
        return link

    return connect


def test_reply_parsing_accepts_both_grammars():
    assert gen4.parse_reply("id model = RBRoem, serial = 850032, version = 1.0.12, fwtype = 150")[1] == {
        "model": "RBRoem", "serial": "850032", "version": "1.0.12", "fwtype": "150"}
    words, pairs = gen4.parse_reply("STORAGE used=0 remaining=132120576 size=132120576")
    assert words == ["STORAGE"] and pairs["remaining"] == "132120576"
    assert gen4.parse_reply("download d/s/data bytecount = 56000, samplecount=2000, samplestart=0")[1] == {
        "bytecount": "56000", "samplecount": "2000", "samplestart": "0"}


def test_snapshot_and_clock(port):
    link = port(FakeGen4())
    snap = gen4.snapshot(link)
    assert snap["status"] == "sampling" and snap["status_l2"] == "logging"
    assert snap["sampling"] == {"mode": "continuous", "period": "1000"}
    assert [c["label"] for c in snap["channel_list"]] == [c[0] for c in CHANNELS]
    cond = snap["channel_list"][0]
    assert cond["type"] == "cond19" and cond["equation"] == "corr_cond3" and cond["coefficients"]["x1"] == -20.795860e-6
    assert cond["coefficients"]["n0"] == "temperature_00"  # n entries are channel labels, not numbers (3.6.2)
    assert snap["schedules"]["sch_ctd"]["channels"][:3] == ["conductivity_00", "temperature_00", "pressure_00"]
    assert snap["meminfo"]["size"] == 134217728 and snap["storage_access"] == "instrument"
    assert snap["power"]["battery_voltage_V"] == 6.52
    assert snap["power"]["energy_remaining_J"] == pytest.approx(138000 - 100100)
    assert snap["datasets"][0]["status"] == "open"
    assert len(gen4.clock_now(link)) == 14


def test_download_and_decode(port, tmp_path):
    fake = FakeGen4()
    link = port(fake, tmp_path)
    got = gen4.download(link, tmp_path / "part", "210000")
    assert set(got) == set(fake.objects) and all(got[k] == fake.objects[k] for k in got)
    snap = gen4.snapshot(link)
    ds, sch, cols, meta, fmt = gen4.columns(got, snap)
    assert (ds, sch, fmt) == ("dataset_01", "sch_ctd", "<f4") and not meta.warnings
    assert [c["label"] for c in cols] == ["conductivity_00", "temperature_00", "pressure_00"]  # not-stored one left out
    assert cols[0]["coefficients"]["n1"] == 3 and cols[1]["equation"] == "tmp"
    assert meta.serial_number == 210000 and meta.model == "RBRconcerto3" and meta.utc_offset_ms == 3_600_000
    t, v, err, events = gen4.decode(got, snap)
    assert np.array_equal(t, fake.t) and v.shape == (5000, 3)
    assert np.isnan(v[10, 1]) and err[10, 1] == 0xFFC0000E and np.count_nonzero(err) == 1
    ok = ~np.isnan(v)
    assert np.allclose(v[ok], fake.v.astype(np.float32).astype(np.float64)[ok])
    assert events == [(ENABLE_MS + 5000, 0x0F, 0xFFFFFFFFFFFFFFFF)]  # the schedule-2 event is not ours
    transcript = (tmp_path / "t.log").read_text()
    assert "download dataset_01/sch_ctd/data bytecount=32000 bytestart=256" in transcript
    assert "download CRC matched as ccitt-false" in transcript


def test_float64_and_reflected_crc(port, tmp_path):
    fake = FakeGen4(n_samples=100, crc="le", data_format=2)
    link = port(fake, tmp_path)
    got = gen4.download(link, tmp_path / "part", "210000")
    t, v, err, _ = gen4.decode(got, gen4.snapshot(link))
    assert err[10, 1] == 0xFFC0000E and np.isnan(v[10, 1]) and v[0, 0] == 40.0  # float64 keeps full precision
    assert "download CRC matched as reflected" in (tmp_path / "t.log").read_text()


def test_resume_after_failure(port, tmp_path):
    fake = FakeGen4(fail_after_blocks=1)
    link = port(fake, tmp_path)
    with pytest.raises(LoggerError):
        gen4.download(link, tmp_path / "part", "210000")
    part = tmp_path / "part" / "210000__dataset_01__sch_ctd__data.part"
    assert part.stat().st_size == 256 + 32000
    fake2 = FakeGen4()
    link = port(fake2, tmp_path)
    got = gen4.download(link, tmp_path / "part", "210000")
    assert got["dataset_01/sch_ctd/data"] == fake2.data
    assert fake2.downloaded < sum(len(b) for b in fake2.objects.values())
    assert "resuming dataset_01/sch_ctd/data from 32256 bytes" in (tmp_path / "t.log").read_text()


def test_bad_crc_is_retried_then_fails(port, tmp_path):
    link = port(FakeGen4(n_samples=10, bad_crc=True), tmp_path)
    with pytest.raises(LinkError, match="failed after 2 attempts"):
        gen4.download_block(link, "dataset_01/meta", 256, 0, retries=2)


def test_usbhost_storage_refuses_download(port, tmp_path):
    fake = FakeGen4(access="usbhost")
    link = port(fake, tmp_path)
    with pytest.raises(gen4.UsbHostStorage, match="USB drive"):
        gen4.download(link, tmp_path / "part", "210000")
    assert not any(c.startswith("download") for c in fake.commands)


def test_unpadded_fw_info_is_read_too():
    meta = gen4.parse_meta(build_meta(fw_pad=False))
    assert [meta.channels[i]["label"] for i in sorted(meta.channels)] == [c[0] for c in CHANNELS]
    assert meta.channels[3]["coefficients"]["n0"] == 2 and not meta.warnings


def test_metadata_tag_is_checked():
    with pytest.raises(ValueError, match="not Gen4 metadata"):
        gen4.parse_meta(b"\x01" + build_meta()[1:])
