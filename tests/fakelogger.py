"""Simulated RBR loggers behind pyserial-like ports, for offload tests without hardware.

Reply formats copy what real loggers sent: SN100689 (fwtype 9, raw/100689_20260925T204852Z.log) and, in
Ruskin 2.26.1's serial logs, SN076313 (fwtype 0), SN081500 (RBRduet, 102), SN060275 (RBRconcerto, 103)
and SN233442 (RBRconcerto3, 104). Memory images are real formats: L2 rawbin (header, time-sync event,
readings) and Gen3 EasyParse (dataset 1 records, dataset 0 events); transfers carry a real CRC.
"""

from __future__ import annotations

import datetime as dt
import struct
import threading
import time

import numpy as np

from rbr_tpw.crc import crc16_ccitt

MEMORY_SIZE = 132_120_576
SYNC_S = 843_696_000  # 2026-09-25T00:00:00Z in seconds since 2000


def solo_image(n: int, sync_s: int = SYNC_S, serial: int = 100689, period_ms: int = 500, nchan: int = 1) -> bytes:
    """L2 rawbin memory: 512-byte header, a time-sync event at `sync_s` (s since 2000), n sample sets."""
    hdr = bytearray(b"\xff" * 512)
    for off, v in ((0, 512), (4, 1000), (8, serial), (12, sync_s), (16, sync_s), (20, 3_155_759_999),
                   (24, period_ms), (40, 0)):
        struct.pack_into("<I", hdr, off, v)
    body = bytes([0x01, 0xF7]) + struct.pack("<I", sync_s)
    readings = np.repeat(np.linspace(0.40, 0.45, n)[:, None], nchan, axis=1)
    readings = (readings + 0.01 * np.arange(nchan)) * (1 << 30)
    return bytes(hdr) + struct.pack(">H", crc16_ccitt(body)) + body + readings.astype("<u4").tobytes()


def l2_sectioned_image(n: int, channels: list[tuple[str, int, list[tuple[str, float]]]], serial: int = 81500,
                       period_ms: int = 500, sync_s: int = SYNC_S) -> bytes:
    """RBRduet/RBRconcerto memory: a sectioned deployment header (version 1009 layout, L3 ref 5.3.1 with Ruskin's
    L2HeaderDeploymentVersionOffsets), a CRC, a time-sync event, then n sample sets of signed readings.
    channels: (type, status, [(coefficient name, value)]) with names in `calibration N` order (c.., x.., n..)."""
    s2 = bytearray(b"\xff" * 503)
    s2[0], s2[1:3] = 2, struct.pack("<H", 503)
    for rel, v in ((3, 3220), (7, serial), (11, sync_s), (15, 0), (19, 3_155_759_999), (23, period_ms), (27, 2),
                   (31, 2), (35, 9600), (39, 0)):
        struct.pack_into("<I", s2, rel, v)
    blobs = []
    for ctype, status, coeffs in channels:
        words = b"".join(struct.pack("<i", int(v)) if name.startswith("n") else struct.pack("<f", v)
                         for name, v in coeffs)
        blobs.append(ctype.encode().ljust(6, b"\0") + struct.pack("<HIB", status, 558_177_718, len(coeffs)) + words)
    head = 4 + 2 * len(blobs)
    starts, off = [], head
    for blob in blobs:
        starts.append(off)
        off += len(blob)
    s3 = bytes([3]) + struct.pack("<H", off) + bytes([len(blobs)]) + b"".join(struct.pack("<H", a) for a in starts)
    s3 += b"".join(blobs)
    length = 9 + len(s2) + len(s3) + 2
    hdr = bytes([1]) + struct.pack("<H", 9) + struct.pack("<I", 1009) + struct.pack("<H", length) + bytes(s2) + s3
    body = bytes([0x01, 0xF7]) + struct.pack("<I", sync_s)
    stored = sum(1 for _, status, _ in channels if not status & 0x04)
    readings = (np.linspace(0.40, 0.45, n)[:, None] + 0.01 * np.arange(stored)) * (1 << 30)
    return (hdr + struct.pack(">H", crc16_ccitt(hdr)) + struct.pack(">H", crc16_ccitt(body)) + body
            + readings.astype("<i4").tobytes())


def easyparse_image(n: int, nchan: int, t0_ms: int, period_ms: int = 1000) -> bytes:
    """Gen3 EasyParse dataset 1: int64 ms + float32 per channel (L3 ref 5.2)."""
    rec = np.dtype([("t", "<i8"), ("v", "<f4", (nchan,))])
    a = np.zeros(n, rec)
    a["t"] = t0_ms + period_ms * np.arange(n)
    a["v"] = np.linspace(10, 20, n)[:, None] + np.arange(nchan)
    return a.tobytes()


def easyparse_event(ms: int, etype: int, payload: int = 0xFFFFFFFF) -> bytes:
    body = bytes([etype, 0xF4]) + struct.pack("<QI", ms, payload)
    return struct.pack(">H", crc16_ccitt(body)) + body


class FakePort:
    """pyserial surface plus `read data` (L2) and `readdata` (Gen3) transfers of `self.datasets`.
    `bytes_per_s` throttles transfers like a real logger; `fail_at_offset` makes a transfer fail."""

    def __init__(self, port: str, bytes_per_s: float | None = None, fail_at_offset: int | None = None,
                 skew_s: float = 0.0):
        self.port = port
        self.bytes_per_s = bytes_per_s
        self.fail_at_offset = fail_at_offset
        self.skew_s = skew_s
        self.datasets: dict[int, bytes] = {}
        self.timeout = 0.02
        self.commands: list[str] = []
        self.reads: list[tuple[float, float]] = []  # (start, end) host time of each transfer
        self._out = bytearray()
        self._in = b""
        self._ready_at = 0.0
        self._lock = threading.Lock()

    # --- pyserial surface used by rbr_tpw.link.Link
    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._out) if time.time() >= self._ready_at else 0

    def reset_input_buffer(self):
        with self._lock:
            self._out.clear()

    def write(self, data: bytes) -> int:
        self._in += data
        while b"\r" in self._in:
            line, self._in = self._in.split(b"\r", 1)
            self._handle(line.decode("ascii"))
        return len(data)

    def read(self, n: int = 1) -> bytes:
        deadline = time.time() + self.timeout
        while True:
            with self._lock:
                if self._out and time.time() >= self._ready_at:
                    chunk = bytes(self._out[:n])
                    del self._out[:n]
                    return chunk
            if time.time() >= deadline:
                return b""
            time.sleep(0.002)

    def close(self):
        pass

    # --- behaviour
    def now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(time.time() + self.skew_s, dt.UTC)

    def replies(self) -> dict[str, str]:
        raise NotImplementedError

    def _reply(self, text: str):
        with self._lock:
            self._out += text.encode("ascii") + b"\r\nReady: "

    def _transfer(self, header: str, dataset: int, size: int, offset: int):
        if offset == self.fail_at_offset:
            self._reply("E0104 simulated read failure")
            return
        block = self.datasets[dataset][offset : offset + size]
        t0 = time.time()
        with self._lock:
            self._out += header.format(d=dataset, n=len(block), o=offset).encode() + block
            self._out += struct.pack(">H", crc16_ccitt(block)) + b"Ready: "
            self._ready_at = t0 + (len(block) / self.bytes_per_s if self.bytes_per_s else 0)
        self.reads.append((t0, max(t0, self._ready_at)))

    def _handle(self, cmd: str):
        self.commands.append(cmd)
        if cmd == "":
            with self._lock:
                self._out += b"Ready: "
            return
        if cmd.startswith("read data "):
            dataset, size, offset = (int(x) for x in cmd.split()[2:5])
            self._transfer("data {d} {n} {o}\r\n", dataset, size, offset)
            return
        if cmd.startswith("readdata "):
            kv = dict(part.strip().split(" = ") for part in cmd[len("readdata "):].split(","))
            self._transfer("readdata dataset = {d}, size = {n}, offset = {o}\r\n", int(kv["dataset"]),
                           int(kv["size"]), int(kv["offset"]))
            return
        r = self.replies()
        self._reply(r[cmd] if cmd in r else "E0102 invalid command")


class FakeSolo(FakePort):
    """An RBRsolo (fwtype 9, or 0 with `fwtype=0`), one temperature channel."""

    def __init__(self, port: str, serial: int = 100689, n_samples: int = 2000, fwtype: int = 9, **kw):
        super().__init__(port, **kw)
        self.serial, self.fwtype = serial, fwtype
        self.image = solo_image(n_samples, SYNC_S, serial)
        self.datasets = {1: self.image}

    def replies(self):
        used = len(self.image)
        sampling = ("sampling schedule = 1, mode = continuous, period = 500" if self.fwtype == 0
                    else "sampling mode = continuous, period = 500")
        power = ("powerstatus source = usb, int = 3491" if self.fwtype == 0
                 else "powerstatus source = usb, int = 3634, remaining = 1ACDF88")
        return {
            "id": f"id model = RBRsolo, version = {'1.110' if self.fwtype == 0 else '1.000'}, "
                  f"serial = {self.serial}, fwtype = {self.fwtype}",
            "now": f"now = {self.now():%Y%m%d%H%M%S}",
            "status": "status = logging",
            "starttime": "starttime = 20000101000010",
            "endtime": "endtime = 20991231235959",
            "sampling": sampling,
            "channels": "channels count = 1, latency = 100, readtime = 350, minperiod = 550",
            "channel 1": "channel 1 type = temp02, equation = tmp, factoryunits = C, userunits = C, module = 1, "
                         "latency = 100, readtime = 350",
            "calibration 1": "calibration 1 type = temp02, datetime = 20160212143123, c0 = 3B639047, "
                             "c1 = B9845460, c2 = 36209858, c3 = B3AD3E1B",
            "meminfo": f"meminfo used = {used}, remaining = {MEMORY_SIZE - used}, size = {MEMORY_SIZE}",
            "powerstatus": power,
        }


class FakeDuet(FakePort):
    """An RBRduet (fwtype 102): temperature, pressure corrected with the hidden compensation thermistor
    (channel 3), rawbin00 memory with a sectioned header. Coefficients from SN081500's `calibration` replies."""

    TEMP = [("c0", 3.4740720e-003), ("c1", -252.09197e-006), ("c2", 2.4784860e-006), ("c3", -85.943688e-009)]
    PRES = [("c0", -2.8169400), ("c1", 200.80464), ("c2", 2.6564110), ("c3", -3.8574970), ("x0", 10.054800),
            ("x1", 17.362684e-003), ("x2", 0.0), ("x3", 0.0), ("x4", 0.0), ("x5", 20.0), ("n0", 3)]
    COMP = [("c0", 3.3924840e-003), ("c1", -277.40445e-006), ("c2", 4.7023100e-006), ("c3", -839.29645e-009)]

    def __init__(self, port: str, serial: int = 81500, n_samples: int = 2000, **kw):
        super().__init__(port, **kw)
        self.serial = serial
        self.image = l2_sectioned_image(n_samples, [("temp12", 0, self.TEMP), ("pres21", 0, self.PRES),
                                                    ("temp05", 9, self.COMP)], serial)
        self.datasets = {1: self.image}

    def replies(self):
        used = len(self.image)
        return {
            "id": f"id model = RBRduet, version = 3.220, serial = {self.serial:06d}, fwtype = 102",
            "now": f"now = {self.now():%Y%m%d%H%M%S}",
            "status": "status = logging",
            "starttime": "starttime = 20000101000000",
            "endtime": "endtime = 20991231235959",
            "sampling": "sampling mode = continuous, period = 500",
            "channels": "channels count = 3, on = 3, latency = 50, readtime = 300, minperiod = 430",
            "channel 1": "channel 1 type = temp12, module = 1, status = 0, latency = 50, readtime = 300, "
                         "equation = tmp, userunits = C",
            "channel 2": "channel 2 type = pres21, module = 2, status = 0, latency = 50, readtime = 300, "
                         "equation = corr_pres2, userunits = dbar",
            "channel 3": "channel 3 type = temp05, module = 6, status = 9, latency = 33, readtime = 40, "
                         "equation = tmp, userunits = C",
            **{f"calibration {i}": f"calibration {i} type = {t}, datetime = 20170908092158, "
                                   + ", ".join(f"{k} = {v:.7e}" if not k.startswith("n") else f"{k} = {v}"
                                               for k, v in cal)
               for i, (t, cal) in enumerate([("temp12", self.TEMP), ("pres21", self.PRES), ("temp05", self.COMP)],
                                            1)},
            "meminfo": f"meminfo used = {used}, remaining = {528_482_304 - used}, size = 528482304",
            "memformat": "memformat type = rawbin00",
            "settings": "settings fetchpoweroffdelay = 8000, sensorpoweralwayson = off, temperature = 15.0000, "
                        "atmosphere = 10.1325010, pressure = 10.1325, conductivity = 42.9140, density = 1.0260206",
            "powerstatus": "powerstatus source = usb, int =  3.61, capacity = 6.480",
            "powerstatus remaining": "powerstatus remaining = 1BA8140",
        }


class FakeConcerto3(FakePort):
    """An RBRconcerto3 (fwtype 104, Gen3 L3), EasyParse (calbin00) memory in datasets 2, 1 and 0."""

    CHANNELS = [("cond19", "corr_cond3", "mS/cm", "conductivity_00", 0), ("temp14", "tmp", "C", "temperature_00", 0),
                ("pres24", "corr_pres2", "dbar", "pressure_00", 0), ("pres08", "deri_seapres", "dbar",
                                                                      "seapressure_00", 0),
                ("dpth01", "deri_depth", "m", "depth_00", 0), ("sal_00", "deri_salinity", "PSU", "salinity_00", 0),
                ("temp22", "tmp", "C", "conductivitycelltemperature_00", 13),
                ("temp10", "tmp", "C", "pressuretemperature_00", 13)]

    def __init__(self, port: str, serial: int = 233442, n_samples: int = 2000, **kw):
        super().__init__(port, **kw)
        self.serial = serial
        t0 = 1_790_000_000_000
        self.datasets = {2: bytes(range(256)) * 4 + bytes(180),  # 1204 bytes like SN233442's header; not decoded
                         1: easyparse_image(n_samples, 6, t0),
                         0: easyparse_event(t0 - 500, 0x18) + easyparse_event(t0 + 1000 * n_samples, 0x19)}

    def replies(self):
        r = {
            "id": f"id model = RBRconcerto3, version = 1.162, serial = {self.serial}, fwtype = 104",
            "clock": f"clock datetime = {self.now():%Y%m%d%H%M%S}, offsetfromutc = +0.00",
            "deployment": "deployment starttime = 20000101000000, endtime = 20991231235959, status = logging",
            "deployment status": "deployment status = logging",
            "sampling": "sampling mode = continuous, period = 1000",
            "channels": "channels count = 8, on = 8, settlingtime = 60, readtime = 290, minperiod = 470",
            "memformat": "memformat type = calbin00, newtype = calbin00, availabletypes = rawbin00|calbin00",
            "power": "power source = usb, int = 14.63, ext = 0.01, reg = n/a",
            "powerinternal": "powerinternal batterytype = lisocl2, capacity = 232.0e+003, used = 28.93e+003",
            "settings": "settings fetchpoweroffdelay = 8000, sensorpoweralwayson = off, atmosphere = 10.1325000",
            "info": "info pn = L3-M11-F15-BEC11-OP1-G1-SCT12-SP11, fwlock = off",
        }
        for i, (typ, eq, units, label, status) in enumerate(self.CHANNELS, 1):
            r[f"channel {i}"] = (f"channel {i} type = {typ}, module = {i}, status = {status}, settlingtime = 60, "
                                 f"readtime = 290, equation = {eq}, userunits = {units}, derived = off, "
                                 f"label = {label}")
            r[f"calibration {i}"] = (f"calibration {i} label = {label}, datetime = 20250528165608, "
                                     f"c0 = 30.056130e-003, c1 = 158.07142e+000")
        for d, blob in self.datasets.items():
            size = MEMORY_SIZE
            r[f"meminfo dataset {d}"] = (f"meminfo dataset = {d}, used = {len(blob)}, remaining = "
                                         f"{size - len(blob)}, size = {size}")
        return r
