"""A simulated RBRsolo (fwtype 9) behind a pyserial-like port, for offload tests without hardware.

Replies copy the formats SN100689 sent on 2026-09-25 (raw/100689_20260925T204852Z.log). The memory is
a real fwtype-9 image (header, time-sync event, readings); `read data` blocks carry a real CRC.
"""

from __future__ import annotations

import datetime as dt
import struct
import threading
import time

import numpy as np

from rbr_tpw.crc import crc16_ccitt

MEMORY_SIZE = 132_120_576


def solo_image(n: int, sync_s: int, serial: int = 100689, period_ms: int = 500) -> bytes:
    """fwtype-9 memory: 512-byte header, a time-sync event at `sync_s` (s since 2000), n readings."""
    hdr = bytearray(b"\xff" * 512)
    for off, v in ((0, 512), (4, 1000), (8, serial), (12, sync_s), (16, sync_s), (20, 3_155_759_999),
                   (24, period_ms), (40, 0)):
        struct.pack_into("<I", hdr, off, v)
    body = bytes([0x01, 0xF7]) + struct.pack("<I", sync_s)
    readings = (np.linspace(0.40, 0.45, n) * (1 << 30)).astype("<u4")
    return bytes(hdr) + struct.pack(">H", crc16_ccitt(body)) + body + readings.tobytes()


class FakeSolo:
    """Enough of an RBRsolo's command set for a read-only offload: id, now, status, settings, meminfo,
    powerstatus and `read data`. `bytes_per_s` throttles `read data` like a real transfer;
    `fail_at_offset` makes `read data` at that offset reply with a logger error."""

    def __init__(self, port: str, serial: int = 100689, n_samples: int = 2000, skew_s: float = 0.0,
                 bytes_per_s: float | None = None, fail_at_offset: int | None = None):
        self.port = port
        self.serial = serial
        self.skew_s = skew_s
        self.bytes_per_s = bytes_per_s
        self.fail_at_offset = fail_at_offset
        self.image = solo_image(n_samples, 843_696_000, serial)
        self.timeout = 0.02
        self.commands: list[str] = []
        self.reads: list[tuple[float, float]] = []  # (start, end) host time of each `read data` transfer
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

    # --- logger behaviour
    def _reply(self, text: str):
        with self._lock:
            self._out += text.encode("ascii") + b"\r\nReady: "

    def _handle(self, cmd: str):
        self.commands.append(cmd)
        used = len(self.image)
        now = dt.datetime.fromtimestamp(time.time() + self.skew_s, dt.UTC)
        replies = {
            "": None,
            "id": f"id model = RBRsolo, version = 1.000, serial = {self.serial}, fwtype = 9",
            "now": f"now = {now:%Y%m%d%H%M%S}",
            "status": "status = logging",
            "starttime": "starttime = 20000101000010",
            "endtime": "endtime = 20991231235959",
            "sampling": "sampling mode = continuous, period = 500",
            "channels": "channels count = 1, latency = 100, readtime = 350, minperiod = 550",
            "channel 1": "channel 1 type = temp02, equation = tmp, factoryunits = C, userunits = C, module = 1, "
                         "latency = 100, readtime = 350",
            "calibration 1": "calibration 1 type = temp02, datetime = 20160212143123, c0 = 3B639047, "
                             "c1 = B9845460, c2 = 36209858, c3 = B3AD3E1B",
            "meminfo": f"meminfo used = {used}, remaining = {MEMORY_SIZE - used}, size = {MEMORY_SIZE}",
            "powerstatus": "powerstatus source = usb, int = 3634, remaining = 1ACDF88",
        }
        if cmd in replies:
            if replies[cmd] is None:
                with self._lock:
                    self._out += b"Ready: "
            else:
                self._reply(replies[cmd])
            return
        if cmd.startswith("read data 1 "):
            size, offset = (int(x) for x in cmd.split()[3:5])
            if offset == self.fail_at_offset:
                self._reply("E0104 simulated read failure")
                return
            block = self.image[offset : offset + size]
            t0 = time.time()
            with self._lock:
                self._out += f"data 1 {len(block)} {offset}\r\n".encode() + block
                self._out += struct.pack(">H", crc16_ccitt(block)) + b"Ready: "
                self._ready_at = t0 + (len(block) / self.bytes_per_s if self.bytes_per_s else 0)
            self.reads.append((t0, max(t0, self._ready_at)))
            return
        self._reply("E0102 invalid command")
