"""Simulated RBR loggers behind pyserial-like ports, for offload tests without hardware.

Reply formats copy what real loggers sent: SN100689 (fwtype 9, raw/100689_20260925T204852Z.log) and, in
Ruskin 2.26.1's serial logs, SN076313 (fwtype 0), SN081500 (RBRduet, 102), SN060275 (RBRconcerto, 103)
and SN233442 (RBRconcerto3, 104). Memory images are real formats: L2 rawbin (header, time-sync event,
readings) and Gen3 EasyParse (dataset 1 records, dataset 0 events); transfers carry a real CRC.
"""

from __future__ import annotations

import datetime as dt
import re
import struct
import threading
import time

import numpy as np

from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.lock import unlock_key

MEMORY_SIZE = 132_120_576
SYNC_S = 843_696_000  # 2026-09-25T00:00:00Z in seconds since 2000


def solo_event(etype: int, seconds: int) -> bytes:
    """An 8-byte L2 event record: <crc16 BE of bytes 2..7><type><0xF7><uint32 LE seconds since 2000>."""
    body = bytes([etype, 0xF7]) + struct.pack("<I", seconds)
    return struct.pack(">H", crc16_ccitt(body)) + body


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
        if ctype.startswith("pres"):  # one more word after n0 on pressure channels: 0x1111 on SN081500's pres21
            words += struct.pack("<I", 0x1111)
        blobs.append(ctype.encode().ljust(6, b"\0") + struct.pack("<HIB", status, 558_177_718, len(words) // 4)
                     + words)
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
                 skew_s: float = 0.0, drop_every: int | None = None, drop_first: set[str] | None = None):
        self.port = port
        self.bytes_per_s = bytes_per_s
        self.fail_at_offset = fail_at_offset
        self.skew_s = skew_s
        # No reply at all (no line, no prompt), as RBRconcerto SN060275 (fw 1.460) does now and then: to every Nth
        # command (drop_every), or to the first occurrence of each command named in drop_first.
        self.drop_every = drop_every
        self.drop_first = set(drop_first or ())
        self.dropped: list[str] = []
        # The write lock (lock.py). Lockable fakes list hidden channels and numeric statuses only when unlocked, as
        # SN060275 and SN233442 did on 2026-09-26; refuse_unlock answers `lock OFF` with E0108 (wrong key).
        self.lockable = False
        self.locked = True
        self.refuse_unlock = False
        self.unlocks = 0
        self._challenge: dt.datetime | None = None  # the last `now`/`clock` reply, whole seconds
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

    def _handle_lock(self, cmd: str) -> bool:
        """`lock`, `lock OFF = <key>`, `lock on`; remembers the challenge (`now`/`clock`) for the key."""
        from rbr_tpw.lock import unlock_key

        if cmd in ("now", "clock"):
            self._challenge = self.now().replace(microsecond=0)
            self._reply(f"now = {self._challenge:%Y%m%d%H%M%S}" if cmd == "now" else
                        f"clock datetime = {self._challenge:%Y%m%d%H%M%S}, offsetfromutc = +0.00")
            return True
        if cmd == "lock":
            self._reply(f"lock = {'on' if self.locked else 'off'}")
            return True
        if cmd == "lock on":
            self.locked = True
            self._reply("lock = on")
            return True
        if cmd.startswith("lock OFF = "):
            e2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
            secs = int((self._challenge - e2000).total_seconds()) if self._challenge else -1
            if self.refuse_unlock or int(cmd.split("=")[1]) != unlock_key(self.serial, secs):
                self._reply("E0108 invalid argument to command")
            else:
                self.locked = False
                self.unlocks += 1
                self._reply("lock = off")
            return True
        return False

    @staticmethod
    def locked_view(channel_lines: dict[str, str], count_line: str) -> dict[str, str]:
        """What a locked logger shows of `channels` and `channel N`: hidden channels (status bit 0x01) left out,
        the rest renumbered, and `status = on` instead of the bitfield."""
        shown = []
        for i in range(1, len(channel_lines) + 1):
            line = channel_lines[f"channel {i}"]
            status = int(re.search(r"status = (\d+)", line).group(1))
            if not status & 0x01:
                shown.append(re.sub(r"status = \d+", "status = on", line))
        out = {f"channel {k}": re.sub(r"^channel \d+", f"channel {k}", line) for k, line in enumerate(shown, 1)}
        out["channels"] = re.sub(r"count = \d+, on = \d+", f"count = {len(shown)}, on = {len(shown)}", count_line)
        return out

    def _handle(self, cmd: str):
        self.commands.append(cmd)
        if cmd == "":
            with self._lock:
                self._out += b"Ready: "
            return
        if (self.drop_every and len(self.commands) % self.drop_every == 0) or cmd in self.drop_first:
            self.drop_first.discard(cmd)
            self.dropped.append(cmd)  # silence: no reply line, no prompt
            return
        if self.lockable and self._handle_lock(cmd):
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
        self.status = "logging"  # what `status` answers; a test sets "stopped" after appending the 0x02 event
        self.endtime = "20991231235959"

    def set_image(self, image: bytes):
        """Replace the memory (bytes are immutable: both names must be rebound)."""
        self.image = image
        self.datasets[1] = image

    def grow(self, n_samples: int, seed: int = 0):
        """The logger keeps logging: `n_samples` more readings after what is there."""
        rng = np.random.default_rng(seed + len(self.image))
        more = (0x2000_0000 + rng.integers(0, 0x0100_0000, n_samples)).astype("<u4").tobytes()
        self.set_image(self.image + more)

    def append_event(self, etype: int, seconds: int):
        """An 8-byte event record, as the logger writes one (e.g. 0x02 when stopped)."""
        self.set_image(self.image + solo_event(etype, seconds))

    def erase_and_enable(self, sync_s: int):
        """What Ruskin's enable does to the memory: a fresh header (new enable time) and sync marker, no data."""
        self.set_image(solo_image(0, sync_s, self.serial))

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
            "status": f"status = {self.status}",
            "starttime": "starttime = 20000101000010",
            "endtime": f"endtime = {self.endtime}",
            "sampling": sampling,
            "channels": "channels count = 1, latency = 100, readtime = 350, minperiod = 550",
            "channel 1": "channel 1 type = temp02, equation = tmp, factoryunits = C, userunits = C, module = 1, "
                         "latency = 100, readtime = 350",
            "calibration 1": "calibration 1 type = temp02, datetime = 20160212143123, c0 = 3B639047, "
                             "c1 = B9845460, c2 = 36209858, c3 = B3AD3E1B",
            "meminfo": f"meminfo used = {used}, remaining = {MEMORY_SIZE - used}, size = {MEMORY_SIZE}",
            "powerstatus": power,
        }


class FakeSoloWritable(FakeSolo):
    """A FakeSolo that also takes the writes configure() sends, answering as SN100689 did on 2026-09-25
    (output/raw/100689_20260925T2*.log): `now = X`, `starttime = X`, `endtime = X` and
    `sampling mode = M, period = P` are answered with the command text itself; `stop`, `lock`, `permit`,
    `powerstatus remaining`, `verify` and `enable` answer differently, and `memclear` gives only the prompt.
    Writes need `lock OFF = <key>` first, keyed on the last `now` reply. The reply to a write while locked is
    assumed (never seen from the logger: Ruskin always unlocks first)."""

    E2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)

    def __init__(self, port: str, memory_follows: bool = False, **kw):
        """`memory_follows`: the memory changes as the logger's does, for tests of what an offload finds after
        Ruskin's (or --configure's) writes: `stop` appends a 0x02 event, `memclear` empties it, `enable` writes a
        fresh header and time-sync marker. Off, only `used` changes (0 after memclear, 512 after enable)."""
        super().__init__(port, **kw)
        self.memory_follows = memory_follows
        self.state = {"status": "logging", "starttime": "20000101000010", "endtime": "20991231235959",
                      "sampling": "mode = continuous, period = 500", "remaining": "1ACDF88", "used": len(self.image)}
        self.locked = True
        self.challenge: dt.datetime | None = None  # the last `now` reply, whole seconds
        self.offset_s = 0.0  # written by `now = X`: logger clock minus host clock

    def _s2000(self) -> int:
        return int((self.now() - self.E2000).total_seconds())

    def set_image(self, image: bytes):
        super().set_image(image)
        if hasattr(self, "state"):
            self.state["used"] = len(image)

    def now(self) -> dt.datetime:
        return super().now() + dt.timedelta(seconds=self.offset_s)

    def replies(self):
        r = super().replies()
        st = self.state
        r.update({
            "status": f"status = {st['status']}",
            "starttime": f"starttime = {st['starttime']}",
            "endtime": f"endtime = {st['endtime']}",
            "sampling": f"sampling {'schedule = 1, ' if self.fwtype == 0 else ''}{st['sampling']}",
            "meminfo": f"meminfo used = {st['used']}, remaining = {MEMORY_SIZE - st['used']}, size = {MEMORY_SIZE}",
            "powerstatus": ("powerstatus source = usb, int = 3491" if self.fwtype == 0 else
                            f"powerstatus source = usb, int = 3634, remaining = {st['remaining']}"),
        })
        return r

    WRITES = ("stop", "now = ", "starttime = ", "endtime = ", "sampling mode", "powerstatus remaining = ",
              "permit memclear", "memclear", "verify", "enable", "lock OFF = ", "lock on")

    def _handle(self, cmd: str):
        st = self.state
        if cmd == "now":
            self.commands.append(cmd)
            self.challenge = self.now().replace(microsecond=0)
            return self._reply(f"now = {self.challenge:%Y%m%d%H%M%S}")
        if not cmd.startswith(self.WRITES):
            return super()._handle(cmd)
        self.commands.append(cmd)
        if cmd.startswith("lock OFF = "):
            secs = int((self.challenge - self.E2000).total_seconds()) if self.challenge else -1
            self.locked = int(cmd.split("=")[1]) != unlock_key(self.serial, secs)
            return self._reply("E0108" if self.locked else "lock = off")
        if cmd == "lock on":
            self.locked = True
            return self._reply("lock = on")
        if cmd == "verify":  # a read, in the L2 warning form: "E0401 , verify = logging" on SN100689
            if st["used"]:
                return self._reply(f"E0402 , verify = {st['status']}")
            return self._reply("E0401 , verify = logging")
        if self.locked:
            return self._reply("E0102 invalid command")
        if self.fwtype == 0 and cmd.startswith(("endtime = ", "powerstatus remaining = ")):
            return self._reply("E0102 invalid command")  # never seen from a fwtype-0 logger: assumed unsupported
        if cmd == "stop":
            if st["status"] not in ("logging", "pending"):
                return self._reply(f"E0406 stop = {st['status']}")
            st["status"] = "stopped"
            if self.memory_follows:
                self.append_event(0x02, self._s2000())
            return self._reply("stop = stopped")
        if cmd.startswith("now = "):
            t = dt.datetime.strptime(cmd[6:], "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
            self.offset_s = t.timestamp() - time.time() - self.skew_s
            return self._reply(cmd)
        m = re.fullmatch(r"(starttime|endtime) = (\d{14})", cmd)
        if m:
            st[m.group(1)] = m.group(2)
            return self._reply(cmd)
        if cmd.startswith("sampling mode = "):
            st["sampling"] = cmd.split(" ", 1)[1]
            return self._reply(cmd)
        if cmd.startswith("powerstatus remaining = "):
            st["remaining"] = cmd.rsplit(" ", 1)[1]
            return self._reply(f"powerstatus source = usb, int = 0, remaining = {st['remaining']}")
        if cmd == "permit memclear":
            return self._reply("permit = memclear")
        if cmd == "memclear":
            st["used"] = 0
            if self.memory_follows:
                self.set_image(b"")
            with self._lock:
                self._out += b"Ready: "  # no reply line, only the prompt
            return
        if cmd == "enable":
            if st["used"]:
                return self._reply("E0402 memory not empty, erase first")
            st["used"] = 512
            if self.memory_follows:
                self.set_image(solo_image(0, self._s2000(), self.serial))
            st["status"] = "logging" if st["starttime"] <= f"{self.now():%Y%m%d%H%M%S}" else "pending"
            return self._reply("E0401 , enable = logging" if st["status"] == "logging" else "enable = pending")
        return self._reply("E0102 invalid command")


class FakeL2(FakePort):
    """An RBRduet or RBRconcerto: the L2 command set, a sectioned memory header, and the writes Ruskin 2.26.1 sent
    them (its serial logs, 5 duet and 10 concerto enables, 2026-09-11..22), answered as the loggers answered:
    `stop` -> `stop = stopped`; `permit settings` -> `permit = settings`; `settings offsetfromutc = 0.0` ->
    `settings offsetfromutc = +0.00`; `now = X` (which leaves the offset `unknown`, SN082492 2026-11-26),
    `starttime = X`, `endtime = X`, `sampling period = N` and `twistactivation state = on` with the command text;
    `verify` -> `E0402 memory not empty, erase first, verify = stopped`; `enable erasememory = true` -> `E0401
    estimated memory usage exceeds capacity, enable = logging`, or `enable = gated` on a twist-gated concerto that is
    twisted off. `memclear` answers `memclear used = 0` and leaves the duet `disabled`; `verify` on an empty memory
    answers `E0401 estimated memory usage exceeds capacity, verify = logging` (SN082492, 2026-11-26). A write while
    locked is answered E0102 (assumed).

    The memory follows the writes: `stop` appends a 0x02 event, `memclear` empties it, an enable writes a fresh
    header (the logger time and period patched into the model's header, CRC recomputed) and a time-sync marker."""

    E2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
    model = "RBRduet"
    version = "3.220"
    fwtype = 102
    memory_size = 528_482_304
    stored = 3  # readings per sample set
    period_ms = 500
    userperiodlimit = 500

    def __init__(self, port: str, serial: int, **kw):
        super().__init__(port, **kw)
        self.serial = serial
        self.lockable = True
        self.offset_s = 0.0  # written by `now = X`: logger clock minus host clock
        self.twisted_on = False  # a concerto's twist switch
        self.state = {"status": "logging", "starttime": "20000101000000", "endtime": "20991231235959",
                      "mode": "continuous", "period": self.period_ms, "offsetfromutc": "+0.00",
                      "twist": None, "remaining": "1BA8140"}
        self._permit: str | None = None
        self.writes: list[str] = []
        # A write's reply lost: no reply at all to the first command starting with one of `drop_once_prefixes`
        # (never received), or the first one starting with one of `lose_reply_once_prefixes` carried out but its
        # reply lost.
        self.drop_once_prefixes: set[str] = set()
        self.lose_reply_once_prefixes: set[str] = set()

    @property
    def image(self) -> bytes:
        return self.datasets.get(1, b"")

    @image.setter
    def image(self, value: bytes):
        self.datasets[1] = value

    def _s2000(self) -> int:
        return int((self.now() - self.E2000).total_seconds())

    def now(self) -> dt.datetime:
        return super().now() + dt.timedelta(seconds=self.offset_s)

    def grow(self, n_samples: int):
        """The logger keeps logging: `n_samples` more sample sets of signed readings (R in (0.40, 0.46))."""
        readings = (np.linspace(0.40, 0.45, n_samples)[:, None] + 0.01 * np.arange(self.stored)) * (1 << 30)
        self.image = self.image + readings.astype("<i4").tobytes()

    def append_event(self, etype: int, seconds: int):
        self.image = self.image + solo_event(etype, seconds)

    def fresh_image(self, sync_s: int) -> bytes:
        raise NotImplementedError

    def replies(self):
        r = self._replies()
        if self.locked:  # locked: hidden channels left out, `status = on` (SN060275, 2026-09-26)
            r.update(self.locked_view({k: v for k, v in r.items() if re.fullmatch(r"channel \d+", k)}, r["channels"]))
            for k in [k for k in r if k.startswith("calibration ") and int(k.split()[1]) > self.shown_channels()]:
                r.pop(k)  # E0108 like the hidden `channel N`
        return r

    def shown_channels(self) -> int:
        return len([k for k in self._replies() if re.fullmatch(r"channel \d+", k)]) - 1

    def sampling_reply(self) -> str:
        st = self.state
        return (f"sampling schedule = 1, mode = {st['mode']}, period = {st['period']}, gate = none, "
                f"userperiodlimit = {self.userperiodlimit}")

    def common_replies(self) -> dict[str, str]:
        st = self.state
        used = len(self.image)
        return {
            "id": f"id model = {self.model}, version = {self.version}, serial = {self.serial:06d}, "
                  f"fwtype = {self.fwtype}",
            "status": f"status = {st['status']}",
            "starttime": f"starttime = {st['starttime']}",
            "endtime": f"endtime = {st['endtime']}",
            "sampling": self.sampling_reply(),
            "meminfo": f"meminfo used = {used}, remaining = {self.memory_size - used}, size = {self.memory_size}",
            "memformat": "memformat type = rawbin00",
        }

    WRITES = ("stop", "permit ", "settings offsetfromutc = ", "now = ", "starttime = ", "endtime = ",
              "sampling period = ", "sampling mode = ", "twistactivation state = ", "memclear", "verify", "enable",
              "powerstatus remaining = ")

    def _handle(self, cmd: str):
        if not cmd.startswith(self.WRITES):
            return super()._handle(cmd)
        if self._drops(cmd):
            return
        lose = next((x for x in self.lose_reply_once_prefixes if cmd.startswith(x)), None)
        if lose is None:
            return self._handle_write(cmd)
        self.lose_reply_once_prefixes.discard(lose)
        with self._lock:
            n = len(self._out)
        self._handle_write(cmd)
        with self._lock:
            del self._out[n:]

    def _handle_write(self, cmd: str):
        self.commands.append(cmd)
        st = self.state
        if cmd == "verify":  # a read
            if self.image:
                return self._reply(f"E0402 memory not empty, erase first, verify = {st['status']}")
            return self._reply("E0401 estimated memory usage exceeds capacity, verify = logging")  # SN082492
        if self.locked:
            return self._reply("E0102 invalid command")
        self.writes.append(cmd)
        permit, self._permit = self._permit, None  # a permit is for the next command only
        if cmd.startswith("permit "):
            self._permit = cmd.split(" ", 1)[1]
            return self._reply(f"permit = {self._permit}")
        if cmd == "stop":
            if st["status"] not in ("logging", "pending", "gated"):
                return self._reply(f"E0406 stop = {st['status']}")
            st["status"] = "stopped"
            self.append_event(0x02, self._s2000())
            return self._reply("stop = stopped")
        if cmd.startswith("settings offsetfromutc = "):
            if permit != "settings":
                return self._reply("E0110 permission denied")
            st["offsetfromutc"] = f"{float(cmd.rsplit(' ', 1)[1]):+.2f}"
            return self._reply(f"settings offsetfromutc = {st['offsetfromutc']}")
        if cmd.startswith("now = "):
            t = dt.datetime.strptime(cmd[6:], "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
            self.offset_s = t.timestamp() - time.time() - self.skew_s
            st["offsetfromutc"] = "unknown"  # as SN082492 on 2026-11-26, even right after an offset write
            return self._reply(cmd)
        m = re.fullmatch(r"(starttime|endtime) = (\d{14})", cmd)
        if m:
            st[m.group(1)] = m.group(2)
            return self._reply(cmd)
        m = re.fullmatch(r"sampling period = (\d+)", cmd)
        if m:
            if int(m.group(1)) < self.userperiodlimit:
                return self._reply("E0108 invalid argument to command")
            st["period"] = int(m.group(1))
            return self._reply(cmd)
        m = re.fullmatch(r"twistactivation state = (on|off)", cmd)
        if m and st["twist"] is not None:
            st["twist"] = m.group(1)
            return self._reply(cmd)
        if cmd == "memclear":
            if permit != "memclear":
                return self._reply("E0110 permission denied")
            self.image = b""
            st["status"] = "disabled"  # SN082492 after memclear (it read `stopped` after `stop`)
            return self._reply("memclear used = 0")
        m = re.fullmatch(r"enable( erasememory = true(, memformat = RAWBIN00)?)?", cmd)
        if m:
            if self.image and not m.group(1):
                return self._reply("E0402 memory not empty, erase first")
            self.image = self.fresh_image(self._s2000())
            if st["starttime"] > f"{self.now():%Y%m%d%H%M%S}":
                st["status"] = "pending"
                return self._reply("enable = pending")
            if st["twist"] == "on" and not self.twisted_on:
                st["status"] = "gated"
                return self._reply("enable = gated")
            st["status"] = "logging"
            return self._reply("E0401 estimated memory usage exceeds capacity, enable = logging")
        return self._reply("E0102 invalid command")

    def _drops(self, cmd: str) -> bool:
        """Whether FakePort's drop rules silence this command (they are applied to writes too)."""
        prefix = next((x for x in self.drop_once_prefixes if cmd.startswith(x)), None)
        if (self.drop_every and (len(self.commands) + 1) % self.drop_every == 0) or cmd in self.drop_first or prefix:
            self.drop_first.discard(cmd)
            self.drop_once_prefixes.discard(prefix)
            self.commands.append(cmd)
            self.dropped.append(cmd)
            return True
        return False


class FakeDuet(FakeL2):
    """An RBRduet (fwtype 102): temperature, pressure corrected with the hidden compensation thermistor
    (channel 3), rawbin00 memory with a sectioned header. Coefficients from SN081500's `calibration` replies."""

    TEMP = [("c0", 3.4740720e-003), ("c1", -252.09197e-006), ("c2", 2.4784860e-006), ("c3", -85.943688e-009)]
    PRES = [("c0", -2.8169400), ("c1", 200.80464), ("c2", 2.6564110), ("c3", -3.8574970), ("x0", 10.054800),
            ("x1", 17.362684e-003), ("x2", 0.0), ("x3", 0.0), ("x4", 0.0), ("x5", 20.0), ("n0", 3)]
    COMP = [("c0", 3.3924840e-003), ("c1", -277.40445e-006), ("c2", 4.7023100e-006), ("c3", -839.29645e-009)]

    def __init__(self, port: str, serial: int = 81500, n_samples: int = 2000, **kw):
        super().__init__(port, serial, **kw)
        self.image = l2_sectioned_image(n_samples, self.channel_table(), serial)

    def channel_table(self):
        return [("temp12", 0, self.TEMP), ("pres21", 0, self.PRES), ("temp05", 9, self.COMP)]

    def fresh_image(self, sync_s: int) -> bytes:
        return l2_sectioned_image(0, self.channel_table(), self.serial, period_ms=self.state["period"], sync_s=sync_s)

    def _replies(self):
        return {
            **self.common_replies(),
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
            "settings": "settings fetchpoweroffdelay = 8000, sensorpoweralwayson = off, temperature = 15.0000, "
                        "atmosphere = 10.1325010, pressure = 10.1325, conductivity = 42.9140, density = 1.0260206, "
                        f"castdetection = on, offsetfromutc = {self.state['offsetfromutc']}",
            "powerstatus": "powerstatus source = usb, int =  3.61, capacity = 6.480",
            "powerstatus remaining": f"powerstatus remaining = {self.state['remaining']}",
        }


REAL_CONCERTO_HEADER_BYTES = 1037  # tests/real_l2/concerto_reset.bin: version 1014, 9 channels, as SN060275


def concerto_header(serial: int, logger_s: int, period_ms: int) -> bytes:
    """The real RBRconcerto deployment header of tests/real_l2/concerto_reset.bin (fw 1.460, 9 channels: cond06,
    temp09, pres24, five derived, the hidden temp10), with its serial number, logger time at enable and period
    patched into section 2 (which starts at byte 9) and its CRC recomputed."""
    from pathlib import Path

    real = (Path(__file__).parent / "real_l2" / "concerto_reset.bin").read_bytes()
    hdr = bytearray(real[:REAL_CONCERTO_HEADER_BYTES])
    for rel, v in ((7, serial), (11, logger_s), (23, period_ms)):
        struct.pack_into("<I", hdr, 9 + rel, v)
    struct.pack_into(">H", hdr, len(hdr) - 2, crc16_ccitt(bytes(hdr[:-2])))
    return bytes(hdr)


class FakeConcerto(FakeL2):
    """An RBRconcerto (fwtype 103, fw 1.460), twist-gated, with SN060275's replies (Ruskin's serial log,
    2026-09-13) and a real concerto header (concerto_header). Stored per sample set: conductivity, temperature,
    pressure and the hidden temp10; the five derived channels are not stored (status 4)."""

    model = "RBRconcerto"
    version = "1.460"
    fwtype = 103
    memory_size = 132_120_576
    stored = 4
    period_ms = 1000
    userperiodlimit = 167

    def __init__(self, port: str, serial: int = 60275, n_samples: int = 2000, **kw):
        super().__init__(port, serial, **kw)
        self.state["twist"] = "on"
        self.state["status"] = "gated"
        self.state["remaining"] = None
        self.image = self.fresh_image(SYNC_S)
        self.grow(n_samples)

    def fresh_image(self, sync_s: int) -> bytes:
        body = bytes([0x01, 0xF7]) + struct.pack("<I", sync_s)
        return concerto_header(self.serial, sync_s, self.state["period"]) + struct.pack(">H", crc16_ccitt(body)) + body

    def sampling_reply(self) -> str:
        st = self.state
        gate = "twistactivation" if st["twist"] == "on" else "none"
        return (f"sampling schedule = 1, mode = {st['mode']}, period = {st['period']}, burstlength = 10, "
                f"burstinterval = 300000, gate = {gate}, userperiodlimit = {self.userperiodlimit}")

    CHANNELS = [
        ("cond06", 32, 0, 60, 290, "corr_cond", "mS/cm"), ("temp09", 1, 0, 50, 260, "tmp", "C"),
        ("pres24", 2, 0, 50, 290, "corr_pres2", "dbar"), ("pres08", 240, 4, 0, 0, "deri_seapres", "dbar"),
        ("dpth01", 241, 4, 0, 0, "deri_depth", "m"), ("sal_00", 242, 4, 0, 0, "deri_salinity", "PSU"),
        ("sos_00", 244, 4, 0, 0, "deri_sos", "m/s"), ("scon00", 245, 4, 0, 0, "deri_speccond", "uS/cm"),
        ("temp10", 6, 9, 50, 260, "tmp", "C")]
    CALS = [
        "datetime = 20231026170646, c0 = 29.648888e-003, c1 = 154.56442e+000, x0 = 257.00213e-006, "
        "x1 = -9.7100980e-006, x2 = 600.00000e-009, x3 = 14.995120e+000, x4 = 10.000000e+000, n0 = 2, n1 = 3",
        "datetime = 20231023132858, c0 = 3.3587150e-003, c1 = -255.07301e-006, c2 = 2.3641970e-006, "
        "c3 = -78.563912e-009",
        "datetime = 20231025163856, c0 = -31.343394e+000, c1 = 2.3460230e+003, c2 = 69.649632e+000, "
        "c3 = -38.275800e+000, x0 = 10.106700e+000, x1 = 189.41363e-003, x2 = 396.48000e-006, x3 = -88.085144e-009, "
        "x4 = 202.11552e-006, x5 = 21.231548e+000, n0 = 9",
        "datetime = 20170719154601, n0 = 3, n1 = value", "datetime = 20170719154601, n0 = 3, n1 = value",
        "datetime = 20170719154601, n0 = 2, n1 = 3, n2 = 1, n3 = value", "datetime = 20170719154602, n0 = 2, n1 = 4, "
        "n2 = 6", "datetime = 20170719154602, n0 = 1, n1 = 2",
        "datetime = 20231023132914, c0 = 3.3863120e-003, c1 = -277.61296e-006, c2 = 5.4398570e-006, "
        "c3 = -744.36256e-009"]

    def _replies(self):
        r = {**self.common_replies(),
             "channels": "channels count = 9, on = 9, latency = 60, readtime = 290, minperiod = 480",
             "memformat": "memformat type = rawbin00",
             "settings": "settings fetchpoweroffdelay = 8000, sensorpoweralwayson = off, temperature = 0.0000, "
                         "atmosphere = 10.1325000, pressure = 10.1325, density = 1.0281000, speccondtempco = 0.0191, "
                         "salinity = 35.0000, avgsoundspeed = 1550.7440, castdetection = off, inputtimeout = 10000, "
                         f"offsetfromutc = {self.state['offsetfromutc']}",
             "powerstatus": "powerstatus source = usb, int = 12.34, ext =  0.00, capacity = 24.000",
             "twistactivation": f"twistactivation state = {self.state['twist']}, location = off"}
        for i, (t, mod, status, lat, rt, eq, units) in enumerate(self.CHANNELS, 1):
            r[f"channel {i}"] = (f"channel {i} type = {t}, module = {mod}, status = {status}, latency = {lat}, "
                                 f"readtime = {rt}, equation = {eq}, userunits = {units}")
            r[f"calibration {i}"] = f"calibration {i} type = {t}, {self.CALS[i - 1]}"
        return r


class FakeConcerto3(FakePort):
    """An RBRconcerto3 (fwtype 104, Gen3 L3), EasyParse (calbin00) memory in datasets 2, 1 and 0.

    It takes the writes of configure._apply_gen3, answered as SN233442 answered Ruskin (its serial logs,
    2026-09-11..16) or as the L3 reference rev L shows: `disable` -> `disable status = stopped`; `clock datetime = X,
    offsetfromutc = 0.0` -> `clock datetime = X, offsetfromutc = +0.00` (`clock datetime = X` alone erases the offset,
    4.1.1); `deployment endtime = E, starttime = S` -> `deployment starttime = S, endtime = E`; `sampling period = N`
    with the text (sub-second N only from `availablefastperiods`); `twistactivation enabled = true` -> `twistactivation
    enabled = true, state = n/a`; `verify` -> `E0402 memory not empty, erase first`; `enable erasememory = true,
    memformat = CALBIN00` -> `enable status = gated, warning = none`; `permit command = memclear` + `memclear` ->
    `memclear used = 0`. Writes need the unlock (keyed on the `clock` reply). The memory follows: an enable writes a
    new deployment header (dataset 2) and empties datasets 0 and 1, memclear empties all three."""

    CHANNELS = [("cond19", "corr_cond3", "mS/cm", "conductivity_00", 0), ("temp14", "tmp", "C", "temperature_00", 0),
                ("pres24", "corr_pres2", "dbar", "pressure_00", 0), ("pres08", "deri_seapres", "dbar",
                                                                      "seapressure_00", 0),
                ("dpth01", "deri_depth", "m", "depth_00", 0), ("sal_00", "deri_salinity", "PSU", "salinity_00", 0),
                ("temp22", "tmp", "C", "conductivitycelltemperature_00", 13),
                ("temp10", "tmp", "C", "pressuretemperature_00", 13)]
    FAST = [500, 250, 125, 63, 42, 31]

    def __init__(self, port: str, serial: int = 233442, n_samples: int = 2000, **kw):
        """Locked, it lists its 6 stored channels with `status = on`, as SN233442 did on 2026-09-26; unlocked (the
        key from its `clock` reply), all 8 with the channelStatus bitfield, as in Ruskin's logs."""
        super().__init__(port, **kw)
        self.serial = serial
        self.lockable = True
        self.t0 = 1_790_000_000_000
        self.datasets = {2: bytes(range(256)) * 4 + bytes(180),  # 1204 bytes like SN233442's header; not decoded
                         1: easyparse_image(n_samples, 6, self.t0),
                         0: easyparse_event(self.t0 - 500, 0x18) + easyparse_event(self.t0 + 1000 * n_samples, 0x19)}
        self.n = n_samples
        self.offset_s = 0.0
        self.twisted_on = False
        self.state = {"status": "logging", "starttime": "20000101000000", "endtime": "20991231235959",
                      "period": 1000, "twist": "false", "offsetfromutc": "+0.00"}
        self._permit: str | None = None
        self.writes: list[str] = []

    def now(self) -> dt.datetime:
        return super().now() + dt.timedelta(seconds=self.offset_s)

    def grow(self, n_samples: int):
        self.datasets[1] = self.datasets[1] + easyparse_image(n_samples, 6, self.t0 + 1000 * self.n)
        self.n += n_samples

    def replies(self):
        r = self._replies()
        if self.locked:
            r.update(self.locked_view({k: v for k, v in r.items() if re.fullmatch(r"channel \d+", k)}, r["channels"]))
        return r

    def _replies(self):
        st = self.state
        gate = "twistactivation" if st["twist"] == "true" else "none"
        tstate = "n/a" if st["twist"] != "true" or st["status"] not in ("logging", "gated") else (
            "running" if st["status"] == "logging" else "paused")
        r = {
            "id": f"id model = RBRconcerto3, version = 1.162, serial = {self.serial}, fwtype = 104",
            "clock": f"clock datetime = {self.now():%Y%m%d%H%M%S}, offsetfromutc = {st['offsetfromutc']}",
            "deployment": f"deployment starttime = {st['starttime']}, endtime = {st['endtime']}, "
                          f"status = {st['status']}",
            "deployment status": f"deployment status = {st['status']}",
            "sampling": f"sampling mode = continuous, period = {st['period']}, burstlength = 10, "
                        f"burstinterval = 300000, gate = {gate}, userperiodlimit = 31",
            "sampling availablefastperiods": f"sampling availablefastperiods = {'|'.join(map(str, self.FAST))}",
            "twistactivation": f"twistactivation enabled = {st['twist']}, state = {tstate}",
            "channels": "channels count = 8, on = 8, settlingtime = 60, readtime = 290, minperiod = 470",
            "memformat": "memformat type = calbin00, newtype = calbin00, availabletypes = rawbin00|calbin00",
            "power": "power source = usb, int = 14.63, ext = 0.01, reg = n/a",
            "powerinternal": "powerinternal batterytype = lisocl2, capacity = 232.0e+003, used = 28.93e+003",
            "settings": "settings fetchpoweroffdelay = 8000, sensorpoweralwayson = off, atmosphere = 10.1325000",
            "info": "info pn = L3-M11-F15-BEC11-OP1-G1-SCT12-SP11, fwlock = off",
        }
        for i, (typ, eq, units, label, status) in enumerate(self.CHANNELS, 1):
            r[f"channel {i}"] = (f"channel {i} type = {typ}, module = {i}, status = {status}, settlingtime = 60, "
                                 f"readtime = 290, equation = {eq}, userunits = {units}, "
                                 f"derived = {'on' if eq.startswith('deri_') else 'off'}, label = {label}")
            r[f"calibration {i}"] = (f"calibration {i} label = {label}, datetime = 20250528165608, "
                                     f"c0 = 30.056130e-003, c1 = 158.07142e+000")
        for d in (0, 1, 2):
            blob = self.datasets.get(d, b"")
            size = MEMORY_SIZE
            r[f"meminfo dataset {d}"] = (f"meminfo dataset = {d}, used = {len(blob)}, remaining = "
                                         f"{size - len(blob)}, size = {size}")
        return r

    WRITES = ("disable", "clock datetime = ", "clock offsetfromutc = ", "deployment endtime = ",
              "deployment starttime = ", "sampling period = ", "twistactivation enabled = ", "verify", "enable",
              "permit command = ", "memclear")

    def _handle(self, cmd: str):
        if cmd == "clock" and not (self.drop_every or self.drop_first):  # the challenge, with the offset as it is
            self.commands.append(cmd)
            self._challenge = self.now().replace(microsecond=0)
            return self._reply(f"clock datetime = {self._challenge:%Y%m%d%H%M%S}, "
                               f"offsetfromutc = {self.state['offsetfromutc']}")
        if not cmd.startswith(self.WRITES):
            return super()._handle(cmd)
        self.commands.append(cmd)
        st = self.state
        if cmd == "verify":
            if self.datasets.get(1) or self.datasets.get(0):
                return self._reply("E0402 memory not empty, erase first")
            return self._reply("verify status = logging, warning = none")
        if self.locked:
            return self._reply("E0102 invalid command")
        self.writes.append(cmd)
        permit, self._permit = self._permit, None
        if cmd.startswith("permit command = "):
            self._permit = cmd.split("= ", 1)[1]
            return self._reply(cmd)
        if cmd == "disable":
            st["status"] = "stopped"
            return self._reply("disable status = stopped")
        m = re.fullmatch(r"clock datetime = (\d{14})(, offsetfromutc = ([-+0-9.]+))?", cmd)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
            self.offset_s = t.timestamp() - time.time() - self.skew_s
            st["offsetfromutc"] = f"{float(m.group(3)):+.2f}" if m.group(3) else "unknown"
            return self._reply(f"clock datetime = {m.group(1)}" + (f", offsetfromutc = {st['offsetfromutc']}"
                                                                    if m.group(3) else ""))
        m = re.fullmatch(r"clock offsetfromutc = ([-+0-9.]+)", cmd)
        if m:
            st["offsetfromutc"] = f"{float(m.group(1)):+.2f}"
            return self._reply(f"clock offsetfromutc = {st['offsetfromutc']}")
        m = re.fullmatch(r"deployment endtime = (\d{14}), starttime = (\d{14})", cmd)
        if m:
            st["endtime"], st["starttime"] = m.group(1), m.group(2)
            return self._reply(f"deployment starttime = {st['starttime']}, endtime = {st['endtime']}")
        m = re.fullmatch(r"sampling period = (\d+)", cmd)
        if m:
            n = int(m.group(1))
            if n < 1000 and n not in self.FAST:
                return self._reply(f"E0108 invalid argument to command: '{n}'")
            st["period"] = n
            return self._reply(cmd)
        m = re.fullmatch(r"twistactivation enabled = (true|false)", cmd)
        if m:
            st["twist"] = m.group(1)
            return self._reply(f"twistactivation enabled = {st['twist']}, state = n/a")
        if cmd == "memclear":
            if permit != "memclear":
                return self._reply("E0103 protected command, use 'permit command = <command>'")
            self.datasets = {0: b"", 1: b"", 2: b""}
            st["status"] = "stopped"
            return self._reply("memclear used = 0")
        m = re.fullmatch(r"enable( erasememory = true)?(, memformat = CALBIN00)?", cmd)
        if m:
            if (self.datasets.get(1) or self.datasets.get(0)) and not m.group(1):
                return self._reply("E0402 memory not empty, erase first")
            stamp = f"{self.now():%Y%m%d%H%M%S}".encode()
            self.datasets = {2: (stamp * 100)[:1204], 1: b"", 0: b""}  # a new deployment header
            self.n, self.t0 = 0, int(self.now().timestamp() * 1000)
            st["status"] = "gated" if st["twist"] == "true" and not self.twisted_on else "logging"
            return self._reply(f"enable status = {st['status']}, warning = none")
        return self._reply("E0102 invalid command")


# --- real transports: the simulated loggers behind a TCP socket (any OS) or a pseudo-terminal (POSIX), so the
# code under test opens them with real pyserial (serial_for_url), not a replaced one.

def serve_socket(fake: FakePort):
    """Serve `fake` on 127.0.0.1: returns ("socket://127.0.0.1:<port>", stop). Accepts one client at a time,
    again after each disconnect (so an offload can be resumed)."""
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(0.05)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.settimeout(0.002)
            with conn:
                while not stop.is_set():
                    try:
                        data = conn.recv(4096)
                        if not data:
                            break  # the client closed the port
                        fake.write(data)
                    except TimeoutError:
                        pass
                    except OSError:
                        break
                    out = fake.read(65536)
                    if out:
                        try:
                            conn.sendall(out)
                        except OSError:
                            break

    fake.timeout = 0.002  # keep the relay loop responsive
    t = threading.Thread(target=run, name=f"fake-socket-{fake.port}", daemon=True)
    t.start()

    def close():
        stop.set()
        t.join(2)
        srv.close()

    return f"socket://127.0.0.1:{srv.getsockname()[1]}", close


def serve_pty(fake: FakePort):
    """Serve `fake` on a pseudo-terminal (macOS/Linux): returns (slave device path, stop). pyserial opens the
    path like a USB serial port: termios settings, exclusive flock, real read timeouts."""
    import os
    import select

    master, slave = os.openpty()  # slave kept open so the master never reads EIO between clients
    path = os.ttyname(slave)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            ready, _, _ = select.select([master], [], [], 0.002)
            if ready:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    data = b""
                if data:
                    fake.write(data)
            out = fake.read(65536)
            while out and not stop.is_set():
                out = out[os.write(master, out):]

    fake.timeout = 0.002  # keep the relay loop responsive
    t = threading.Thread(target=run, name=f"fake-pty-{fake.port}", daemon=True)
    t.start()

    def close():
        stop.set()
        t.join(2)
        os.close(master)
        os.close(slave)

    return path, close


def serve_com(fake: FakePort, device: str):
    """Serve `fake` on one end of a virtual COM port pair (com0com on Windows); the code under test opens the
    other end, through pyserial's Windows serial backend. Returns stop."""
    import serial

    ser = serial.Serial(device, 115200, timeout=0.002)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            data = ser.read(4096)
            if data:
                fake.write(data)
            out = fake.read(65536)
            if out:
                ser.write(out)

    fake.timeout = 0.002  # keep the relay loop responsive
    t = threading.Thread(target=run, name=f"fake-com-{device}", daemon=True)
    t.start()

    def close():
        stop.set()
        t.join(2)
        ser.close()

    return close
