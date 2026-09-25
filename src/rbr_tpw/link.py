"""Serial command link to an RBR logger over its USB CDC port.

Protocol notes (RBRsolo fwtype 9, observed 2026-09-25 and in Ruskin 2.26.1 serial logs):
- Commands are ASCII terminated by CR. Replies are one line terminated by CRLF;
  the logger then sends a "Ready: " prompt with no line ending, and a stale prompt
  can precede the next reply, so prompts are stripped wherever they appear.
- Errors reply "E<4 digits> <text>".
- `read data <dataset> <size> <offset>` replies "data <dataset> <size> <offset>\\r\\n",
  then <size> bytes, then a 2-byte CRC (see crc.py).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import serial

from .crc import check_appended

PROMPT_RE = re.compile(rb"Ready: ?")
LINE_RE = re.compile(rb"([^\r\n]*)\r\n")
ERROR_RE = re.compile(r"^E(\d{4})\b\s*(.*)$")
DATA_HDR_RE = re.compile(rb"data (\d+) (\d+) (\d+)\r\n")
DATA_ERR_RE = re.compile(rb"(E\d{4}[^\r\n]*)\r\n")


class LinkError(Exception):
    pass


class LoggerError(LinkError):
    def __init__(self, command: str, code: str, text: str):
        super().__init__(f"{command!r} -> E{code} {text}".rstrip())
        self.command, self.code, self.text = command, code, text


@dataclass
class TranscriptEntry:
    t_ns: int  # host time.time_ns()
    direction: str  # "TX" | "RX" | "NOTE"
    text: str


class Link:
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        # exclusive=True takes a flock on the tty, so a second copy of this tool cannot interleave.
        self.ser = serial.Serial(port, baudrate, timeout=0.02, exclusive=True)
        self.transcript: list[TranscriptEntry] = []
        self._buf = b""

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def note(self, text: str):
        self.transcript.append(TranscriptEntry(time.time_ns(), "NOTE", text))

    def _send(self, cmd: str):
        self.transcript.append(TranscriptEntry(time.time_ns(), "TX", cmd))
        self.ser.write(cmd.encode("ascii") + b"\r")

    def _drain(self, quiet: float = 0.05):
        t_last = time.monotonic()
        while time.monotonic() - t_last < quiet:
            b = self.ser.read(4096)
            if b:
                self._buf += b
                t_last = time.monotonic()
        self._buf = b""

    def wake(self):
        """RBR wake-up: a lone CR, a pause, then discard whatever comes back."""
        self.ser.reset_input_buffer()
        self.ser.write(b"\r")
        time.sleep(0.05)
        self._drain(0.1)

    def command(self, cmd: str, timeout: float = 3.0) -> str:
        """Send a command and return its one-line reply (prompts stripped)."""
        self._send(cmd)
        deadline = time.monotonic() + timeout
        while True:
            self._buf = PROMPT_RE.sub(b"", self._buf)
            m = LINE_RE.search(self._buf)
            while m and not m.group(1).strip():  # skip empty lines
                self._buf = self._buf[m.end():]
                m = LINE_RE.search(self._buf)
            if m:
                line = m.group(1).decode("ascii", errors="replace").strip()
                self._buf = self._buf[m.end():]
                self.transcript.append(TranscriptEntry(time.time_ns(), "RX", line))
                e = ERROR_RE.match(line)
                if e:
                    raise LoggerError(cmd, e.group(1), e.group(2))
                return line
            if time.monotonic() > deadline:
                raise LinkError(f"timeout waiting for reply to {cmd!r} (buffer {self._buf[:80]!r})")
            # Return as soon as anything arrives (read(n) would wait out the timeout for n bytes).
            self._buf += self.ser.read(max(1, self.ser.in_waiting))

    def command_until_prompt(self, cmd: str, timeout: float = 60.0) -> str:
        """For commands with no reply line (e.g. `memclear`): wait for the next "Ready:" prompt.

        Returns any text received before the prompt; raises on an error line.
        """
        self._drain(0.05)
        self._send(cmd)
        deadline = time.monotonic() + timeout
        while True:
            m = PROMPT_RE.search(self._buf)
            if m:
                text = self._buf[:m.start()].decode("ascii", errors="replace").strip()
                self._buf = self._buf[m.end():]
                self.transcript.append(TranscriptEntry(time.time_ns(), "RX", f"{text} <Ready>".strip()))
                e = ERROR_RE.match(text)
                if e:
                    raise LoggerError(cmd, e.group(1), e.group(2))
                return text
            if time.monotonic() > deadline:
                raise LinkError(f"timeout waiting for prompt after {cmd!r}")
            self._buf += self.ser.read(max(1, self.ser.in_waiting))

    def query(self, cmd: str, timeout: float = 3.0) -> dict[str, str]:
        return parse_pairs(self.command(cmd, timeout))

    def read_data(self, dataset: int, size: int, offset: int, timeout: float = 10.0, retries: int = 5) -> bytes:
        """One CRC-checked `read data` block. Retries on CRC failure or timeout."""
        last = None
        for attempt in range(1, retries + 1):
            try:
                return self._read_data_once(dataset, size, offset, timeout)
            except LinkError as err:
                if isinstance(err, LoggerError):
                    raise
                last = err
                self.note(f"read data {dataset} {size} {offset}: attempt {attempt} failed: {err}")
                self._drain(0.3)
        raise LinkError(f"read data {dataset} {size} {offset} failed after {retries} attempts: {last}")

    def _read_data_once(self, dataset: int, size: int, offset: int, timeout: float) -> bytes:
        self._buf = b""
        cmd = f"read data {dataset} {size} {offset}"
        self._send(cmd)
        deadline = time.monotonic() + timeout
        need = None
        while True:
            if need is None:
                m = DATA_HDR_RE.search(self._buf)
                if m:
                    ds, n, off = (int(g) for g in m.groups())
                    if (ds, off) != (dataset, offset) or n > size:
                        raise LinkError(f"unexpected reply header {m.group(0)!r} to {cmd!r}")
                    self._buf = self._buf[m.end():]
                    need = n + 2
                else:
                    e = DATA_ERR_RE.search(self._buf)
                    if e:
                        text = e.group(1).decode("ascii", errors="replace")
                        self.transcript.append(TranscriptEntry(time.time_ns(), "RX", text))
                        em = ERROR_RE.match(text)
                        raise LoggerError(cmd, em.group(1), em.group(2))
            if need is not None and len(self._buf) >= need:
                block, self._buf = self._buf[:need], self._buf[need:]
                ok = check_appended(block)
                note = f"<{need - 2} data bytes + CRC {block[-2:].hex()} {'OK' if ok else 'BAD'}>"
                self.transcript.append(TranscriptEntry(time.time_ns(), "RX", note))
                if not ok:
                    raise LinkError(f"CRC mismatch on {cmd!r}")
                return block[:-2]
            if time.monotonic() > deadline:
                raise LinkError(f"timeout on {cmd!r}: have {len(self._buf)} of {need} bytes")
            want = 65536 if need is None else max(1, need - len(self._buf))
            self._buf += self.ser.read(min(want, 65536))


def parse_pairs(line: str) -> dict[str, str]:
    """'meminfo used = 1, remaining = 2' -> {'used': '1', 'remaining': '2'}.

    The key is the last word left of '=', so command prefixes such as
    'meminfo' or 'channel 1' drop out.
    """
    out = {}
    for part in line.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.split()[-1].lower() if k.split() else ""
        out[k] = v.strip()
    return out
