"""Serial command link to an RBR logger over its USB CDC port.

Protocol notes (RBRsolo fwtype 9, observed 2026-09-25 and in Ruskin 2.26.1 serial logs):
- Commands are ASCII terminated by CR. Replies are one line terminated by CRLF;
  the logger then sends a "Ready: " prompt with no line ending, and a stale prompt
  can precede the next reply, so prompts are stripped wherever they appear.
- Errors reply "E<4 digits> <text>".
- `read data <dataset> <size> <offset>` replies "data <dataset> <size> <offset>\\r\\n",
  then <size> bytes, then a 2-byte CRC (see crc.py).
- Gen3 (L3, fwtype 104) instead takes `readdata size = <s>, offset = <o>, dataset = <d>` and replies
  "readdata dataset = <d>, size = <s>, offset = <o>\\r\\n", then the bytes and the CRC (L3 ref 4.6.4;
  the form Ruskin 2.26.1 sends to RBRconcerto3 SN233442).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import serial

from .crc import check_appended

serial_log = logging.getLogger("rbr_tpw.serial")  # every exchange, at DEBUG, into the session log

PROMPT_RE = re.compile(rb"Ready: ?")
LINE_RE = re.compile(rb"([^\r\n]*)\r\n")
ERROR_RE = re.compile(r"^E(\d{4})\b\s*(.*)$")
DATA_HDR_RE = re.compile(rb"data (\d+) (\d+) (\d+)\r\n")
L3_DATA_HDR_RE = re.compile(rb"readdata dataset = (\d+), size = (\d+), offset = (\d+)\r\n")
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
    direction: str  # "TX" | "RX" | "NOTE" | "DROP" (bytes received and discarded)
    text: str

    def line(self) -> str:
        t = dt.datetime.fromtimestamp(self.t_ns / 1e9, dt.UTC).isoformat(timespec="milliseconds")
        return f"{t.replace('+00:00', 'Z')}  {self.direction:4s}  {self.text}\n"


def _show(data: bytes, limit: int = 120) -> str:
    return f"{len(data)} bytes {data[:limit]!r}{'...' if len(data) > limit else ''}"


def open_serial(port: str, baudrate: int):
    """The logger's port: a device name (/dev/cu.usbmodem101, /dev/ttyACM0, COM3) or a pyserial URL such as
    socket://host:port (tests; a remote logger). exclusive=True takes a flock on POSIX ttys, so a second copy
    of this tool cannot interleave; Windows COM ports are always exclusive."""
    return serial.serial_for_url(port, baudrate=baudrate, timeout=0.02, exclusive=True)


class Link:
    """One logger's serial port.

    The transcript file is written as it happens (line-buffered) once start_transcript() names it, normally
    as soon as the logger reports its serial number; entries before that are buffered, and every entry also
    goes to the session log. If the transcript was never started, close() writes the buffer to
    `fallback_transcript` (e.g. named by port), so a failure before `id` still leaves one."""

    def __init__(self, port: str, baudrate: int = 115200, fallback_transcript: Path | None = None):
        self.port = port
        self.ser = open_serial(port, baudrate)
        self.transcript: list[TranscriptEntry] = []
        self.transcript_path: Path | None = None
        self.fallback_transcript = fallback_transcript
        self._tfile = None
        self._buf = b""
        self.note(f"opened {port}, {baudrate} baud")

    def start_transcript(self, path: Path):
        """Open the transcript file, write what has happened so far, and keep appending as it happens."""
        if self._tfile is not None:
            return
        self._tfile = open(path, "a", encoding="utf-8", buffering=1)
        self.transcript_path = path
        self._tfile.writelines(e.line() for e in self.transcript)

    def close(self):
        try:
            self.note("closing port")
            self.ser.close()
        except Exception:
            pass
        if self._tfile is None and self.fallback_transcript is not None:
            try:
                self.start_transcript(self.fallback_transcript)
            except OSError:
                pass
        if self._tfile is not None:
            self._tfile.close()
            self._tfile = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _record(self, direction: str, text: str):
        e = TranscriptEntry(time.time_ns(), direction, text)
        self.transcript.append(e)
        if self._tfile is not None:
            self._tfile.write(e.line())
        serial_log.debug("%-4s %s", direction, text)

    def note(self, text: str):
        self._record("NOTE", text)

    def _send(self, cmd: str):
        self._record("TX", cmd)
        self.ser.write(cmd.encode("ascii") + b"\r")

    def _drain(self, quiet: float = 0.05):
        """Read until the line is quiet for `quiet` s and discard everything buffered (recorded as DROP)."""
        got = self._buf
        t_last = time.monotonic()
        while time.monotonic() - t_last < quiet:
            b = self.ser.read(4096)
            if b:
                got += b
                t_last = time.monotonic()
        self._buf = b""
        if got:
            self._record("DROP", _show(got))

    def wake(self):
        """RBR wake-up: a lone CR, a pause, then discard whatever comes back."""
        self.ser.reset_input_buffer()
        self._record("TX", "<CR> (wake-up; input buffer reset first)")
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
                self._record("RX", line)
                e = ERROR_RE.match(line)
                if e:
                    raise LoggerError(cmd, e.group(1), e.group(2))
                return line
            if time.monotonic() > deadline:
                self.note(f"timeout after {timeout:g} s waiting for a reply to {cmd!r}; unparsed {_show(self._buf)}")
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
                self._record("RX", f"{text} <Ready>".strip())
                e = ERROR_RE.match(text)
                if e:
                    raise LoggerError(cmd, e.group(1), e.group(2))
                return text
            if time.monotonic() > deadline:
                self.note(f"timeout after {timeout:g} s waiting for the prompt after {cmd!r}; "
                          f"unparsed {_show(self._buf)}")
                raise LinkError(f"timeout waiting for prompt after {cmd!r}")
            self._buf += self.ser.read(max(1, self.ser.in_waiting))

    def query(self, cmd: str, timeout: float = 3.0) -> dict[str, str]:
        return parse_pairs(self.command(cmd, timeout))

    def read_data(self, dataset: int, size: int, offset: int, timeout: float = 10.0, retries: int = 5,
                  l3: bool = False) -> bytes:
        """One CRC-checked block of a dataset (`read data`, or Gen3 `readdata` if l3). Retries on CRC failure
        or timeout."""
        last = None
        for attempt in range(1, retries + 1):
            try:
                return self._read_data_once(dataset, size, offset, timeout, l3)
            except LinkError as err:
                if isinstance(err, LoggerError):
                    raise
                last = err
                self.note(f"dataset {dataset} {size} bytes at {offset}: attempt {attempt} failed: {err}")
                self._drain(0.3)
        raise LinkError(f"dataset {dataset} {size} bytes at {offset} failed after {retries} attempts: {last}")

    def _read_data_once(self, dataset: int, size: int, offset: int, timeout: float, l3: bool = False) -> bytes:
        leftover = PROMPT_RE.sub(b"", self._buf)
        if leftover.strip():
            self._record("DROP", _show(leftover))
        self._buf = b""
        if l3:
            cmd, header_re = f"readdata size = {size}, offset = {offset}, dataset = {dataset}", L3_DATA_HDR_RE
        else:
            cmd, header_re = f"read data {dataset} {size} {offset}", DATA_HDR_RE
        self._send(cmd)
        deadline = time.monotonic() + timeout
        need = None
        while True:
            if need is None:
                m = header_re.search(self._buf)
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
                        self._record("RX", text)
                        em = ERROR_RE.match(text)
                        raise LoggerError(cmd, em.group(1), em.group(2))
            if need is not None and len(self._buf) >= need:
                block, self._buf = self._buf[:need], self._buf[need:]
                ok = check_appended(block)
                note = f"<{need - 2} data bytes + CRC {block[-2:].hex()} {'OK' if ok else 'BAD'}>"
                self._record("RX", note)
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
