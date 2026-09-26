"""Terminal and log output shared by the offload threads.

- DEVICE names the logger the current thread is working on ("SN100689@usbmodem101"); every log
  line carries it.
- setup_logging(): the console at INFO, and optionally a session log file at DEBUG that gets every
  step and every serial exchange, with UTC millisecond timestamps. logging handlers lock around each
  write, so any thread may log.
- Console owns the terminal. Only the main thread reads the keyboard: worker threads call ask() and
  wait while the main thread answers in serve(). Output from other threads is held back while a
  question is on screen, so it cannot split the prompt.
- run_in_main() runs a function on the main thread, also inside serve(). NetCDF files are written
  this way. netCDF4 releases the interpreter lock around its HDF5 calls, so two threads could be in
  HDF5 at once. netCDF-C also silences HDF5's error printing only for the thread that loaded it: an
  ordinary file create from a worker thread printed a 20-line "HDF5-DIAG: Error detected" block
  (HDF5 1.14.6, netCDF4 1.7.4, 2026-09-25).
"""

from __future__ import annotations

import contextvars
import logging
import queue
import re
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEVICE: contextvars.ContextVar[str] = contextvars.ContextVar("rbr_device", default="-")

FILE_FORMAT = "%(asctime)s %(levelname)-7s %(threadName)-18s %(device)-22s %(name)s: %(message)s"


class _DeviceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.device = DEVICE.get()
        return True


class UTCFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z"


def _enable_windows_ansi(stream) -> bool:
    """Turn on ANSI escape processing for a Windows console (Windows 10+); False if it cannot, e.g. redirected."""
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        return False


@dataclass
class _Question:
    text: str
    answer: str = ""
    cancelled: bool = False
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Call:
    fn: Callable
    args: tuple
    kwargs: dict
    result: Any = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Console:
    """The terminal: log lines from any thread; questions answered on the main thread."""

    def __init__(self, stream=None, input_fn=input, interactive: bool | None = None, color: bool | None = None):
        self.stream = stream or sys.stdout
        self.input_fn = input_fn
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.color = (hasattr(self.stream, "isatty") and self.stream.isatty()) if color is None else color
        if self.color and sys.platform == "win32":
            self.color = _enable_windows_ansi(self.stream)
        self.stop = threading.Event()  # set on Ctrl-C: pending and later questions get "" (= no)
        self._lock = threading.Lock()
        self._held: list[str] | None = None
        self._jobs: queue.Queue[_Question | _Call] = queue.Queue()

    def write(self, text: str):
        with self._lock:
            if self._held is not None:
                self._held.append(text)
                return
            self.stream.write(text + "\n")
            self.stream.flush()

    def ask(self, question: str) -> str:
        """Ask the operator and return the answer; "" at end of input or once stopping. Any thread."""
        if self.stop.is_set():
            return ""
        q = _Question(question)
        if threading.current_thread() is threading.main_thread():
            self._answer(q)
            return q.answer
        self._jobs.put(q)
        while not q.done.wait(0.2):
            if self.stop.is_set():
                q.cancelled = True
                return ""
        return q.answer

    def run_in_main(self, fn: Callable, *args, **kwargs):
        """Run fn(*args, **kwargs) on the main thread (in serve()) and return its result. Any thread.

        The call runs even after Ctrl-C. A Ctrl-C during the call raises RuntimeError here."""
        if threading.current_thread() is threading.main_thread():
            return fn(*args, **kwargs)
        c = _Call(fn, args, kwargs)
        self._jobs.put(c)
        c.done.wait()
        if c.error is not None:
            raise c.error
        return c.result

    def serve(self, timeout: float) -> bool:
        """Main thread: answer one question or run one call, or wait up to `timeout` s for one."""
        try:
            job = self._jobs.get(timeout=timeout)
        except queue.Empty:
            return False
        if isinstance(job, _Call):
            # A Ctrl-C during the call (a NetCDF write) is held until the file is complete, then acted on.
            pressed: list[bool] = []
            old = signal.signal(signal.SIGINT, lambda *_: pressed.append(True))
            try:
                job.result = job.fn(*job.args, **job.kwargs)
            except Exception as err:
                job.error = err
            finally:
                signal.signal(signal.SIGINT, old)
                job.done.set()
            if pressed:
                raise KeyboardInterrupt
            return True
        if job.cancelled:
            return False
        self._answer(job)
        return True

    def _answer(self, q: _Question):
        with self._lock:
            self._held = []
        try:
            q.answer = self.input_fn(q.text)
        except EOFError:
            q.answer = ""
        finally:  # a Ctrl-C at the prompt still releases the asker (with "") and the held output
            q.done.set()
            with self._lock:
                held, self._held = self._held or [], None
            for line in held:
                self.write(line)


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def printable(text: str) -> str:
    """Escape control characters (e.g. ANSI escapes in a logger's reply) before they reach the terminal."""
    return _CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text)


class ConsoleHandler(logging.Handler):
    """INFO and above to the Console, tagged with the device; records with extra={"banner": True} as alarms."""

    RED, RESET = "\033[1;97;41m", "\033[0m"

    def __init__(self, console: Console):
        super().__init__(logging.INFO)
        self.console = console

    def emit(self, record: logging.LogRecord):
        try:
            msg = printable(record.getMessage())
            dev = printable(getattr(record, "device", "-"))
            prefix = f"[{dev}] " if dev != "-" else ""
            if getattr(record, "banner", False):
                text = f"!! {prefix}{msg} !!"
                if self.console.color:
                    text = f"\a{self.RED}{text}{self.RESET}"
            elif record.levelno >= logging.WARNING:
                text = f"{prefix}{record.levelname}: {msg}"
            else:
                text = prefix + msg
            self.console.write(text)
        except Exception:
            self.handleError(record)


def setup_logging(console: Console, logfile: Path | None = None) -> logging.Logger:
    """Route the package's logging to `console` (INFO) and `logfile` (DEBUG). Replaces earlier handlers."""
    root = logging.getLogger("rbr_tpw")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    ch = ConsoleHandler(console)
    ch.addFilter(_DeviceFilter())
    root.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile, encoding="utf-8")  # flushed after every record
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(UTCFormatter(FILE_FORMAT))
        fh.addFilter(_DeviceFilter())
        root.addHandler(fh)
    return root
