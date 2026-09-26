"""Host clock: offset from an NTP server via macOS/BSD `sntp` (query only; never sets the clock), and a
lock for the steps that depend on host timestamps to the millisecond."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)
_TIMING_LOCK = threading.Lock()
_quiet = threading.Condition()  # transfers vs timing-critical steps
_transfers = 0
_timing = False


@contextmanager
def transfer() -> Iterator[None]:
    """Wrap one bulk block transfer: it waits while a timing-critical step runs on another logger."""
    global _transfers
    with _quiet:
        while _timing:
            _quiet.wait()
        _transfers += 1
    try:
        yield
    finally:
        with _quiet:
            _transfers -= 1
            _quiet.notify_all()


@contextmanager
def timing_critical(what: str = "") -> Iterator[None]:
    """Run one host-timestamp-sensitive step (clock-skew measurement, clock set) at a time, with the
    other loggers' block transfers paused (in-flight blocks finish first), so USB traffic from parallel
    downloads cannot widen the timing brackets. While the step runs the interpreter switches threads
    every 0.5 ms instead of every 5 ms."""
    global _timing
    t0 = time.monotonic()
    with _TIMING_LOCK:
        with _quiet:
            _timing = True
            while _transfers:
                _quiet.wait()
        waited = time.monotonic() - t0
        if waited > 0.05:
            log.debug("waited %.1f s for the timing lock and in-flight transfers (%s)", waited, what)
        old = sys.getswitchinterval()
        sys.setswitchinterval(min(old, 0.0005))
        try:
            yield
        finally:
            sys.setswitchinterval(old)
            with _quiet:
                _timing = False
                _quiet.notify_all()


# sntp prints e.g. "+0.000233 +/- 0.024659 time.apple.com 17.253.16.125".
# Offset follows the NTP convention (server minus host), so UTC ~= host + offset.
_LINE = re.compile(r"^([+-]?\d+\.\d+)\s+\+/-\s+(\d+\.\d+)\s+(\S+)", re.M)


def ntp_offset(server: str = "time.apple.com", timeout: float = 3.0) -> dict:
    exe = shutil.which("sntp")
    if not exe:
        return {"server": server, "error": "sntp not found"}
    try:
        p = subprocess.run([exe, "-t", str(timeout), server], capture_output=True, text=True,
                           timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return {"server": server, "error": "sntp timed out"}
    m = _LINE.search(p.stdout)
    if not m:
        return {"server": server, "error": (p.stdout + p.stderr).strip()[-200:] or f"exit {p.returncode}"}
    return {"server": m.group(3), "offset_s": float(m.group(1)), "uncertainty_s": float(m.group(2))}
