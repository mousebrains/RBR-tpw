"""Host clock: its offset from an NTP server (an SNTP query from Python; never sets the clock), and a lock
for the steps that depend on host timestamps to the millisecond."""

from __future__ import annotations

import logging
import socket
import struct
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


NTP_TO_UNIX = 2_208_988_800  # seconds from 1900-01-01 (NTP era 0) to 1970-01-01


def _from_ntp(b: bytes) -> float:
    sec, frac = struct.unpack("!II", b)
    return sec - NTP_TO_UNIX + frac / 2**32


def _to_ntp(t: float) -> bytes:
    return struct.pack("!II", (int(t) + NTP_TO_UNIX) & 0xFFFFFFFF, int((t % 1) * 2**32) & 0xFFFFFFFF)


def ntp_query(server: str, timeout: float = 1.0, port: int = 123) -> dict:
    """One SNTP exchange (RFC 4330). offset = ((T2 - T1) + (T3 - T4)) / 2 is server minus host, so
    UTC ~= host + offset; delay = (T4 - T1) - (T3 - T2). Raises OSError or ValueError."""
    family, _, _, _, addr = socket.getaddrinfo(server, port, type=socket.SOCK_DGRAM)[0]
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        request = bytearray(48)
        request[0] = 0x23  # leap indicator 0, version 4, mode 3 (client)
        t1 = time.time()
        request[40:48] = _to_ntp(t1)  # the server echoes this as the reply's originate timestamp
        sock.sendto(bytes(request), addr)
        while True:
            data, _ = sock.recvfrom(512)
            t4 = time.time()
            if len(data) >= 48 and data[24:32] == request[40:48]:
                break  # the reply to this request, not a stray packet
    leap, mode, stratum = data[0] >> 6, data[0] & 7, data[1]
    if mode not in (4, 5) or not 1 <= stratum <= 15 or leap == 3:
        raise ValueError(f"unusable NTP reply (mode {mode}, stratum {stratum}, leap indicator {leap})")
    t2, t3 = _from_ntp(data[32:40]), _from_ntp(data[40:48])
    return {"address": addr[0], "offset_s": ((t2 - t1) + (t3 - t4)) / 2, "delay_s": (t4 - t1) - (t3 - t2),
            "root_delay_s": struct.unpack("!i", data[4:8])[0] / 2**16,
            "root_dispersion_s": struct.unpack("!I", data[8:12])[0] / 2**16, "stratum": stratum}


def ntp_offset(server: str = "time.apple.com", timeout: float = 3.0, samples: int = 4, port: int = 123) -> dict:
    """Host clock offset from `server` (UTC ~= host + offset_s), by SNTP from Python: no `sntp` program needed,
    so the same on macOS, Linux and Windows. The lowest-delay reply of `samples` queries is used;
    uncertainty = delay/2 + root delay/2 + root dispersion. Never sets the host clock."""
    results, errors = [], []
    deadline = time.monotonic() + timeout
    for _ in range(samples):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            results.append(ntp_query(server, min(1.0, remaining), port))
        except socket.gaierror as err:  # no DNS (e.g. offline at sea): more tries will not help
            errors.append(f"cannot resolve {server}: {err}")
            break
        except (OSError, ValueError) as err:
            errors.append(str(err) or type(err).__name__)
    if not results:
        return {"server": server, "error": errors[-1] if errors else "no reply"}
    best = min(results, key=lambda r: r["delay_s"])
    return {"server": server, "address": best["address"], "offset_s": best["offset_s"],
            "uncertainty_s": best["delay_s"] / 2 + best["root_delay_s"] / 2 + best["root_dispersion_s"],
            "delay_s": best["delay_s"], "stratum": best["stratum"], "n": len(results),
            "method": "SNTP (RFC 4330) from Python; lowest-delay of n replies; uncertainty = delay/2 + "
                      "root delay/2 + root dispersion"}
