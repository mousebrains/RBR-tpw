"""Host clock offset from an NTP server, via macOS/BSD `sntp` (query only; never sets the clock)."""

from __future__ import annotations

import re
import shutil
import subprocess

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
