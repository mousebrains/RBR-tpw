"""The write lock on RBR L2-family loggers (`lock`, `lock OFF = <key>`, `lock on`), and what it hides.

Ruskin unlocks every logger it connects to, right after reading the clock. Locked, a logger hides part of its
channel list: RBRconcerto SN060275 (fwtype 103, fw 1.460, bench 2026-09-26) lists 8 channels with
`status = on|off` while locked and 9 with the numeric channelStatus bitfield (0, 4, 9) when unlocked; the
ninth is the hidden pressure-compensation thermistor, which IS stored in every sample set. RBRconcerto3
SN233442 (fwtype 104) likewise lists 6 channels locked and 8 unlocked (its two hidden channels are not
stored in EasyParse, but they would be in rawbin00, and the record should show the whole table).
"""

from __future__ import annotations

import datetime as dt
import struct
from contextlib import contextmanager

from .crc import crc16_ccitt
from .link import Link, LinkError

E2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)


class UnlockError(LinkError):
    """The logger did not accept `lock OFF = <key>`."""


def unlock_key(serial: int, logger_seconds: int) -> int:
    """Write-unlock key for `lock OFF = <key>` on L2-family loggers.

    Challenge-response on the logger's most recently reported `now` (seconds
    since 2000-01-01) and its serial number. Reproduces Ruskin 2.26.1
    PhysicalL2.computeKey; verified against three keys Ruskin sent SN100689
    on 2026-09-25 and one it sent SN060275 on 2026-09-10 (little-endian int32
    bytes, CRC-16/CCITT-FALSE).
    """
    c_sn = crc16_ccitt(struct.pack("<i", serial))
    c_t = crc16_ccitt(struct.pack("<i", logger_seconds))
    hi = (((serial >> 16) & 0xFFFF) ^ c_t) & 0xFFFF
    lo = ((logger_seconds & 0xFFFF) ^ c_sn) & 0xFFFF
    return (hi << 16) | lo


def logger_seconds(now_reply: str) -> int:
    """`now = YYYYMMDDhhmmss` (or just the value) -> seconds since 2000-01-01."""
    value = now_reply.split("=", 1)[-1].strip()
    t = dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC)
    return int((t - E2000).total_seconds())


CLOCK_COMMANDS = {"now": "now", "clock": "datetime"}  # command -> key of the datetime in its reply


class Session:
    """Unlocked write session; always re-locks on exit.

    `clock_cmd` names the clock read that sets the challenge: `now` on L2 loggers, `clock` on Gen3 (fwtype 104).
    In Ruskin 2.26.1's serial logs every key sent to fwtypes 0, 102 and 103 (246 keys), 156 of 157 sent to
    fwtype 9 and 30 of 31 sent to fwtype 104 follow unlock_key(serial, seconds of the last such reply)."""

    def __init__(self, link: Link, serial: int, clock_cmd: str = "now"):
        self.link = link
        self.serial = serial
        self.clock_cmd = clock_cmd

    def unlock(self):
        stamp = self.link.query(self.clock_cmd)[CLOCK_COMMANDS[self.clock_cmd]]  # sets the logger's challenge
        r = self.link.command(f"lock OFF = {unlock_key(self.serial, logger_seconds(stamp))}")
        if "off" not in r.lower():
            raise UnlockError(f"unlock failed: {r!r}")

    def __enter__(self):
        self.unlock()
        return self

    def __exit__(self, *exc):
        try:
            self.link.command("lock on")
        except Exception as err:
            self.link.note(f"lock on failed: {err}")

    def write(self, cmd: str, expect: str | None = None, timeout: float = 3.0) -> str:
        """Send a setting; the logger echoes it back. `expect` is a substring the echo must contain."""
        self.unlock()
        r = self.link.command(cmd, timeout)
        want = (expect if expect is not None else cmd).replace(" ", "").lower()
        if want not in r.replace(" ", "").lower():
            from .configure import ConfigError  # here, not at the top: configure imports this module
            raise ConfigError(f"{cmd!r}: unexpected reply {r!r}")
        return r


@contextmanager
def unlocked_if_possible(link: Link, serial: int | None, clock_cmd: str = "now"):
    """Unlocked for the block if `serial` is given and the logger accepts the key; otherwise the block runs
    locked, with a transcript note. Yields True if unlocked. The lock is restored afterwards."""
    if serial is None:
        yield False
        return
    session = Session(link, serial, clock_cmd)
    try:
        session.unlock()
    except LinkError as err:  # no `lock` command (older firmware?), wrong key, or a dropped reply
        link.note(f"could not unlock to read the full channel list ({err}); reading it locked")
        yield False
        return
    try:
        yield True
    finally:
        session.__exit__(None, None, None)
