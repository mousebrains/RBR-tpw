"""Fixes from an adversarial review (2026-09-25): corrupt events, non-finite temperatures, port discovery,
serial validation, terminal escapes, --rebuild paths, NTP reuse, and transfers pausing for clock timing."""

import json
import shutil
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fakelogger import solo_image

from rbr_tpw import cli, hostclock
from rbr_tpw.console import printable
from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.link import LinkError
from rbr_tpw.rawbin import decode, tmp_equation
from rbr_tpw.solo import identify


def test_corrupt_event_record_is_dropped_whole():
    image = bytearray(solo_image(10))
    body = bytes([0x01, 0xF7]) + struct.pack("<I", 843_696_100)
    bad = struct.pack(">H", crc16_ccitt(body) ^ 0x5555) + body  # an event whose CRC no longer matches
    k = len(image) - 4 * 5  # before the last five readings
    image[k:k] = bad
    d = decode(bytes(image), 1)
    assert d.raw.shape == (10, 1) and d.bad_event_words == 1  # not 12: no phantom readings
    assert np.all(np.diff(d.time_ms) == 500)


def test_non_finite_temperature_is_flagged():
    t, bad = tmp_equation(np.array([1 << 29], np.uint32), (0.0, 0.0, 0.0, 0.0))  # 1/0
    assert bad[0] and np.isnan(t[0])


def _port(device, manufacturer="RBR"):
    return SimpleNamespace(device=device, manufacturer=manufacturer, product="")


def test_port_discovery_on_linux_and_macos(monkeypatch):
    ports = [_port("/dev/ttyACM0"), _port("/dev/cu.usbmodem101"), _port("/dev/tty.usbmodem101"),
             _port("/dev/ttyACM1", "Arduino")]
    monkeypatch.setattr(cli.list_ports, "comports", lambda: ports)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    assert cli.rbr_ports() == {"/dev/ttyACM0", "/dev/cu.usbmodem101", "/dev/tty.usbmodem101"}
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    assert cli.rbr_ports() == {"/dev/cu.usbmodem101"}


def test_explicit_port_is_used_as_given(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "rbr_ports", lambda: set())
    dev = tmp_path / "ttyACM7"
    assert cli._present(str(dev)) == set()
    dev.touch()
    assert cli._present(str(dev)) == {str(dev)}


class _IdLink:
    def __init__(self, reply):
        self.reply = reply

    def wake(self):
        pass

    def query(self, cmd):
        from rbr_tpw.link import parse_pairs
        return parse_pairs(self.reply)


def test_serial_number_must_be_safe_for_file_names():
    assert identify(_IdLink("id model = RBRsolo, version = 1.000, serial = 076313, fwtype = 0"))["serial"] == "076313"
    for bad in ("../../tmp/x", "", "a/b", "x" * 40):
        with pytest.raises(LinkError, match="implausible serial"):
            identify(_IdLink(f"id model = RBRsolo, version = 1.000, serial = {bad}, fwtype = 9"))


def test_control_characters_are_escaped_for_the_terminal():
    assert printable("RBR\x1b[2Jsolo\x07") == "RBR\\x1b[2Jsolo\\x07" and printable("a\tb\nc") == "a\tb\nc"


def test_rebuild_finds_raw_files_beside_a_moved_record(tmp_path):
    from rbr_tpw.solo import sha256
    image = solo_image(50)
    record = {"id": {"model": "RBRsolo", "version": "1.000", "serial": "100689", "fwtype": 9},
              "offload_started": "2026-09-26T00:00:00.000Z",
              "snapshot_before": {"status": "logging", "sampling": {"mode": "continuous", "period": "500"},
                                  "channel_list": [{"type": "temp02", "equation": "tmp", "coefficients": {
                                      "c0": 0.00347, "c1": -2.52e-4, "c2": 2.39e-6, "c3": -8.07e-8}}]},
              "raw": {"file": "raw/x.bin", "bytes": len(image), "sha256": sha256(image)}, "warnings": []}
    moved = tmp_path / "somewhere"
    moved.mkdir()
    (moved / "x.json").write_text(json.dumps(record))
    (moved / "x.bin").write_bytes(image)
    cli.main([str(tmp_path / "out"), "--rebuild", str(moved / "x.json")])
    assert (tmp_path / "out" / "x.nc").exists()
    shutil.rmtree(moved)


def test_ntp_is_reused_between_loggers(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "ntp_offset", lambda server: calls.append(server) or {"offset_s": 0.001})
    cli._ntp_cache.clear()
    a, b = cli._ntp("time.example"), cli._ntp("time.example")
    assert calls == ["time.example"] and a["offset_s"] == b["offset_s"] == 0.001 and "measured_at" in a


def test_transfers_pause_while_a_clock_is_timed():
    events = []

    def downloader():
        with hostclock.transfer():  # in flight when the timing step starts: it must finish first
            events.append(("block1 start", time.monotonic()))
            time.sleep(0.2)
            events.append(("block1 end", time.monotonic()))
        time.sleep(0.05)
        with hostclock.transfer():  # requested during the timing step: must wait for it
            events.append(("block2 start", time.monotonic()))

    t = threading.Thread(target=downloader)
    t.start()
    time.sleep(0.05)
    with hostclock.timing_critical("test"):
        events.append(("timing start", time.monotonic()))
        time.sleep(0.3)
        events.append(("timing end", time.monotonic()))
    t.join(5)
    order = [name for name, _ in sorted(events, key=lambda e: e[1])]
    assert order == ["block1 start", "block1 end", "timing start", "timing end", "block2 start"]
