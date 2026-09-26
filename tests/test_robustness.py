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


def _port(device, manufacturer="RBR", vid=None, pid=None):
    return SimpleNamespace(device=device, manufacturer=manufacturer, product="", vid=vid, pid=pid,
                           serial_number=None, description="", location=None, hwid="")


def test_port_discovery_on_each_os(monkeypatch):
    ports = [_port("/dev/ttyACM0"), _port("/dev/cu.usbmodem101"), _port("/dev/tty.usbmodem101"),
             _port("/dev/ttyACM1", "Arduino"),
             _port("COM3", "Microsoft", vid=0x0451, pid=0xBEF1),  # Windows' own driver hides the maker's name
             _port("COM4", "Microsoft", vid=0x0451, pid=0xF432)]  # a TI device that is not an RBR logger
    monkeypatch.setattr(cli.list_ports, "comports", lambda: ports)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    assert cli.rbr_ports() == {"/dev/ttyACM0", "/dev/cu.usbmodem101", "/dev/tty.usbmodem101", "COM3"}
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    assert cli.rbr_ports() == {"/dev/cu.usbmodem101"}
    monkeypatch.setattr(cli.platform, "system", lambda: "Windows")
    assert "COM3" in cli.rbr_ports() and "COM4" not in cli.rbr_ports()
    assert cli._present("com3") == {"com3"} and cli._present("COM9") == set()
    assert cli.port_info("COM3")["vid"] == "0x0451" and cli.port_info("COM3")["pid"] == "0xBEF1"


def test_explicit_port_is_used_as_given(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "rbr_ports", lambda: set())
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    dev = tmp_path / "ttyACM7"
    assert cli._present(str(dev)) == set()
    dev.touch()
    assert cli._present(str(dev)) == {str(dev)}
    assert cli._present("socket://127.0.0.1:5000") == {"socket://127.0.0.1:5000"}


def test_port_names_are_safe_in_file_names():
    assert cli._port_name("/dev/cu.usbmodem101") == "usbmodem101" and cli._port_name("COM3") == "COM3"
    assert cli._port_name("socket://127.0.0.1:5000") == "127.0.0.1_5000"
    assert cli._port_name("\\\\.\\COM12") == "COM12"


def test_missing_process_tools_do_not_crash(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("pgrep")
    monkeypatch.setattr(cli.subprocess, "run", missing)
    assert cli.ruskin_running() is False


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
    in_flight = threading.Event()

    def downloader():
        with hostclock.transfer():  # in flight when the timing step starts: it must finish first
            events.append(("block1 start", time.monotonic()))
            in_flight.set()
            time.sleep(0.2)
            events.append(("block1 end", time.monotonic()))
        time.sleep(0.05)
        with hostclock.transfer():  # requested during the timing step: must wait for it
            events.append(("block2 start", time.monotonic()))

    t = threading.Thread(target=downloader)
    t.start()
    assert in_flight.wait(5)
    with hostclock.timing_critical("test"):
        events.append(("timing start", time.monotonic()))
        time.sleep(0.3)
        events.append(("timing end", time.monotonic()))
    t.join(5)
    order = [name for name, _ in sorted(events, key=lambda e: e[1])]
    assert order == ["block1 start", "block1 end", "timing start", "timing end", "block2 start"]


def _fake_ntp_server(offset_s):
    """A local SNTP server whose clock is host + offset_s; returns (port, stop)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.1)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(512)
            except OSError:
                continue
            t2 = time.time() + offset_s
            reply = bytearray(48)
            reply[0], reply[1] = 0x24, 1  # version 4, mode 4 (server); stratum 1
            reply[24:32] = data[40:48]  # originate = the client's transmit timestamp
            reply[32:40] = hostclock._to_ntp(t2)
            reply[40:48] = hostclock._to_ntp(time.time() + offset_s)
            sock.sendto(bytes(reply), addr)

    threading.Thread(target=serve, daemon=True).start()
    return sock.getsockname()[1], stop


def test_sntp_recovers_a_known_offset():
    port, stop = _fake_ntp_server(0.250)
    try:
        r = hostclock.ntp_offset("127.0.0.1", port=port)
    finally:
        stop.set()
    assert r["offset_s"] == pytest.approx(0.250, abs=0.005) and r["n"] == 4 and r["stratum"] == 1
    assert 0 <= r["uncertainty_s"] < 0.005


def test_sntp_offline_fails_fast():
    t0 = time.monotonic()
    r = hostclock.ntp_offset("no-such-host.invalid")
    assert "cannot resolve" in r["error"] and time.monotonic() - t0 < 3.5


def test_host_clock_resolution_is_sub_millisecond():
    """Skew and clock setting rely on it (Python >= 3.13 on Windows: GetSystemTimePreciseAsFileTime)."""
    deltas = []
    last = time.time_ns()
    while len(deltas) < 2000:
        now = time.time_ns()
        if now != last:
            deltas.append(now - last)
            last = now
    assert min(deltas) < 1_000_000
