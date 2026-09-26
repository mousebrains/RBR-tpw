"""Fixes from a second external review (2026-09-26): timing-flag safety, stale replies and echoes, equation
cleanup and infinities, reset clocks on RBRsolos, depth attributes, --once grace, --rebuild guards, the
NOT READY exit status, and Windows sleep resolution."""

import io
import json
import math
import statistics
import struct
import sys
import threading
import time

import netCDF4
import numpy as np
import pytest
from fakelogger import FakeSolo, l2_sectioned_image, solo_image

from rbr_tpw import cli, hostclock
from rbr_tpw import link as link_module
from rbr_tpw.console import Console, setup_logging
from rbr_tpw.equations import evaluate
from rbr_tpw.link import Link
from rbr_tpw.ncwrite import write_netcdf
from rbr_tpw.rawbin import EPOCH2000_MS


def test_timing_flag_is_lowered_even_if_the_step_fails_early(monkeypatch):
    calls = {"n": 0}
    real = sys.setswitchinterval

    def flaky(interval):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure right after the flag is raised")
        real(interval)

    monkeypatch.setattr(hostclock.sys, "setswitchinterval", flaky)
    with pytest.raises(RuntimeError):
        with hostclock.timing_critical("test"):
            pass
    assert hostclock._timing is False
    done = threading.Event()

    def download():
        with hostclock.transfer():
            done.set()

    threading.Thread(target=download, daemon=True).start()
    assert done.wait(2)  # transfers are not stuck


@pytest.fixture
def fake_link(monkeypatch):
    def connect(fake):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
        return Link(fake.port)
    return connect


def test_a_late_reply_is_not_taken_for_the_next_one(fake_link):
    fake = FakeSolo("/dev/cu.fake")
    link = fake_link(fake)
    fake._out += b"sampling mode = continuous, period = 500\r\nReady: "  # arrived after a timeout
    assert link.command("id").startswith("id model = RBRsolo")
    assert any(e.direction == "DROP" and "sampling" in e.text for e in link.transcript)


class EchoingSolo(FakeSolo):
    """Echoes each command line first, with bare-LF line endings (e.g. behind a terminal server)."""

    def _handle(self, cmd):
        if cmd:
            with self._lock:
                self._out += cmd.encode() + b"\n"
        super()._handle(cmd)

    def _reply(self, text):
        with self._lock:
            self._out += text.encode("ascii") + b"\nReady: "


def test_echo_and_bare_line_feeds(fake_link):
    link = fake_link(EchoingSolo("/dev/cu.fake"))
    assert link.command("status") == "status = logging"
    assert link.query("meminfo")["used"].isdigit()


def test_a_failed_channel_is_not_reported_as_a_cycle_and_infinities_become_nan():
    tmp = {"c0": 3.4740720e-003, "c1": -252.09197e-006, "c2": 2.4784860e-006, "c3": -85.943688e-009}
    channels = [
        {"index": 1, "type": "pres21", "equation": "corr_pres2", "status": 0,
         "coefficients": {"c0": 0.0, "c1": 100.0, "c2": 0.0, "c3": 0.0, "n0": 2}},  # x's missing: fails
        {"index": 2, "type": "temp12", "equation": "tmp", "status": 0, "coefficients": tmp},
        {"index": 3, "type": "pres24", "equation": "corr_pres2", "status": 0,  # 1 + x4*dT = 0 at dT = -1
         "coefficients": {"c0": 0.0, "c1": 100.0, "c2": 0.0, "c3": 0.0, "x0": 10.0, "x1": 0.0, "x2": 0.0,
                          "x3": 0.0, "x4": 1.0, "x5": 0.0, "n0": "value"}},
    ]
    raw = np.array([[1 << 29, 1 << 29, 1 << 29]], np.uint32)
    values, bad, problems = evaluate(raw, channels, defaults={"temperature": -1.0})
    assert "missing coefficient" in problems[0] and "circular" not in problems[0]
    assert np.isfinite(values[0, 1]) and 1 not in problems  # the temperature channel is unaffected
    assert math.isnan(values[0, 2]) and bad[0, 2]  # a zero denominator gives NaN, not inf


def test_solo_enabled_on_a_reset_clock_is_retimed(tmp_path):
    """Enabled after its clock restarted at 2000-01-01: enable time and samples are all in 2000."""
    image = bytearray(solo_image(20, sync_s=100))
    struct.pack_into("<II", image, 12, 90, 90)  # enable and start times on the reset clock too
    true_first = 1_790_000_000_000
    skew = (EPOCH2000_MS + 100_000 - true_first) / 1000  # logger minus UTC
    record = {"offload_started": "2026-09-26T00:00:00.000Z", "offload_finished": "2026-09-26T00:00:00.000Z",
              "id": {"model": "RBRsolo", "version": "1.000", "serial": "100689", "fwtype": 9},
              "clock_skew": {"n": 3, "skew_vs_host_s": skew, "uncertainty_s": 0.003, "spread_s": 0.001,
                             "measured_at": "2026-09-26T00:00:00.000Z"},
              "host_ntp": {"offset_s": 0.0, "uncertainty_s": 0.01},
              "snapshot_before": {"status": "logging", "sampling": {"mode": "continuous", "period": "500"},
                                  "channel_list": [{"type": "temp02", "equation": "tmp", "coefficients": {
                                      "c0": 3.4740720e-003, "c1": -252.09197e-006, "c2": 2.4784860e-006,
                                      "c3": -85.943688e-009}}]},
              "raw": {"file": "raw/x.bin", "bytes": len(image), "sha256": ""}, "warnings": []}
    _, _, t = write_netcdf(bytes(image), record, tmp_path / "x.nc")
    assert t[0] == true_first and len(t) == 20
    with netCDF4.Dataset(tmp_path / "x.nc") as ds:
        assert set(ds["time_flag"][:]) == {6} and ds.clock_reset_detected == "yes"


def test_depth_gets_positive_down_on_the_decoded_path(tmp_path):
    coeffs = [("c0", 0.0), ("c1", 100.0)]
    image = l2_sectioned_image(10, [("dpth01", 0, coeffs)])
    ch = {"index": 1, "type": "dpth01", "equation": "lin", "status": 0, "coefficients": dict(coeffs)}
    record = {"offload_started": "2026-09-26T00:00:00.000Z",
              "id": {"model": "RBRduet", "version": "3.220", "serial": "081500", "fwtype": 102},
              "snapshot_before": {"status": "logging", "sampling": {"mode": "continuous", "period": "500"},
                                  "channel_list": [ch], "channels_all": [ch]}, "warnings": []}
    write_netcdf(image, record, tmp_path / "d.nc")
    with netCDF4.Dataset(tmp_path / "d.nc") as ds:
        assert ds["depth"].positive == "down" and ds["depth"].standard_name == "depth"


def test_once_waits_for_a_logger_that_appears_a_moment_later(tmp_path, monkeypatch):
    fakes = {"/dev/cu.A": FakeSolo("/dev/cu.A", serial=11, n_samples=50),
             "/dev/cu.B": FakeSolo("/dev/cu.B", serial=22, n_samples=50)}
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fakes[port])
    monkeypatch.setattr(cli, "ruskin_running", lambda: False)
    monkeypatch.setattr(cli, "SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "POLL_S", 0.05)
    monkeypatch.setattr(cli, "ONCE_GRACE_S", 1.0)
    monkeypatch.setattr(cli, "measure_clock_skew", lambda link, **k: {
        "n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "measured_at": ""})
    # B enumerates only after A is done, as on a hub that is slow to present a second logger
    monkeypatch.setattr(cli, "rbr_ports", lambda: {"/dev/cu.A"} | (
        {"/dev/cu.B"} if list(tmp_path.glob("11_*.nc")) else set()))
    console = Console(stream=io.StringIO(), interactive=False, color=False)
    setup_logging(console, tmp_path / "s.log")
    cli.run(cli.Settings(outdir=tmp_path, ntp_server=None, console=console), once=True, port=None)
    assert list(tmp_path.glob("11_*.nc")) and list(tmp_path.glob("22_*.nc"))
    setup_logging(Console(stream=io.StringIO()))


def test_rebuild_skips_empty_records_and_refuses_paths_outside(tmp_path):
    empty = tmp_path / "raw" / "empty.json"
    empty.parent.mkdir()
    empty.write_text(json.dumps({"id": {"fwtype": 9}, "raw": None, "datasets": {}}))
    cli.main([str(tmp_path / "out"), "--rebuild", str(empty)])  # no crash, nothing written
    assert not list((tmp_path / "out").glob("*.nc"))
    evil = tmp_path / "raw" / "evil.json"
    evil.write_text(json.dumps({"id": {"fwtype": 9}, "raw": {"file": "../../etc/passwd", "sha256": "x"}}))
    with pytest.raises(SystemExit, match="refusing raw file path"):
        cli.main([str(tmp_path / "out"), "--rebuild", str(evil)])
    setup_logging(Console(stream=io.StringIO()))


def test_exit_status_2_when_a_logger_is_not_ready(tmp_path, monkeypatch):
    fake = FakeSolo("/dev/cu.X", n_samples=50)  # cannot be configured: it has no write commands
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    monkeypatch.setattr(cli, "rbr_ports", lambda: {"/dev/cu.X"})
    monkeypatch.setattr(cli, "ruskin_running", lambda: False)
    monkeypatch.setattr(cli, "SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "ONCE_GRACE_S", 0.0)
    monkeypatch.setattr(cli, "measure_clock_skew", lambda link, **k: {
        "n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "measured_at": ""})
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path), "--once", "--no-ntp", "--configure", "--no-clock", "--yes"])
    assert exc.value.code == 2
    setup_logging(Console(stream=io.StringIO()))


def test_sleep_resolution_is_fine_enough_for_the_clock_set():
    """set_clock sleeps to ~5 ms before the send, then spins. Python >= 3.11 on Windows uses a high-resolution
    waitable timer; before that sleep ticked in 15.6 ms steps (the review's concern)."""
    overshoot = []
    for _ in range(20):
        t0 = time.perf_counter()
        time.sleep(0.002)
        overshoot.append(time.perf_counter() - t0 - 0.002)
    assert statistics.median(overshoot) < 0.005
