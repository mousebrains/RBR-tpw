"""Offloads through real transports: the simulated loggers behind a TCP socket (every OS) or a pseudo-terminal
(macOS/Linux), opened with real pyserial. This covers what the replaced-pyserial tests cannot: read timeouts,
partial reads, termios setup, the exclusive lock, and `socket://` ports."""

import io
import json
import sys
import threading

import netCDF4
import pytest
import serial
from fakelogger import FakeConcerto3, FakeDuet, FakeSolo, serve_pty, serve_socket

from rbr_tpw import cli
from rbr_tpw.console import Console, setup_logging
from rbr_tpw.link import Link

TRANSPORTS = ["socket", pytest.param("pty", marks=pytest.mark.skipif(sys.platform == "win32",
                                                                    reason="no pseudo-terminals on Windows"))]


@pytest.fixture
def served(request, tmp_path, monkeypatch):
    """serve(fake) -> port, over the transport named by the test's `transport` parameter."""
    closers = []

    def serve(fake, transport):
        port, close = (serve_socket if transport == "socket" else serve_pty)(fake)
        closers.append(close)
        return port

    monkeypatch.setattr(cli, "ruskin_running", lambda: False)
    monkeypatch.setattr(cli, "SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "POLL_S", 0.05)
    out = io.StringIO()
    console = Console(stream=out, input_fn=lambda q: "", interactive=False, color=False)
    setup_logging(console, tmp_path / "session.log")
    serve.settings = lambda: cli.Settings(outdir=tmp_path, ntp_server=None, console=console,
                                          session_log=tmp_path / "session.log")
    serve.console_text = out.getvalue
    yield serve
    for close in closers:
        close()
    setup_logging(Console(stream=io.StringIO()))


def fast_skew(link, reps=3, max_seconds=15.0, clock=None):
    return {"n": 3, "polls": 9, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001,
            "individual_s": [0.0] * 3, "measured_at": "2026-09-26T00:00:00.000+00:00"}


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_solo_offload_with_a_real_skew_measurement(served, transport, tmp_path, monkeypatch):
    port = served(FakeSolo("fake", n_samples=40_000, skew_s=0.4, bytes_per_s=2_000_000), transport)
    monkeypatch.setattr(cli, "rbr_ports", lambda: {port})
    cli.run(served.settings(), once=True, port=None)
    (nc,) = tmp_path.glob("100689_*.nc")
    rec = json.loads((tmp_path / "raw" / f"{nc.stem}.json").read_text())
    assert rec["clock_skew"]["skew_vs_host_s"] == pytest.approx(0.4, abs=0.03)  # through the transport's latency
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 40_000
    assert "done with" in served.console_text() and "ERROR" not in served.console_text()


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("cls, sn, n", [(FakeDuet, "081500", 20_000), (FakeConcerto3, "233442", 10_000)])
def test_other_families(served, transport, cls, sn, n, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew)
    port = served(cls("fake", n_samples=n), transport)
    monkeypatch.setattr(cli, "rbr_ports", lambda: {port})
    cli.run(served.settings(), once=True, port=None)
    (nc,) = tmp_path.glob(f"{sn}_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == n


def test_explicit_socket_port(served, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew)
    port = served(FakeSolo("fake", n_samples=100), "socket")
    monkeypatch.setattr(cli, "rbr_ports", lambda: set())  # found only because --port names it
    cli.run(served.settings(), once=True, port=port)
    assert list(tmp_path.glob("100689_*.nc"))


@pytest.mark.skipif(sys.platform == "win32", reason="no pseudo-terminals on Windows")
def test_exclusive_lock_on_a_real_tty(served, tmp_path, monkeypatch):
    port = served(FakeSolo("fake", n_samples=100), "pty")
    holder = Link(port)  # e.g. another copy of rbr-offload
    try:
        with pytest.raises(serial.SerialException, match="exclusively lock"):
            Link(port)
        monkeypatch.setattr(cli, "rbr_ports", lambda: {port})
        cli.run(served.settings(), once=True, port=None)
        assert "is in use by another program" in served.console_text()
    finally:
        holder.close()


def test_stop_and_resume_over_a_socket(served, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew)
    port = served(FakeSolo("fake", serial=77, n_samples=150_000, bytes_per_s=300_000), "socket")
    s = served.settings()
    t = threading.Thread(target=cli._worker, args=(port, s))
    t.start()
    threading.Event().wait(0.9)
    s.stop.set()
    t.join(10)
    part = tmp_path / "raw" / ".partial" / "77.bin.part"
    assert part.exists() and (part.stat().st_size - 512) % 68_000 == 0
    s.stop.clear()
    cli._worker(port, s)
    (nc,) = tmp_path.glob("77_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 150_000
