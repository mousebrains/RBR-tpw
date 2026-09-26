"""rbr-offload end to end against simulated loggers (tests/fakelogger.py): logging, parallel workers,
failures, Ctrl-C and resume, and the console's prompt handling."""

import io
import itertools
import json
import logging
import threading
import time
import types

import netCDF4
import numpy as np
import pytest
import serial
from fakelogger import FakeConcerto3, FakeDuet, FakeSolo

from rbr_tpw import cli
from rbr_tpw import link as link_module
from rbr_tpw.configure import DeployConfig
from rbr_tpw.console import Console, setup_logging


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """Simulated loggers on fake ports; returns a namespace to add loggers and build Settings."""
    fakes: dict[str, FakeSolo] = {}
    busy: set[str] = set()

    def open_port(port, baudrate=115200):
        if port in busy:
            raise serial.SerialException(35, f"Could not exclusively lock port {port}: [Errno 35]")
        return fakes[port]

    monkeypatch.setattr(link_module, "open_serial", open_port)
    monkeypatch.setattr(cli, "rbr_ports", lambda: set(fakes) | busy)
    monkeypatch.setattr(cli, "ruskin_running", lambda: False)
    monkeypatch.setattr(cli, "SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "ONCE_GRACE_S", 0.0)
    monkeypatch.setattr(cli, "POLL_S", 0.05)
    out = io.StringIO()
    console = Console(stream=out, input_fn=lambda q: "", interactive=False, color=False)
    session_log = tmp_path / "raw" / "session.log"
    session_log.parent.mkdir(parents=True)
    setup_logging(console, session_log)

    class Rig:
        def add(self, name, cls=FakeSolo, **kw):
            fakes[f"/dev/cu.{name}"] = cls(f"/dev/cu.{name}", **kw)
            return fakes[f"/dev/cu.{name}"]

        def settings(self, **kw):
            return cli.Settings(outdir=tmp_path, ntp_server=None, console=console, session_log=session_log, **kw)

        def console_text(self):
            return out.getvalue()

        def session_text(self):
            return session_log.read_text()

    r = Rig()
    r.busy, r.tmp = busy, tmp_path
    yield r
    setup_logging(Console(stream=io.StringIO()))  # detach the per-test file handler


def fast_skew(counter):
    """Stand-in for measure_clock_skew that records how many run at once."""
    def measure(link, reps=3, max_seconds=15.0, clock=None):
        with counter["lock"]:
            counter["now"] += 1
            counter["max"] = max(counter["max"], counter["now"])
        time.sleep(0.2)
        with counter["lock"]:
            counter["now"] -= 1
        return {"n": 3, "polls": 9, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001,
                "individual_s": [0.0] * 3, "measured_at": "2026-09-25T00:00:00.000+00:00"}
    return measure


def test_single_offload_logs_everything(rig):
    fake = rig.add("usbmodem101", serial=100689, n_samples=5000, skew_s=0.25)
    cli.run(rig.settings(), once=True, port=None)

    raw = rig.tmp / "raw"
    (nc,) = rig.tmp.glob("100689_*.nc")
    stem = nc.stem
    record = json.loads((raw / f"{stem}.json").read_text())
    assert record["clock_skew"]["skew_vs_host_s"] == pytest.approx(0.25, abs=0.03)  # the real skew measurement
    assert record["transcript"] == f"raw/{stem}.log" and record["session_log"] == "raw/session.log"
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 5000

    transcript = (raw / f"{stem}.log").read_text()
    assert not list(raw.glob("*_usbmodem101.log"))  # renamed once the serial number was known
    for expected in ("NOTE  opened /dev/cu.usbmodem101", "TX    <CR> (wake-up", "DROP", "TX    id",
                     "RX    id model = RBRsolo", "RX    <20008 data bytes + CRC", "NOTE  closing port"):
        assert expected in transcript, expected
    assert transcript.count("TX    now") >= 3

    session = rig.session_text()
    assert "SN100689@usbmodem101" in session and "rbr_tpw.serial: TX   read data 1 512 0" in session
    assert "stage: downloading" in session and "wrote " in session
    console = rig.console_text()
    assert "[SN100689@usbmodem101] clock skew (logger - host): +0.25" in console
    assert "done with /dev/cu.usbmodem101" in console and "TX" not in console  # serial traffic stays in the file
    assert fake.commands[0] == ""  # wake-up first


def test_parallel_offloads_overlap_but_skew_does_not(rig, monkeypatch, capfd):
    counter = {"lock": threading.Lock(), "now": 0, "max": 0}
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew(counter))
    fakes = [rig.add(f"usbmodem{i}", serial=100 + i, n_samples=100_000, bytes_per_s=400_000) for i in (1, 2, 3)]
    t0 = time.monotonic()
    cli.run(rig.settings(), once=True, port=None)
    elapsed = time.monotonic() - t0

    assert sorted(p.name.split("_")[0] for p in rig.tmp.glob("*.nc")) == ["101", "102", "103"]
    assert counter["max"] == 1  # timing-critical steps never overlapped
    assert "HDF5-DIAG" not in capfd.readouterr().err  # NetCDF written on the main thread
    overlaps = [min(a[-1][1], b[-1][1]) - max(a[1][0], b[1][0])
                for a, b in itertools.combinations([f.reads for f in fakes], 2)]
    assert max(overlaps) > 0.3  # downloads ran at the same time
    assert elapsed < 3 * 1.0 + 1.5  # each download alone takes ~1 s; one after another would be ~3.6 s
    for sn in ("101", "102", "103"):
        (log,) = (rig.tmp / "raw").glob(f"{sn}_*.log")
        text = log.read_text()
        assert f"serial = {sn}," in text and all(f"serial = {o}," not in text for o in {"101", "102", "103"} - {sn})


def test_failure_mid_download_keeps_transcript_and_partial(rig):
    rig.add("usbmodemX", serial=4242, n_samples=100_000, fail_at_offset=512 + 68_000)
    cli.run(rig.settings(), once=True, port=None)

    raw = rig.tmp / "raw"
    (transcript,) = raw.glob("4242_*.log")
    assert "RX    E0104 simulated read failure" in transcript.read_text()
    assert (raw / ".partial" / "4242.bin.part").stat().st_size == 512 + 68_000
    assert not list(rig.tmp.glob("*.nc"))
    session, console = rig.session_text(), rig.console_text()
    assert "ERROR" in session and "Traceback" in session
    assert "[SN4242@usbmodemX] ERROR: LoggerError:" in console and "Traceback" not in console
    assert "partial download resumes" in console


def test_busy_port_is_skipped_quietly(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodemA", serial=1)
    rig.busy.add("/dev/cu.usbmodemB")
    cli.run(rig.settings(), once=True, port=None)
    console = rig.console_text()
    assert "/dev/cu.usbmodemB is in use by another program" in console and "ERROR" not in console
    assert list(rig.tmp.glob("1_*.nc"))


def test_stop_between_blocks_then_resume(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodemS", serial=77, n_samples=150_000, bytes_per_s=300_000)
    s = rig.settings()
    t = threading.Thread(target=cli._worker, args=("/dev/cu.usbmodemS", s))
    t.start()
    time.sleep(0.9)
    s.stop.set()
    t.join(10)
    part = rig.tmp / "raw" / ".partial" / "77.bin.part"
    size = part.stat().st_size
    assert (size - 512) % 68_000 == 0 and 512 < size < 512 + 600_000  # stopped on a block boundary
    assert "download stopped (Ctrl-C)" in rig.console_text()

    s.stop.clear()
    cli._worker("/dev/cu.usbmodemS", s)
    (nc,) = rig.tmp.glob("77_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 150_000
    assert f"resuming from {size} bytes" in rig.session_text()


def test_configure_is_skipped_once_stopping(rig):
    s = rig.settings(deploy=DeployConfig())
    s.stop.set()
    link = types.SimpleNamespace(port="/dev/cu.stopping")
    assert cli._configure_step(link, s, {}, {}, "1", "x") == {"skipped": "stopping"}
    assert "not configured" in cli._not_ready.pop(link.port)  # a requested configure that did not happen


def test_console_asks_one_question_at_a_time():
    out = io.StringIO()
    inside = threading.Lock()
    asked = []

    def answer(question):
        assert inside.acquire(blocking=False), "two questions at once"
        try:
            asked.append(question)
            time.sleep(0.1)
            logging.getLogger("rbr_tpw.test").info("while asking")  # held until the question is answered
            return "y" if "one" in question else "n"
        finally:
            inside.release()

    console = Console(stream=out, input_fn=answer, interactive=True, color=False)
    setup_logging(console)
    answers = {}
    threads = [threading.Thread(target=lambda k=k: answers.update({k: console.ask(f"{k}? ")})) for k in ("one", "two")]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 5
    while len(answers) < 2 and time.monotonic() < deadline:
        console.serve(0.05)
    assert answers == {"one": "y", "two": "n"} and len(asked) == 2
    assert out.getvalue().count("while asking") == 2
    console.stop.set()
    assert console.ask("after stop? ") == ""
    setup_logging(Console(stream=io.StringIO()))


def test_ctrl_c_in_run_stops_downloads_and_returns(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodemC", serial=55, n_samples=150_000, bytes_per_s=300_000)
    s = rig.settings()
    real_serve, t0, fired = s.console.serve, time.monotonic(), []

    def serve(timeout):  # the operator presses Ctrl-C once, 0.9 s in
        if not fired and time.monotonic() - t0 > 0.9:
            fired.append(True)
            raise KeyboardInterrupt
        return real_serve(timeout)

    monkeypatch.setattr(s.console, "serve", serve)
    cli.run(s, once=False, port=None)  # returns only through the Ctrl-C shutdown
    console = rig.console_text()
    assert "Ctrl-C: stopping" in console and "SN55@usbmodemC downloading" in console
    assert "download stopped (Ctrl-C)" in console and console.rstrip().endswith("Stopped.")
    assert not list(rig.tmp.glob("*.nc")) and (rig.tmp / "raw" / ".partial" / "55.bin.part").exists()


def test_bench_mix_solo0_duet_concerto3_in_parallel(rig, monkeypatch):
    """Tomorrow's bench: three families at once, read-only."""
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    solo0 = rig.add("usbmodem101", serial=76313, fwtype=0, n_samples=30_000, bytes_per_s=300_000)
    duet = rig.add("usbmodem2101", cls=FakeDuet, n_samples=20_000, bytes_per_s=300_000)
    c3 = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=10_000, bytes_per_s=300_000)
    cli.run(rig.settings(), once=True, port=None)
    raw = rig.tmp / "raw"

    # fwtype 0 solo: decoded like fwtype 9; no energy counter, so memory-limited days only
    (nc0,) = rig.tmp.glob("76313_*.nc")
    rec0 = json.loads((raw / f"{nc0.stem}.json").read_text())
    assert rec0["family"] == "L2" and rec0["remaining_time"]["limited_by"] == "memory (energy not modelled)"
    with netCDF4.Dataset(nc0) as ds:
        assert len(ds["time"]) == 30_000 and ds.instrument_firmware_type == 0

    # duet: 3 stored channels; the raw download is kept whatever the decoder does
    (dj,) = raw.glob("081500_*.json")
    recd = json.loads(dj.read_text())
    assert recd["raw"]["bytes"] == len(duet.image) and len(recd["snapshot_before"]["channel_list"]) == 3
    assert recd["after"]["power"]["battery_voltage_V"] == pytest.approx(3.61)
    assert recd["after"]["power"]["remaining_raw"] == "1BA8140"
    assert (raw / f"{dj.stem}.bin").read_bytes() == duet.image
    with netCDF4.Dataset(rig.tmp / f"{dj.stem}.nc") as ds:  # pressure uses channel 3 (hidden thermistor)
        assert len(ds["time"]) == 20_000 and {"temperature", "pressure", "temperature03"} <= set(ds.variables)
        p, t3 = ds["pressure"][:], ds["temperature03"][:]
        assert np.all(np.isfinite(p)) and np.all(np.isfinite(t3))
        assert "standard_name" not in ds["temperature03"].ncattrs()  # hidden channel
    assert "no converter" not in rig.console_text() and "raw readings only" not in rig.console_text()

    # concerto3: three datasets saved, EasyParse decoded, 6 stored of 8 channels
    (cj,) = raw.glob("233442_*.json")
    recc = json.loads(cj.read_text())
    assert sorted(recc["datasets"]) == ["dataset0", "dataset1", "dataset2"]
    for name, info in recc["datasets"].items():
        assert (rig.tmp / info["file"]).read_bytes() == c3.datasets[int(name[-1])]
    assert recc["bytes_per_sample"] == 8 + 4 * 6 and recc["after"]["power"]["battery_voltage_V"] == 14.63
    with netCDF4.Dataset(rig.tmp / f"{cj.stem}.nc") as ds:
        assert len(ds["time"]) == 10_000 and list(ds["event_type"][:]) == [0x18, 0x19]
        assert {"conductivity", "temperature", "pressure", "sea_pressure", "depth", "salinity"} <= set(ds.variables)
    assert any(c.startswith("readdata size = ") for c in c3.commands)
    assert not any("=" in c and not c.startswith("readdata") for c in c3.commands + duet.commands + solo0.commands)


def test_configure_refused_for_other_families(rig, monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    duet = rig.add("usbmodemD", cls=FakeDuet, n_samples=100)
    cli.run(rig.settings(deploy=DeployConfig(), assume_yes=True), once=True, port=None)
    assert "--configure is implemented only for RBRsolo fwtype 9" in rig.console_text()
    assert not any(c.startswith(("lock", "stop", "enable", "memclear", "permit")) for c in duet.commands)


def test_unsupported_fwtype_is_left_alone(rig):
    fake = rig.add("usbmodemU", serial=5)
    fake.fwtype = 77
    cli.run(rig.settings(), once=True, port=None)
    assert "fwtype 77 is not supported yet" in rig.console_text() and fake.commands[-1] == "id"


def test_rebuild_gen3_record(rig, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodemC", cls=FakeConcerto3, n_samples=500)
    cli.run(rig.settings(), once=True, port=None)
    (rec,) = (rig.tmp / "raw").glob("233442_*.json")
    out = tmp_path / "rebuilt"
    cli.main([str(out), "--rebuild", str(rec)])
    with netCDF4.Dataset(out / f"{rec.stem}.nc") as ds:
        assert len(ds["time"]) == 500
