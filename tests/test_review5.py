"""Fixes from a fifth external review (issue #9, 2026-09-26, on 4944021): a setting write answered with the
command text (Link.command skipped it as an echo, so --configure could not complete on a real fwtype-9 logger),
the L2 decoder's tail after a reading that looks like a truncated event, --rebuild of a Gen3 record with events
but no samples, and the exit status after a second Ctrl-C."""

import io
import json
import queue
import struct
import threading
import time

import netCDF4
import pytest
from fakelogger import SYNC_S, FakeConcerto3, FakeDuet, FakeSoloWritable, l2_sectioned_image
from test_offload import fast_skew, rig  # noqa: F401  (rig is a fixture)

from rbr_tpw import cli
from rbr_tpw import link as link_module
from rbr_tpw.configure import DeployConfig, configure
from rbr_tpw.console import Console, setup_logging
from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.equations import decode_l2, parse_l2_header
from rbr_tpw.link import Link, LoggerError

SKEW = {"n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "measured_at": ""}


@pytest.fixture
def fake_link(monkeypatch):
    def connect(fake):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
        return Link(fake.port)
    return connect


# --- 1. a reply equal to the command


def test_a_reply_equal_to_the_command_is_the_reply(fake_link):
    """SN100689 answers `now = X`, `starttime = X`, `endtime = X` and `sampling mode = ...` with the command
    text and nothing else. From d55d897 to 4944021 Link.command skipped such a line as an echo and timed out."""
    fake = FakeSoloWritable("/dev/cu.fake")
    link = fake_link(fake)
    fake.locked = False
    for cmd in ("now = 20260925210752", "starttime = 20000101000000", "endtime = 20991231235959",
                "sampling mode = continuous, period = 500"):
        t0 = time.monotonic()
        assert link.command(cmd) == cmd
        assert time.monotonic() - t0 < 1.0  # accepted after the echo grace, not after the 3 s timeout
    assert link.command("permit memclear") == "permit = memclear"
    assert fake.state["sampling"] == "mode = continuous, period = 500"
    assert not any(e.direction == "NOTE" and "timeout" in e.text for e in link.transcript)


class EchoingWritable(FakeSoloWritable):
    """A transport that echoes the command line before the logger's reply."""

    def _handle(self, cmd):
        if cmd:
            with self._lock:
                self._out += cmd.encode() + b"\r\n"
        super()._handle(cmd)


def test_an_echo_followed_by_a_reply_is_still_skipped(fake_link):
    fake = EchoingWritable("/dev/cu.fake")
    link = fake_link(fake)
    fake.locked = False
    assert link.command("status") == "status = logging"
    assert link.command("starttime = 20000101000000") == "starttime = 20000101000000"  # echo, then the same reply
    with pytest.raises(LoggerError, match="E0102"):
        link.command("bogus")  # echo, then an error line


def test_configure_completes_through_the_real_link(fake_link):
    """configure() end to end through Link.command, not through a fake that is the link (every earlier configure
    test did that, which is why the echo skip went unnoticed)."""
    fake = FakeSoloWritable("/dev/cu.fake", n_samples=50)
    link = fake_link(fake)
    cfg = DeployConfig(set_clock=False, fresh_battery=True, period_ms=500)
    report = configure(link, 100689, cfg, ntp_offset_s=0.0, log=lambda *a: None)
    assert [s["step"] for s in report["steps"]] == ["stop_sent", "stop", "battery_counter", "schedule", "erase_sent",
                                                    "erase", "verify", "enable_sent", "enable"]
    assert report["readback"]["status"] == "logging" and report["readback"]["sampling"]["period"] == "500"
    assert fake.state["used"] == 512 and fake.state["remaining"] == "2022900" and fake.locked
    tx = [e.text for e in link.transcript if e.direction == "TX"]
    assert tx.count("starttime = 20000101000000") == tx.count("endtime = 20991231235959") == 1
    assert not any(e.direction == "NOTE" and "timeout" in e.text for e in link.transcript)


def test_cli_configure_succeeds_on_a_logger_that_answers_writes_verbatim(tmp_path, monkeypatch):
    fake = FakeSoloWritable("/dev/cu.X", n_samples=50)
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    monkeypatch.setattr(cli, "rbr_ports", lambda: {"/dev/cu.X"})
    monkeypatch.setattr(cli, "ruskin_running", lambda: False)
    monkeypatch.setattr(cli, "SETTLE_S", 0.0)
    monkeypatch.setattr(cli, "ONCE_GRACE_S", 0.0)
    monkeypatch.setattr(cli, "measure_clock_skew", lambda link, **k: SKEW)
    code = 0
    try:
        cli.main([str(tmp_path), "--once", "--no-ntp", "--configure", "--no-clock", "--yes", "--period-ms", "1000"])
    except SystemExit as exc:
        code = exc.code
    finally:
        setup_logging(Console(stream=io.StringIO()))
    assert code == 0  # 2 (NOT READY TO DEPLOY) on 4944021: the first write timed out
    (rep,) = (tmp_path / "raw").glob("100689_*_configure.json")
    report = json.loads(rep.read_text())
    assert [s["step"] for s in report["steps"]][-1] == "enable" and "error" not in report
    assert report["readback"]["sampling"]["period"] == "1000" and report["readback"]["status"] == "logging"
    assert fake.state["used"] == 512 and fake.state["status"] == "logging"


# --- 2. decode_l2: a reading that looks like a truncated event


def _duet_image(n=50):
    return l2_sectioned_image(n, [("temp12", 0, FakeDuet.TEMP), ("pres21", 0, FakeDuet.PRES),
                                  ("temp05", 0, FakeDuet.COMP)])


def test_decode_l2_keeps_the_tail_after_a_reading_that_looks_like_a_truncated_event():
    clean = _duet_image()
    first = parse_l2_header(clean).length + 8  # header, then the 8-byte time-sync event, then 150 reading words
    cases = {  # reading word index -> value. Byte 10 of a would-be 0xF3 record is byte 2 of the next-next word.
        "0xF3 near the end with length byte 0x30 (a 192-byte record)": [(130, 0xF3A0B0C0), (132, 0x1A30_1234)],
        "0xF7 as the last word": [(149, 0xF7A0B0C0)],
        "0xF5 as the last word but one": [(148, 0xF5A0B0C0)],
        "the same 0xF3 word mid-image (control)": [(30, 0xF3A0B0C0), (32, 0x1A30_1234)],
    }
    for name, pokes in cases.items():
        img = bytearray(clean)
        for k, v in pokes:
            struct.pack_into("<I", img, first + 4 * k, v)
        d = decode_l2(bytes(img), 3)
        assert d.raw.shape == (50, 3), name  # 4944021: 43, 49, 49 and 50 sets
        assert d.bad_event_words == 1 and d.trailing_bytes == 0 and len(d.events) == 1, name
        k, v = pokes[0]
        assert d.raw[k // 3, k % 3] == v, name  # kept as a (negative) reading
    # a genuine time-sync event at the very end is still an event
    body = bytes([0x01, 0xF7]) + struct.pack("<I", SYNC_S + 25)
    d = decode_l2(clean + struct.pack(">H", crc16_ccitt(body)) + body, 3)
    assert d.raw.shape == (50, 3) and len(d.events) == 2 and d.bad_event_words == 0 and d.trailing_bytes == 0


# --- 3. --rebuild of a Gen3 record with events but no samples


def test_rebuild_of_a_gen3_record_with_events_but_no_samples(rig, monkeypatch, tmp_path):  # noqa: F811
    """`meminfo dataset = 1, used = 0` beside dataset-0 events: a gated concerto3 never activated (10 such replies
    in Ruskin's logs). Live, download() gives dataset 1 as b""; the record lists only the non-empty datasets."""
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    rig.add("usbmodemC", cls=FakeConcerto3, n_samples=0)
    cli.run(rig.settings(), once=True, port=None)
    (rec,) = (rig.tmp / "raw").glob("233442_*.json")
    assert sorted(json.loads(rec.read_text())["datasets"]) == ["dataset0", "dataset2"]
    (live,) = rig.tmp.glob("233442_*.nc")
    with netCDF4.Dataset(live) as ds:
        assert len(ds["time"]) == 0
    out = tmp_path / "rebuilt"
    cli.main([str(out), "--rebuild", str(rec)])  # KeyError: 'dataset1' on 4944021
    with netCDF4.Dataset(out / f"{rec.stem}.nc") as ds:
        assert len(ds["time"]) == 0


# --- 4. a second Ctrl-C with a NetCDF write still queued


def test_a_second_ctrl_c_reports_a_queued_netcdf_write_as_a_failure(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    monkeypatch.setattr(cli, "_stages", {})  # tests that call _configure_step() directly leave a stage behind
    rig.add("usbmodemA", n_samples=50)
    s = rig.settings()
    real_serve, presses = s.console.serve, []

    def serve(timeout):
        """The operator presses Ctrl-C as the worker hands its NetCDF write to the main thread, then again."""
        try:
            job = s.console._jobs.get(timeout=timeout)
        except queue.Empty:
            return False
        s.console._jobs.put(job)  # left queued: the worker stays blocked in run_in_main()
        presses.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(s.console, "serve", serve)
    cli.run(s, once=False, port=None)
    assert len(presses) == 2 and not list(rig.tmp.glob("*.nc"))
    assert "quit at once; interrupted: SN100689@usbmodemA writing NetCDF" in rig.console_text()
    assert s.failed == ["SN100689@usbmodemA: interrupted (Ctrl-C twice) while writing NetCDF"]  # exit 1, not 0
    assert not s.not_ready
    # let the abandoned worker finish its write, so it does not outlive the test
    deadline = time.monotonic() + 10
    while (any(t.name.startswith("offload-") and t.is_alive() for t in threading.enumerate())
           and time.monotonic() < deadline):
        real_serve(0.2)
