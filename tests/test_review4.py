"""Fixes from a fourth external review (2026-09-25): Gen3 `offsetfromutc = unknown`, .rsk acquisition order,
the offload record after a failed estimate, settings checked up front, clock corrections left in 2000, NTP
failures and fallbacks, the --once grace, short header reads, --no-enable, Gen4 column sizes, and the
--rebuild path guard."""

import io
import json
import socket
import sys
import threading
import time
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from fakelogger import FakeConcerto3, FakeSolo
from test_gen4 import FakeGen4
from test_power_and_cli import FakeSolo as ConfigFake
from test_power_and_cli import VirtualClock
from test_rsk import CONCERTO, _concerto_values, make_rsk

from rbr_tpw import cli, gen4, hostclock, solo
from rbr_tpw import link as link_module
from rbr_tpw.configure import ConfigError, DeployConfig, configure, validate
from rbr_tpw.console import Console, setup_logging
from rbr_tpw.drivers import Gen4Driver, L2Driver
from rbr_tpw.link import Link, LinkError
from rbr_tpw.rawbin import EPOCH2000_MS, TFLAG_RESET_CLOCK, event_indices, resolve_time_arrays
from rbr_tpw.rsk import convert

T = 1_727_740_800_000  # 2024-10-01T00:00:00Z
R = EPOCH2000_MS  # where a logger clock restarts after a power loss
SKEW = {"n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "measured_at": ""}


@pytest.fixture
def console():
    out = io.StringIO()
    setup_logging(Console(stream=out, interactive=False, color=False))
    yield out
    setup_logging(Console(stream=io.StringIO()))


@pytest.fixture
def once(monkeypatch, tmp_path):
    """Run `rbr-offload --once` against fake loggers; returns (exit status, console text)."""
    def go(fakes, *extra, ports=None, grace=0.0):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fakes[port])
        monkeypatch.setattr(cli, "rbr_ports", ports or (lambda: set(fakes)))
        monkeypatch.setattr(cli, "ruskin_running", lambda: False)
        monkeypatch.setattr(cli, "SETTLE_S", 0.0)
        monkeypatch.setattr(cli, "ONCE_GRACE_S", grace)
        monkeypatch.setattr(cli, "measure_clock_skew", lambda link, **k: dict(SKEW))
        out = io.StringIO()
        monkeypatch.setattr(cli, "Console", lambda: Console(stream=out, interactive=False, color=False))
        code = 0
        try:
            cli.main([str(tmp_path), "--once", "--no-ntp", *extra])
        except SystemExit as exc:
            code = exc.code
        finally:
            setup_logging(Console(stream=io.StringIO()))
        return code, out.getvalue()
    return go


class UnknownOffsetConcerto3(FakeConcerto3):
    """`offsetfromutc = unknown`: the default, and what setting the clock leaves (L3 ref 4.1.1)."""

    def replies(self):
        r = super().replies()
        r["clock"] = r["clock"].replace("+0.00", "unknown")
        return r


def test_gen3_with_offsetfromutc_unknown_is_offloaded(once, tmp_path):
    code, text = once({"/dev/cu.C": UnknownOffsetConcerto3("/dev/cu.C", n_samples=50)})
    assert code == 0, text
    assert list(tmp_path.glob("233442_*.nc")) and list((tmp_path / "raw").glob("233442_*.json"))


def test_rsk_mid_deployment_reset_is_read_in_acquisition_order(tmp_path):
    """10 samples on a set clock, a power loss, 10 on the clock restarted at 2000-01-01 (rows inserted in that
    order). Read in time order the reset samples came first and were all dropped; in rowid order the last
    run is re-timed by Ruskin's drift and the event lands before sample 11."""
    t = np.concatenate([T + 1000 * np.arange(10), R + 5000 + 1000 * np.arange(10)])
    drift = int(R + 5000) - (T + 60_000)  # the reset run began at 2024-10-01T00:01:00 UTC
    rsk = make_rsk(tmp_path / "r.rsk", model="RBRconcerto", serial=60275, fwtype=103, kind="full",
                   channels=CONCERTO, t_ms=t, values=_concerto_values(20), drift_ms=drift,
                   tod_ms=int(t[-1]) + 2000, events=[(int(R + 4000), 0x0A, 11)])
    r = convert(rsk, tmp_path / "r.nc")
    assert r.samples == 20 and r.t_first == T and r.t_last == T + 69_000, r.warnings
    with netCDF4.Dataset(tmp_path / "r.nc") as nc:
        assert np.all(np.diff(nc["time"][:]) > 0)
        index = next(v for name, v in nc.variables.items() if name.startswith("event") and "index" in name)
        assert index[:].tolist() == [10]


def test_event_jitter_is_not_a_clock_restart():
    """Events in acquisition order step back a little (1.7 s in a Gen3 .rsk): that is not a restart, which
    matters once there are two runs on the reset clock."""
    t = np.concatenate([R + 5000 + 1000 * np.arange(600), R + 3000 + 1000 * np.arange(5)])  # 10 min, then a restart
    ev = [R + 100_000, R + 98_300, R + 200_000, R + 2000]
    assert event_indices(t, ev) == [95, 94, 195, 600]  # the jittered event is placed by its own time


def test_a_correction_that_leaves_samples_in_2000_is_refused():
    t = np.concatenate([R + 5000 + 1000 * np.arange(5), T + 1000 * np.arange(5)])
    tflags = np.where(t < T, TFLAG_RESET_CLOCK, 0).astype(np.uint8)
    out, _, keep, notes = resolve_time_arrays(t, tflags, np.zeros(10, np.int32), 0.002, T + 60_000)
    assert keep.tolist() == [False] * 5 + [True] * 5 and any("does not give a consistent time" in n for n in notes)


def test_thresholds_must_be_numbers(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("thresholds:\n  min_days: '90'\n")
    with pytest.raises(ConfigError, match="must be a number"):
        cli.Thresholds.from_yaml(bad)
    good = tmp_path / "g.yaml"
    good.write_text("thresholds:\n  min_days: 90\n  min_battery_voltage: 3\n")
    assert cli.Thresholds.from_yaml(good) == cli.Thresholds(min_battery_voltage=3.0, min_days=90.0)


def test_the_record_is_saved_even_if_the_estimate_fails(once, tmp_path, monkeypatch):
    def broken(*a, **k):
        raise TypeError("simulated odd reply")
    monkeypatch.setattr(cli, "_remaining", broken)
    code, text = once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=200)})
    record = json.loads(next((tmp_path / "raw").glob("100689_*[0-9]Z.json")).read_text())
    assert any("not estimated" in w for w in record["warnings"]) and list(tmp_path.glob("100689_*.nc"))


def test_partial_files_are_kept_until_the_record_is_saved(once, tmp_path, monkeypatch):
    real = cli._write_json

    def failing(path, obj):
        if not path.name.endswith("_configure.json"):
            raise OSError("simulated full disk")
        real(path, obj)
    monkeypatch.setattr(cli, "_write_json", failing)
    code, _ = once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=200)})
    assert code == 1 and list((tmp_path / "raw" / ".partial").glob("100689*"))  # a reconnect can resume


def test_schedule_options_are_checked_before_any_download(once, tmp_path, capsys):
    code, _ = once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=200)}, "--configure", "--yes", "--no-clock",
                   "--period-ms", "250")
    assert code == 2 and "use 500 ms" in capsys.readouterr().err  # a usage error: nothing was touched
    assert not (tmp_path / "raw").exists()


def test_config_and_configure_file_together_are_refused(tmp_path, capsys):
    (tmp_path / "a.yaml").write_text("thresholds: {min_days: 10}\n")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path), "--config", str(tmp_path / "a.yaml"), "--configure", str(tmp_path / "a.yaml")])
    assert exc.value.code == 2 and "not both" in capsys.readouterr().err


def test_a_float_period_reaches_the_logger_as_an_integer():
    assert validate(DeployConfig(period_ms=500.0), 1000)[2] == 500
    assert isinstance(validate(DeployConfig(period_ms=2000.0), 1000)[2], int)
    with pytest.raises(ConfigError, match="whole number"):
        validate(DeployConfig(period_ms=500.5), 1000)


def test_failed_ntp_is_loud_and_labelled_host(console, monkeypatch):
    import rbr_tpw.configure as cf
    vc = VirtualClock()  # the clock set must not depend on host scheduling (62-101 ms late on a CI runner)
    monkeypatch.setattr(cf, "time", vc)
    fake = ConfigFake(status="logging", clock=vc)
    s = cli.Settings(outdir=Path("."), deploy=DeployConfig(), assume_yes=True, ntp_server="time.example")
    snap = {"status": "logging", "channel_list": [{"index": 1, "type": "temp09"}]}
    monkeypatch.setattr(cf, "measure_clock_skew", lambda link, reps=3: {
        "n": 3, "skew_vs_host_s": fake.offset, "uncertainty_s": 0.001})
    report = cli._configure_step(fake, s, snap, {"error": "timed out"}, "100689", "saved")
    text = console.getvalue()
    assert "HOST CLOCK, NOT CHECKED AGAINST UTC" in text and "logger - host" in text and "logger - UTC" not in text
    assert report["clock_reference"] == "host clock (no NTP)"
    assert report["steps"][[st["step"] for st in report["steps"]].index("set_clock")]["reference"] == "host"
    cli._not_ready.pop(fake.port, None)


def test_no_enable_leaves_the_logger_not_ready(console, monkeypatch):
    import rbr_tpw.configure as cf
    monkeypatch.setattr(cf, "IDLE_SAVE_S", 0.0)
    fake = ConfigFake(status="logging")
    s = cli.Settings(outdir=Path("."), deploy=DeployConfig(set_clock=False, erase=False, enable=False),
                     assume_yes=True)
    cli._configure_step(fake, s, {"status": "logging", "channel_list": [{"index": 1, "type": "temp09"}]}, {},
                        "100689", "saved")
    assert "not enabled" in cli._not_ready.pop(fake.port)


def test_once_grace_counts_from_the_end_of_a_long_offload(once, tmp_path, monkeypatch):
    """A's offload outlasts the grace; B enumerates 0.3 s after A is done and must still be offloaded."""
    monkeypatch.setattr(cli, "POLL_S", 0.05)
    fakes = {"/dev/cu.A": FakeSolo("/dev/cu.A", serial=11, n_samples=2000, bytes_per_s=4000),
             "/dev/cu.B": FakeSolo("/dev/cu.B", serial=22, n_samples=50)}
    done_at = []

    def ports():
        if list(tmp_path.glob("11_*.nc")) and not done_at:
            done_at.append(time.monotonic())
        return {"/dev/cu.A"} | ({"/dev/cu.B"} if done_at and time.monotonic() - done_at[0] > 0.3 else set())
    code, text = once(fakes, ports=ports, grace=1.0)
    assert list(tmp_path.glob("11_*.nc")) and list(tmp_path.glob("22_*.nc")), text


class ShortHead:
    """read_data answers 500 of the 512 header bytes (with a good CRC, so the link accepts it)."""

    def read_data(self, dataset, n, offset, l3=False):
        return bytes(min(n, 500) if offset == 0 else n)

    def note(self, text):
        pass


def test_a_short_header_read_is_refused_before_anything_is_written(tmp_path):
    part = tmp_path / "p" / "x.part"
    with pytest.raises(LinkError, match="short header read"):
        solo.download(ShortHead(), 160_520, part)
    assert not part.exists()


def test_gen4_open_dataset_whose_counts_include_a_partial_record(monkeypatch, tmp_path):
    fake = FakeGen4(n_samples=20)
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link("/dev/cu.gen4")
    data = gen4.download(link, tmp_path / "part", "210000")
    snap = Gen4Driver(120).snapshot(link)
    sch = snap["schedules"]["sch_ctd"]
    sch["samplecount"], sch["bytecount"] = 21, len(data["dataset_01/sch_ctd/data"]) + 7  # one being written
    _, _, cols, _, _ = gen4.columns(data, snap)
    assert len(cols) == 3


def test_gen4_raw_names_the_converted_data(monkeypatch, tmp_path):
    fake = FakeGen4(n_samples=20)
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link("/dev/cu.gen4")
    data = gen4.download(link, tmp_path / "part", "210000")
    datasets = {k: {"file": f"raw/210000_{k.replace('/', '_')}.bin", "bytes": len(v), "sha256": ""}
                for k, v in data.items()}
    record = {"offload_started": "2026-09-25T00:00:00.000Z", "id": {"model": "RBRconcerto3", "version": "2.1.0",
              "serial": "210000", "fwtype": 120}, "snapshot_before": Gen4Driver(120).snapshot(link),
              "warnings": [], "datasets": datasets, "raw": datasets["dataset_01/meta"]}
    Gen4Driver(120).write_netcdf(data, record, tmp_path / "g.nc")
    with netCDF4.Dataset(tmp_path / "g.nc") as nc:
        assert nc.raw_file == "raw/210000_dataset_01_sch_ctd_data.bin"


def _ntp_server(stop):
    """A local SNTP server (mode 4, stratum 1) on 127.0.0.1; returns its port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.1)

    def serve():
        while not stop.is_set():
            try:
                req, addr = sock.recvfrom(512)
            except TimeoutError:
                continue
            now = hostclock._to_ntp(time.time())
            sock.sendto(bytes([0x24, 1]) + bytes(22) + req[40:48] + now + now, addr)
        sock.close()
    threading.Thread(target=serve, daemon=True).start()
    return sock.getsockname()[1]


def test_ntp_tries_the_next_address_after_a_failure(monkeypatch):
    stop = threading.Event()
    port = _ntp_server(stop)
    dead = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dead.bind(("127.0.0.1", 0))  # bound, never answers: like an unreachable first (e.g. IPv6) address
    infos = [(socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("127.0.0.1", dead.getsockname()[1])),
             (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("127.0.0.1", port))]
    monkeypatch.setattr(hostclock.socket, "getaddrinfo", lambda *a, **k: infos)
    try:
        r = hostclock.ntp_offset("time.example", timeout=3.0, samples=3)
    finally:
        stop.set()
        dead.close()
    assert r.get("offset_s") is not None and r["address"] == "127.0.0.1" and r["n"] == 2, r
    assert abs(r["offset_s"]) < 0.05


def test_ntp_timestamps_after_2036():
    t = 2_120_000_000.25  # 2037-03-07, in NTP era 1
    assert hostclock._from_ntp(hostclock._to_ntp(t)) == pytest.approx(t, abs=1e-6)
    assert hostclock._from_ntp(hostclock._to_ntp(1_790_000_000.5)) == pytest.approx(1_790_000_000.5, abs=1e-6)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights on Windows")
def test_rebuild_refuses_a_raw_file_that_resolves_outside(tmp_path):
    outside = tmp_path / "elsewhere.bin"
    outside.write_bytes(b"x" * 10)
    raw = tmp_path / "out" / "raw"
    raw.mkdir(parents=True)
    (raw / "a.bin").symlink_to(outside)
    rec = raw / "a.json"
    rec.write_text(json.dumps({"id": {"fwtype": 9}, "raw": {"file": "raw/a.bin", "sha256": "x", "bytes": 10}}))
    with pytest.raises(SystemExit, match="resolves outside"):
        cli.main([str(tmp_path / "nc"), "--rebuild", str(rec)])
    setup_logging(Console(stream=io.StringIO()))


def test_the_erase_note_counts_what_was_logged_during_the_download(once, tmp_path, monkeypatch):
    real = L2Driver.after

    def after(self, link):
        a = real(self, link)
        a["meminfo"] = {**a["meminfo"], "used": a["meminfo"]["used"] + 1234}
        return a
    monkeypatch.setattr(L2Driver, "after", after)
    code, text = once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=200)}, "--configure", "--yes", "--no-clock")
    assert "1234 bytes logged since the download began are NOT saved" in text


def test_synthetic_duet_header_has_the_extra_pressure_word():
    from fakelogger import FakeDuet

    from rbr_tpw.equations import parse_l2_header
    h = parse_l2_header(FakeDuet("/dev/cu.D", n_samples=10).datasets[1])
    words = {ch["type"]: len(ch["coefficient_words"]) for ch in h.channels}
    assert words["pres21"] == 12 and words["temp12"] == 4  # as on SN081500


def test_configure_reports_its_clock_reference(monkeypatch):
    import rbr_tpw.configure as cf
    vc = VirtualClock()
    monkeypatch.setattr(cf, "time", vc)
    fake = ConfigFake(status="logging", clock=vc)
    monkeypatch.setattr(cf, "measure_clock_skew", lambda link, reps=3: {
        "n": 3, "skew_vs_host_s": fake.offset, "uncertainty_s": 0.001, "spread_s": 0.0, "measured_at": ""})
    report = configure(fake, 100689, DeployConfig(), 0.012, log=lambda *a: None)
    assert report["clock_reference"].startswith("UTC")
