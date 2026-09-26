"""Fixes from a third external review (2026-09-25): configure failures that lose a reply, Gen4 columns, clock
resets on per-sample timestamps, exit status for failed offloads and skipped configures, and --rebuild
batches."""

import io
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from fakelogger import FakeDuet, FakeSolo
from test_easyparse import make_event, make_records
from test_gen4 import CHANNELS, FakeGen4, build_meta
from test_power_and_cli import FakeSolo as ConfigFake

from rbr_tpw import cli, gen4
from rbr_tpw import link as link_module
from rbr_tpw.configure import DeployConfig, configure
from rbr_tpw.console import Console, setup_logging
from rbr_tpw.drivers import Gen3Driver, Gen4Driver, L2Driver
from rbr_tpw.link import Link, LinkError
from rbr_tpw.rawbin import EPOCH2000_MS, event_indices, reset_segments, resolve_time_arrays

T = 1_727_740_800_000  # 2024-10-01T00:00:00Z
R = EPOCH2000_MS  # where a logger clock restarts after a power loss


@pytest.fixture
def console():
    out = io.StringIO()
    setup_logging(Console(stream=out, interactive=False, color=False))
    yield out
    setup_logging(Console(stream=io.StringIO()))


class LostEraseReply(ConfigFake):
    """Erases, then the link drops before the prompt comes back."""

    def command_until_prompt(self, cmd, timeout=60.0):
        super().command_until_prompt(cmd, timeout)
        raise LinkError("no prompt after 'memclear' (simulated disconnect)")


class LostReadback(ConfigFake):
    """Everything is done, then the link drops while the settings are read back."""

    def command(self, cmd, timeout=3.0):
        if cmd == "endtime" and "enable" in self.log:
            raise LinkError("timeout on 'endtime' (simulated disconnect)")
        return super().command(cmd, timeout)


def test_a_lost_reply_after_the_erase_is_not_reported_as_memory_not_erased(console):
    with pytest.raises(LinkError) as exc:
        configure(LostEraseReply(status="logging"), 100689, DeployConfig(set_clock=False), 0.0, log=lambda *a: None)
    done = [st["step"] for st in exc.value.report["steps"]]
    assert "erase_sent" in done and "erase" not in done and "error" in exc.value.report
    assert "MEMORY MAY HAVE BEEN ERASED" in cli._configure_state(done)

    fake = LostEraseReply(status="logging")
    s = cli.Settings(outdir=Path("."), deploy=DeployConfig(set_clock=False), assume_yes=True)
    cli._configure_step(fake, s, {"status": "logging"}, {}, "100689", "saved")
    assert "MEMORY MAY HAVE BEEN ERASED" in cli._not_ready.pop(fake.port)
    assert "MEMORY MAY HAVE BEEN ERASED" in console.getvalue() and "memory not erased" not in console.getvalue()
    assert "100689" in s.configure_failed and "100689" not in s.configured

    # reconnected: tried again (its memory was offloaded again first), and says so
    fake2 = ConfigFake(status="stopped")
    snap = {"status": "stopped", "channel_list": [{"index": 1, "type": "temp09"}]}
    report = cli._configure_step(fake2, s, snap, {}, "100689", "saved")
    assert "failed earlier in this session; trying again" in console.getvalue()
    assert report["readback"]["status"] == "logging" and fake2.port not in cli._not_ready


def test_a_failed_readback_still_reports_what_was_done():
    with pytest.raises(LinkError) as exc:
        configure(LostReadback(status="logging"), 100689, DeployConfig(set_clock=False), 0.0, log=lambda *a: None)
    done = [st["step"] for st in exc.value.report["steps"]]
    assert {"stop", "erase", "enable"} <= set(done)
    state = cli._configure_state(done)
    assert "MEMORY WAS ERASED" in state and "logging was enabled" in state


def _gen4_record(snap):
    return {"offload_started": "2026-09-25T00:00:00.000Z", "id": {"model": "RBRconcerto3", "version": "2.1.0",
            "serial": "210000", "fwtype": 120}, "snapshot_before": snap, "warnings": []}


def _gen4_download(monkeypatch, tmp_path, fake):
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link("/dev/cu.gen4")
    data = gen4.download(link, tmp_path / "part", "210000")
    return data, Gen4Driver(120).snapshot(link)


def test_gen4_netcdf_uses_the_datasets_own_columns(monkeypatch, tmp_path):
    """4 channels on the instrument, 3 stored: the NetCDF follows the dataset's metadata, not channel_list."""
    fake = FakeGen4(n_samples=20)
    data, snap = _gen4_download(monkeypatch, tmp_path, fake)
    warnings, t = Gen4Driver(120).write_netcdf(data, _gen4_record(snap), tmp_path / "g.nc")
    assert len(t) == 20
    with netCDF4.Dataset(tmp_path / "g.nc") as ds:
        labels = {v.rbr_channel_label: name for name, v in ds.variables.items() if "rbr_channel_label" in v.ncattrs()}
        assert set(labels) == {"conductivity_00", "temperature_00", "pressure_00"}
        assert np.allclose(ds[labels["temperature_00"]][:], fake.v[:, 1], rtol=1e-6)


def test_gen4_columns_in_an_order_other_than_the_instruments(monkeypatch, tmp_path):
    """The group stores temperature, then conductivity: each column keeps its own label and type."""
    fake = FakeGen4(n_samples=20)
    fake.objects["dataset_01/meta"] = build_meta(order=[1, 0, 2, 3])
    data, snap = _gen4_download(monkeypatch, tmp_path, fake)
    assert [c["label"] for c in snap["channel_list"]][:2] == ["conductivity_00", "temperature_00"]
    Gen4Driver(120).write_netcdf(data, _gen4_record(snap), tmp_path / "g.nc")
    with netCDF4.Dataset(tmp_path / "g.nc") as ds:
        by_label = {v.rbr_channel_label: v for v in ds.variables.values() if "rbr_channel_label" in v.ncattrs()}
        temp, cond = by_label["temperature_00"], by_label["conductivity_00"]
        assert temp.rbr_channel_type == CHANNELS[1][1] and cond.rbr_channel_type == CHANNELS[0][1]
        assert np.allclose(temp[:], fake.v[:, 0], rtol=1e-6)  # the first stored column is temperature
        assert temp.calibration_equation == "tmp" and temp.calibration_datetime.startswith("2024-09-30")


def test_gen4_names_what_it_did_not_convert(monkeypatch, tmp_path):
    fake = FakeGen4(n_samples=20)
    data, snap = _gen4_download(monkeypatch, tmp_path, fake)
    data["dataset_01/sch_fast/data"] = b"\x00" * 64  # a second schedule
    data["dataset_00/sch_ctd/data"] = b"\x00" * 64  # an older dataset
    warnings, _ = Gen4Driver(120).write_netcdf(data, _gen4_record(snap), tmp_path / "g.nc")
    w = next(w for w in warnings if "NOT converted" in w)
    assert "dataset_00/sch_ctd" in w and "dataset_01/sch_fast" in w and "dataset_01 schedule sch_ctd" in w


def test_events_are_placed_within_their_clock_run():
    t = np.array([T, T + 1000, R + 5000, R + 6000], np.int64)
    assert np.searchsorted(t, R + 5000) == 0  # what went wrong: the array is not sorted
    assert event_indices(t, [R + 5000]) == [2]
    assert event_indices(t, [T - 5000, T + 500, R + 200, R + 5500, R + 9000]) == [0, 1, 2, 3, 4]
    assert event_indices(np.array([R + 5000, R + 6000], np.int64), [T - 5000, R + 200]) == [0, 0]
    assert event_indices(np.array([T, T + 1000], np.int64), [T + 500, R + 200]) == [1, 2]  # no samples after it
    assert event_indices(np.zeros(0, np.int64), [T]) == [0]


def test_two_reset_runs_without_restart_events():
    """Two runs on the reset clock: only the last is re-timed; the first is omitted, not given duplicate times."""
    t = np.array([T, T + 1000, R + 5000, R + 6000, R + 5000, R + 6000], np.int64)
    tflags, segment = reset_segments(t, [])
    assert segment.tolist() == [0, 0, 1, 1, 2, 2]
    skew = (R + 5000 - (T + 15_000)) / 1000
    out, _, keep, notes = resolve_time_arrays(t, tflags, segment, skew, T + 60_000)
    assert ((out[keep] - T) // 1000).tolist() == [0, 1, 15, 16] and keep.tolist() == [1, 1, 0, 0, 1, 1]
    with pytest.raises(ValueError, match="not strictly increasing"):
        resolve_time_arrays(np.array([T, T + 1000, T + 1000], np.int64), np.zeros(3, np.uint8),
                            np.zeros(3, np.int32), 0.0, T + 60_000)


def _gen3_record():
    ch = {"index": 1, "type": "temp09", "label": "temperature_00", "status": 0, "userunits": "C",
          "equation": "tmp", "coefficients": {}}
    skew = {"n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001,
            "measured_at": "2024-10-01T00:05:00.000Z"}
    return {"offload_started": "2026-09-25T00:00:00.000Z", "id": {"model": "RBRsolo3", "version": "1.162",
            "serial": "233442", "fwtype": 104}, "clock_skew": skew,
            "snapshot_before": {"status": "logging", "sampling": {"mode": "continuous", "period": "1000"},
                                "memformat": {"type": "calbin00"}, "channel_list": [ch]}, "warnings": []}


def test_gen3_clock_reset_keeps_one_time_per_sample(tmp_path):
    """Samples on a set clock, a power loss, then the reset clock: events map to the right sample, the reset run
    is re-timed by the offload skew, and the time axis stays strictly increasing."""
    t = np.concatenate([T + 1000 * np.arange(10), R + 5000 + 1000 * np.arange(10)])
    events = make_event(T + 2500, 0x0F) + make_event(R + 7500, 0x0F)
    record = _gen3_record()
    skew = (R + 5000 - (T + 60_000)) / 1000  # logger minus UTC, measured at the offload
    record["clock_skew"]["skew_vs_host_s"] = skew
    record["offload_finished"] = "2024-10-01T00:05:00.000Z"
    warnings, t_out = Gen3Driver(104).write_netcdf(
        {"dataset1": make_records(t, np.full((20, 1), 20.0)), "dataset0": events}, record, tmp_path / "c.nc")
    assert len(t_out) == 20 and np.all(np.diff(t_out) > 0) and t_out[10] == T + 60_000
    assert any("clock restarted 1 time" in w for w in warnings)
    with netCDF4.Dataset(tmp_path / "c.nc") as ds:
        index = next(v for name, v in ds.variables.items() if name.startswith("event") and "index" in name)
        assert index[:].tolist() == [3, 13]


def test_duplicate_times_are_not_written(tmp_path):
    t = np.array([T, T + 1000, T + 1000, T + 2000], np.int64)
    with pytest.raises(ValueError, match="not strictly increasing"):
        Gen3Driver(104).write_netcdf({"dataset1": make_records(t, np.ones((4, 1)))}, _gen3_record(),
                                     tmp_path / "d.nc")
    assert not (tmp_path / "d.nc").exists()


@pytest.fixture
def once(monkeypatch, tmp_path):
    """Run `rbr-offload --once` against fake loggers; returns the exit status."""
    def go(fakes, *extra):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fakes[port])
        monkeypatch.setattr(cli, "rbr_ports", lambda: set(fakes))
        monkeypatch.setattr(cli, "ruskin_running", lambda: False)
        monkeypatch.setattr(cli, "SETTLE_S", 0.0)
        monkeypatch.setattr(cli, "ONCE_GRACE_S", 0.0)
        monkeypatch.setattr(cli, "measure_clock_skew", lambda link, **k: {
            "n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "measured_at": ""})
        try:
            cli.main([str(tmp_path), "--once", "--no-ntp", *extra])
        except SystemExit as exc:
            return exc.code
        finally:
            setup_logging(Console(stream=io.StringIO()))
        return 0
    return go


def test_exit_status_1_when_nothing_was_downloaded(once, tmp_path):
    assert once({"/dev/cu.A": FakeSolo("/dev/cu.A", fwtype=77, n_samples=50)}) == 1  # an unsupported model


def test_exit_status_1_when_the_netcdf_fails(once, tmp_path, monkeypatch):
    def broken(self, data, record, path):
        raise ValueError("simulated decoder failure")
    monkeypatch.setattr(L2Driver, "write_netcdf", broken)
    assert once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=50)}) == 1
    assert list((tmp_path / "raw").glob("100689_*.bin"))  # the download itself is saved


def test_exit_status_0_after_a_good_offload(once, tmp_path):
    assert once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=50)}) == 0 and list(tmp_path.glob("100689_*.nc"))


def test_a_configure_the_model_does_not_support_is_not_ready(once, tmp_path):
    """--configure on a duet (read-only so far): offloaded, not configured, so exit status 2."""
    assert once({"/dev/cu.D": FakeDuet("/dev/cu.D", n_samples=50)}, "--configure", "--yes", "--no-clock") == 2
    assert list(tmp_path.glob("081500_*.nc"))


def test_rebuild_skips_configure_reports_and_goes_on_after_a_bad_record(once, tmp_path):
    # a configured offload leaves SN_time.json and SN_time_configure.json side by side (this configure fails)
    assert once({"/dev/cu.A": FakeSolo("/dev/cu.A", n_samples=50)}, "--configure", "--yes", "--no-clock") == 2
    records = sorted((tmp_path / "raw").glob("100689_*.json"))
    assert [p.name.endswith("_configure.json") for p in records] == [False, True]
    out = tmp_path / "rebuilt"
    cli.main([str(out), "--rebuild", *map(str, records)])  # no KeyError, no exit status
    assert [p.name for p in out.glob("*.nc")] == [records[0].stem + ".nc"]

    bad = tmp_path / "raw" / "100689_bad.json"
    rec = json.loads(records[0].read_text())
    rec["raw"]["sha256"] = rec["datasets"]["dataset1"]["sha256"] = "0" * 64
    bad.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="1 of 3 record"):
        cli.main([str(out / "2"), "--rebuild", str(bad), *map(str, records)])
    assert [p.name for p in (out / "2").glob("*.nc")] == [records[0].stem + ".nc"]  # the good one still done
    setup_logging(Console(stream=io.StringIO()))
