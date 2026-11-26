"""--configure and incremental offloads on Gen3 loggers (fwtype 104, the RBRconcerto3), through the real Link: Ruskin
2.26.1's write sequence (configure._apply_gen3), the sub-second periods the logger lists, the UTC offset a clock
write erases (L3 reference 4.1.1), and new-bytes-only reads of datasets 2, 0 and 1."""

import json
import threading
import time

import netCDF4
import numpy as np
import pytest
from fakelogger import FakeConcerto3, easyparse_event
from test_offload import fast_skew, rig  # noqa: F401  (rig is a fixture)

from rbr_tpw import cli
from rbr_tpw import configure as cf
from rbr_tpw import link as link_module
from rbr_tpw.configure import ConfigError, DeployConfig, configure
from rbr_tpw.drivers import driver_for
from rbr_tpw.link import Link


def _skew_of(fake):
    def skew(link, reps=3, **k):
        return {"n": 3, "polls": 9, "skew_vs_host_s": fake.skew_s + fake.offset_s, "uncertainty_s": 0.003,
                "spread_s": 0.001, "individual_s": [0.0] * 3, "measured_at": ""}
    return skew


@pytest.fixture
def connect(monkeypatch):
    def make(fake):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
        link = Link(fake.port)
        drv = driver_for(104)
        drv.serial = fake.serial
        monkeypatch.setattr(cf, "measure_clock_skew", _skew_of(fake))
        return link, drv
    return make


def run_configure(link, drv, fake, cfg):
    return configure(link, fake.serial, cfg, 0.0, log=lambda *a: None, fwtype=104, driver=drv)


def writes(fake):
    return [w for w in fake.writes if not w.startswith("lock")]


def test_gen3_enable_writes_what_ruskin_writes(connect):
    """SN233442 (Ruskin, 2026-09-11/12): disable, `clock datetime = X, offsetfromutc = 0.0`, the period, twist
    activation, then `enable erasememory = true, memformat = CALBIN00` -> `enable status = gated, warning = none`."""
    fake = FakeConcerto3("/dev/cu.c3", n_samples=300)
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(period_ms=31, twist_activation=True))
    w = writes(fake)
    assert w[0] == "disable"
    assert w[1].startswith("clock datetime = ") and w[1].endswith(", offsetfromutc = 0.0")
    assert w[2:] == ["sampling period = 31", "twistactivation enabled = true",
                     "enable erasememory = true, memformat = CALBIN00"]
    back = report["readback"]
    assert back["status"] == "gated" and back["sampling"]["period"] == "31" and back["offsetfromutc"] == "+0.00"
    assert back["twistactivation"]["enabled"] == "true" and back["meminfo"]["used"] == 0
    erase = next(x for x in report["steps"] if x["step"] == "erase")
    assert erase["used_before"] > 0 == erase["used"]
    assert fake.locked


def test_gen3_periods_come_from_the_logger(connect):
    fake = FakeConcerto3("/dev/cu.c3", n_samples=30)
    link, drv = connect(fake)
    for bad in (30, 50, 167):  # below userperiodlimit; not listed; not listed
        with pytest.raises(ConfigError, match="period"):
            run_configure(link, drv, fake, DeployConfig(period_ms=bad))
    assert writes(fake) == []
    run_configure(link, drv, fake, DeployConfig(period_ms=63))  # 16 Hz, listed
    assert "sampling period = 63" in writes(fake)


def test_gen3_cleared_and_left_stopped(connect, monkeypatch):
    monkeypatch.setattr(cf, "IDLE_SAVE_S", 0.0)
    fake = FakeConcerto3("/dev/cu.c3", n_samples=300)
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(enable=False))
    assert writes(fake)[-2:] == ["permit command = memclear", "memclear"]
    assert report["readback"]["status"] == "stopped" and report["readback"]["meminfo"]["used"] == 0
    assert not any(w.startswith("enable") for w in fake.writes)


def test_gen3_offset_is_zeroed_when_the_clock_is_left_alone(connect):
    """Without a clock write the offset may still read `unknown` (a clock set earlier without it erased it)."""
    fake = FakeConcerto3("/dev/cu.c3", n_samples=30)
    fake.state["offsetfromutc"] = "unknown"
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(set_clock=False))
    assert "clock offsetfromutc = 0.0" in writes(fake) and report["readback"]["offsetfromutc"] == "+0.00"
    assert not any(w.startswith("clock datetime") for w in fake.writes)


def test_gen3_start_and_end_are_written_together_as_ruskin_does(connect):
    fake = FakeConcerto3("/dev/cu.c3", n_samples=30)
    link, drv = connect(fake)
    run_configure(link, drv, fake, DeployConfig(end="2099-06-30T00:00:00Z"))
    assert "deployment endtime = 20990630000000, starttime = 20000101000000" in writes(fake)


def test_gen3_battery_counter_is_refused(connect):
    fake = FakeConcerto3("/dev/cu.c3", n_samples=30)
    link, drv = connect(fake)
    with pytest.raises(ConfigError, match="battery counter of fwtype 104"):
        run_configure(link, drv, fake, DeployConfig(fresh_battery=True))
    assert writes(fake) == []


# --- incremental offloads


def _no_skew(monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))


def _records(rg, sn):
    return [json.loads(p.read_text()) for p in sorted((rg.tmp / "raw").glob(f"{sn}_*.json"))
            if not p.name.endswith("_configure.json")]


def test_gen3_second_offload_reads_only_the_new_bytes(rig, monkeypatch):  # noqa: F811
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=20_000)
    cli.run(rig.settings(), once=True, port=None)
    old1, old0 = len(fake.datasets[1]), len(fake.datasets[0])
    fake.grow(400)
    fake.datasets[0] = fake.datasets[0] + easyparse_event(fake.t0 + 20_400_000, 0x18)
    fake.commands.clear()
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    reads = [c for c in fake.commands if c.startswith("readdata")]
    ds1 = [c for c in reads if c.endswith("dataset = 1")]
    offsets = [int(c.split("offset = ")[1].split(",")[0]) for c in ds1]
    r0, r1 = _records(rig, 233442)
    d = r1["deployment"]
    tail = d["tail_check"]
    assert tail == {"head_bytes": 68000, "offset": old1 - 68000, "bytes": 68000, "ok": True}
    assert offsets[:2] == [0, old1 - 68000] and all(o >= old1 for o in offsets[2:]) and len(ds1) > 2
    assert d["download"] == "incremental" and d["stem"] == r0["deployment"]["stem"]
    assert d["segment"] == {**d["segment"], "offset": old1, "bytes": 400 * 32}
    assert r1["datasets"]["dataset0"]["offset"] == old0 and r1["datasets"]["dataset0"]["segment_bytes"] == 16
    assert d["bytes_read"] < 0.3 * len(fake.datasets[1])  # 2 x 68 kB + new + datasets 2 and 0, of 653 kB
    stem = d["stem"]
    assert (rig.tmp / "raw" / f"{stem}_dataset1.bin").read_bytes() == fake.datasets[1]
    assert (rig.tmp / "raw" / f"{stem}_dataset0.bin").read_bytes() == fake.datasets[0]
    (nc,) = rig.tmp.glob("233442_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 20_400 and np.all(np.diff(ds["time"][:]) == 1000)
        assert list(ds["offload_samples"][:]) == [20_000, 20_400]


def test_gen3_changed_header_or_data_is_another_deployment(rig, monkeypatch):  # noqa: F811
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=300)
    cli.run(rig.settings(), once=True, port=None)
    blob = bytearray(fake.datasets[1])
    blob[40] ^= 1  # inside the first sample records
    fake.datasets[1] = bytes(blob)
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    r0, r1 = _records(rig, 233442)
    assert r1["deployment"]["stem"] != r0["deployment"]["stem"]
    assert "dataset1 differs from the deployment image at byte 40" in rig.console_text()
    fake.datasets[2] = bytes(1204)
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    assert len({r["deployment"]["stem"] for r in _records(rig, 233442)}) == 3
    assert "the dataset2 header differs" in rig.console_text()


def test_cli_configures_a_concerto3_gated_and_ready(rig, monkeypatch):  # noqa: F811
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=200)
    monkeypatch.setattr(cf, "measure_clock_skew", _skew_of(fake))
    cli.run(rig.settings(deploy=DeployConfig(period_ms=31, twist_activation=True), assume_yes=True), once=True,
            port=None)
    text = rig.console_text()
    assert "CONFIGURE FAILED" not in text and "NOT READY" not in text
    assert "enabled and gated" in text and fake.state["status"] == "gated" and fake.state["period"] == 31
