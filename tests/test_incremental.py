# ruff: noqa: F811  (the `rig` fixture is imported from test_offload and named by every test)
"""Incremental offloads: one growing NetCDF per deployment, the raw image extended by each offload, the
per-offload state as a series, and the checks that make Ruskin's stop, sync, erase and enable safe
(PLAN-incremental-offload.md, tests 1-16)."""

import json
import struct
import threading
import time

import netCDF4
import numpy as np
from fakelogger import SYNC_S, FakeConcerto3, easyparse_image
from test_offload import fast_skew, rig  # noqa: F401  (a fixture)

from rbr_tpw import cli

HEADER = 512
EVENT = 8


def _no_skew(monkeypatch):
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))


def _skews(monkeypatch, values):
    """measure_clock_skew stand-in answering the given skews (logger minus host, s), one per offload."""
    it = iter(values)

    def measure(link, reps=3, max_seconds=15.0, clock=None):
        s = next(it)
        return {"n": 3, "polls": 9, "skew_vs_host_s": s, "uncertainty_s": 0.003, "spread_s": 0.001,
                "individual_s": [s] * 3, "measured_at": "2026-09-25T00:00:00.000+00:00"}

    monkeypatch.setattr(cli, "measure_clock_skew", measure)


def run(rig, **kw):
    """One offload pass. A later pass waits for the next second, so its record and stem get their own names
    (on a bench, unplugging and replugging takes longer than that)."""
    if getattr(rig, "runs", 0):
        time.sleep(1.05)
    rig.runs = getattr(rig, "runs", 0) + 1
    cli.run(rig.settings(**kw), once=True, port=None)


def _records(rig, sn):
    return [json.loads(p.read_text()) for p in sorted((rig.tmp / "raw").glob(f"{sn}_*.json"))
            if not p.name.endswith("_configure.json")]


def _reads(fake):
    return [c for c in fake.commands if c.startswith("read data")]


def test_second_offload_reads_only_the_new_bytes(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=100689, n_samples=3000)
    run(rig)
    (nc,) = rig.tmp.glob("100689_*.nc")
    stem = nc.stem
    old_len = len(fake.image)
    fake.commands.clear()
    fake.grow(500)
    run(rig)

    reads = _reads(fake)
    assert reads[0] == "read data 1 512 0"  # the header
    assert reads[1] == f"read data 1 {old_len} 0"  # the tail check: the whole (small) image
    assert all(int(c.split()[4]) >= old_len for c in reads[2:]) and len(reads) > 2  # then only new bytes
    assert [p.name for p in rig.tmp.glob("*.nc")] == [f"{stem}.nc"]
    r0, r1 = _records(rig, 100689)
    assert r0["deployment"]["stem"] == stem and r0["deployment"]["download"] == "full"
    assert r1["deployment"] == {**r1["deployment"], "stem": stem, "download": "incremental", "offload_index": 1}
    assert r1["deployment"]["segment"] == {**r1["deployment"]["segment"], "offset": old_len, "bytes": 2000}
    assert r1["deployment"]["tail_check"] == {"offset": 0, "bytes": old_len, "ok": True}
    assert (rig.tmp / "raw" / f"{stem}.bin").read_bytes() == fake.image
    assert r1["raw"]["bytes"] == len(fake.image) and r1["raw"]["file"] == f"raw/{stem}.bin"
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 3500 and np.all(np.diff(ds["time"][:]) == 500)
        assert len(ds["offload_time"]) == 2 and ds.deployment_offloads == 2 and ds.deployment_stem == stem
        assert list(ds["offload_samples"][:]) == [3000, 3500]
        assert list(ds["offload_segment_bytes"][:]) == [old_len, 2000]
        assert list(ds["offload_image_bytes"][:]) == [old_len, len(fake.image)]
        assert list(ds["logger_status"][:]) == ["logging", "logging"]
        assert ds["battery_voltage"][:].tolist() == [3.634, 3.634]
        assert ds.history.count("\n") == 2 and "offload 1 from usbmodem1, incremental, 2000 bytes read" in ds.history
    assert "incremental: 2000 new bytes from offset" in rig.console_text()
    assert not list((rig.tmp / "raw" / ".partial").glob("*")) if (rig.tmp / "raw" / ".partial").is_dir() else True


def test_nothing_new_still_records_the_offload(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=7, n_samples=1000)
    run(rig)
    fake.commands.clear()
    run(rig)
    assert _reads(fake) == ["read data 1 512 0", f"read data 1 {len(fake.image)} 0"]
    r0, r1 = _records(rig, 7)
    assert r1["deployment"]["segment"]["bytes"] == 0 and r1["deployment"]["download"] == "incremental"
    (nc,) = rig.tmp.glob("7_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["offload_time"]) == 2 and len(ds["time"]) == 1000
    assert "incremental: 0 new bytes" in rig.console_text()


def test_erase_and_enable_starts_a_new_deployment(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=8, n_samples=1000)
    run(rig)
    (nc0,) = rig.tmp.glob("8_*.nc")
    before = nc0.read_bytes(), (rig.tmp / "raw" / f"{nc0.stem}.bin").read_bytes()
    fake.erase_and_enable(SYNC_S + 3600)
    fake.grow(200)
    run(rig)
    ncs = sorted(rig.tmp.glob("8_*.nc"))
    assert len(ncs) == 2 and ncs[0] == nc0
    assert (nc0.read_bytes(), (rig.tmp / "raw" / f"{nc0.stem}.bin").read_bytes()) == before
    r0, r1 = _records(rig, 8)
    assert r1["deployment"]["stem"] == ncs[1].stem != r0["deployment"]["stem"]
    assert r1["deployment"]["download"] == "full" and r1["deployment"]["offload_index"] == 0
    assert "does not hold deployment" in rig.console_text()
    with netCDF4.Dataset(ncs[1]) as ds:
        assert len(ds["time"]) == 200 and len(ds["offload_time"]) == 1


def test_image_longer_than_its_record_is_truncated_then_extended(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=9, n_samples=1000)
    run(rig)
    (nc,) = rig.tmp.glob("9_*.nc")
    image = rig.tmp / "raw" / f"{nc.stem}.bin"
    with open(image, "ab") as f:
        f.write(b"\xaa" * 100)  # a crash between the append and the record
    fake.grow(10)
    run(rig)
    assert "truncating it to" in rig.console_text() and image.read_bytes() == fake.image
    assert _records(rig, 9)[1]["deployment"]["stem"] == nc.stem


def test_damaged_image_starts_a_new_deployment_and_is_left_alone(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=10, n_samples=1000)
    run(rig)
    (nc,) = rig.tmp.glob("10_*.nc")
    image = rig.tmp / "raw" / f"{nc.stem}.bin"
    damaged = image.read_bytes()[:-10]
    image.write_bytes(damaged)
    fake.grow(10)
    run(rig)
    assert "does not match its record" in rig.console_text() and image.read_bytes() == damaged
    r0, r1 = _records(rig, 10)
    assert r1["deployment"]["stem"] != r0["deployment"]["stem"] and r1["deployment"]["download"] == "full"


def test_ctrl_c_mid_segment_then_resume(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=77, n_samples=1000, bytes_per_s=300_000)
    run(rig)
    (nc,) = rig.tmp.glob("77_*.nc")
    old_len = len(fake.image)
    fake.grow(150_000)
    s = rig.settings()
    t = threading.Thread(target=cli._worker, args=("/dev/cu.usbmodem1", s))
    t.start()
    part = rig.tmp / "raw" / ".partial" / f"77.{nc.stem}.{old_len}.part"
    deadline = time.monotonic() + 30
    while not (part.exists() and part.stat().st_size >= 68_000) and time.monotonic() < deadline:
        time.sleep(0.01)
    s.stop.set()
    t.join(10)
    size = part.stat().st_size
    assert size % 68_000 == 0 and 0 < size < 600_000
    s.stop.clear()
    cli._worker("/dev/cu.usbmodem1", s)
    assert [p.name for p in rig.tmp.glob("77_*.nc")] == [nc.name]
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 151_000
    assert f"resuming from {old_len + size} bytes" in rig.session_text()
    assert (rig.tmp / "raw" / f"{nc.stem}.bin").read_bytes() == fake.image


def _reset_clock_deployment(rig, n_before=100, n_after=50):
    """A solo that lost power: a restart event on the reset clock (2000-01-01T00:00:05) after `n_before`
    samples, then `n_after` more. Returns (fake, the reset run's skew vs the host, s)."""
    fake = rig.add("usbmodem1", serial=11, n_samples=n_before)
    fake.append_event(0x0A, 5)
    fake.grow(n_after)
    real_reset_s = SYNC_S + 60  # when the logger really came back, s since 2000
    return fake, 5 - real_reset_s


def test_reset_run_spanning_two_offloads_gets_one_skew(rig, monkeypatch):
    fake, skew = _reset_clock_deployment(rig)
    _skews(monkeypatch, [skew, skew + 0.010])  # 10 ms of drift by the second offload
    run(rig)
    fake.grow(20)
    run(rig)
    (nc,) = rig.tmp.glob("11_*.nc")
    with netCDF4.Dataset(nc) as ds:
        t = ds["time"][:]
        assert len(t) == 170
        steps = np.diff(t)
        assert steps[99] == 60_000 - 49_500 - 10  # the reset gap, re-timed with the latest skew (10 ms more lead)
        assert np.all(np.delete(steps, 99) == 500)  # one correction for the whole run: no step at the boundary
        assert list(ds["clock_set_detected"][:]) == [0, 0]
        assert "offload 2 of 2" in ds.warnings


def test_sync_in_ruskin_after_a_reset_run_keeps_the_earlier_skew(rig, monkeypatch):
    """Stop, then sync the clock in Ruskin: the next offload's skew is on the new clock. The reset run keeps
    the earlier offload's correction; samples logged between that offload and the stop have no time."""
    fake, skew = _reset_clock_deployment(rig)
    _skews(monkeypatch, [skew, 0.0])
    run(rig)
    fake.grow(10)  # logged before the stop: only the post-sync offload sees them
    fake.append_event(0x02, 40)  # `stop` in Ruskin, on the reset clock
    fake.status = "stopped"
    run(rig)
    (nc,) = rig.tmp.glob("11_*.nc")
    with netCDF4.Dataset(nc) as ds:
        t = ds["time"][:]
        assert len(t) == 150  # 100 good, 50 re-timed by offload 1, 10 dropped
        assert np.all(np.diff(t)[100:] == 500) and np.diff(t)[99] == 60_000 - 49_500
        assert list(ds["clock_set_detected"][:]) == [0, 1]
        assert list(ds["logger_status"][:]) == ["logging", "stopped"]
        assert "10 samples on a reset clock with no recoverable time are omitted" in ds.warnings
        assert "offload 1 of 2" in ds.warnings
    r1 = _records(rig, 11)[1]
    assert any("status logging -> stopped" in w for w in r1["warnings"])
    assert any("clock was set or reset" in w for w in r1["warnings"])
    assert any("disable_command_received" in w for w in r1["warnings"])


def test_full_download_verifies_the_deployment_image(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=12, n_samples=1000)
    run(rig)
    old_len = len(fake.image)
    fake.grow(100)
    fake.commands.clear()
    run(rig, full_download=True)
    assert _reads(fake)[0] == "read data 1 512 0" and _reads(fake)[1].endswith(" 512")  # from the start
    r1 = _records(rig, 12)[1]
    assert r1["deployment"]["download"] == "full-verified" and r1["deployment"]["segment"]["offset"] == old_len
    assert "verified against deployment" in rig.console_text()
    # a memory that does not extend the image: new deployment, from the full read already made
    image = bytearray(fake.image)
    image[old_len - 3] ^= 0x01
    fake.set_image(bytes(image))
    fake.commands.clear()
    run(rig, full_download=True)
    r2 = _records(rig, 12)[2]
    assert r2["deployment"]["download"] == "full" and r2["deployment"]["stem"] != r1["deployment"]["stem"]
    assert f"differs from the deployment image at byte {old_len - 3}" in rig.console_text()
    assert len(_reads(fake)) == 1 + (len(fake.image) + 67_999 - 512) // 68_000  # read once, not twice


def test_rebuild_a_deployment_from_any_of_its_records(rig, monkeypatch, tmp_path):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=13, n_samples=1000)
    run(rig)
    fake.grow(100)
    run(rig)
    recs = sorted(p for p in (rig.tmp / "raw").glob("13_*.json"))
    out = tmp_path / "rebuilt"
    cli.main([str(out), "--rebuild", str(recs[0]), str(recs[1])])
    (nc,) = out.glob("*.nc")
    (orig,) = rig.tmp.glob("13_*.nc")
    assert nc.name == orig.name
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 1100 and len(ds["offload_time"]) == 2


def test_header_change_with_the_data_intact_is_refreshed_not_split(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=14, n_samples=1000)
    run(rig)
    image = bytearray(fake.image)
    struct.pack_into("<I", image, 32, 7)  # the status word (0xFFFFFFFF on every logger seen)
    fake.set_image(bytes(image))
    fake.grow(10)
    run(rig)
    r0, r1 = _records(rig, 14)
    assert r1["deployment"]["stem"] == r0["deployment"]["stem"]
    assert r1["deployment"]["header_changed"] == [32, 33, 34, 35]
    assert any("memory header changed" in w for w in r1["warnings"])
    assert (rig.tmp / "raw" / f"{r0['deployment']['stem']}.bin").read_bytes() == fake.image


def test_settings_change_is_reported(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=15, n_samples=100)
    run(rig)
    fake.endtime = "20261231000000"
    run(rig)
    r1 = _records(rig, 15)[1]
    assert any("endtime 20991231235959 -> 20261231000000" in w for w in r1["warnings"])
    assert "WARNING" in rig.console_text() and "endtime" in rig.console_text()


def test_concerto3_full_downloads_grow_one_file(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=300)
    run(rig)
    (nc,) = rig.tmp.glob("233442_*.nc")
    t0 = 1_790_000_000_000
    fake.datasets[1] = fake.datasets[1] + easyparse_image(50, 6, t0 + 300 * 1000)
    fake.datasets[0] = fake.datasets[0] + fake.datasets[0][:16]  # one more event record
    run(rig)
    assert [p.name for p in rig.tmp.glob("233442_*.nc")] == [nc.name]
    r0, r1 = _records(rig, 233442)
    assert r1["deployment"]["download"] == "full-verified" and r1["deployment"]["stem"] == r0["deployment"]["stem"]
    assert r1["deployment"]["identity_dataset"] == "dataset2"
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 350 and len(ds["offload_time"]) == 2 and list(ds["offload_samples"][:]) == [300, 350]
        assert "event_payload" in ds.variables and "energy_used_marker" in ds.variables
    # a rewritten deployment header is another deployment
    fake.datasets[2] = bytes(1204)
    run(rig)
    assert len(list(rig.tmp.glob("233442_*.nc"))) == 2 and "does not hold deployment" in rig.console_text()


def test_old_style_records_are_ignored_and_a_new_deployment_starts(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=16, n_samples=100)
    run(rig)
    (rec,) = (rig.tmp / "raw").glob("16_*.json")
    old = json.loads(rec.read_text())
    del old["deployment"]
    rec.write_text(json.dumps(old))
    fake.grow(10)
    run(rig)
    assert len(list(rig.tmp.glob("16_*.nc"))) == 2
    assert _records(rig, 16)[1]["deployment"]["download"] == "full"


def test_two_offload_file_is_cf_compliant(rig, monkeypatch, cf_problems):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=17, n_samples=200)
    run(rig)
    fake.grow(20)
    run(rig)
    (nc,) = rig.tmp.glob("17_*.nc")
    assert not cf_problems(nc)
