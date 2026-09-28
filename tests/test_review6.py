# ruff: noqa: F811
"""Sixth review (PR #13, adversarial review of 7be286e, 2026-09-28): the reviewer's reproductions, each
asserting the intended behaviour (each failed on 7be286e), and tests for the other findings."""

import json
import struct

import netCDF4
import numpy as np
from fakelogger import SYNC_S, FakeConcerto3, FakeDuet, FakeSoloWritable, easyparse_image
from test_gen4 import FakeGen4
from test_incremental import _no_skew, _reads, _records, _skews, run
from test_offload import rig  # noqa: F401

from rbr_tpw import cli, gen4
from rbr_tpw import link as link_module
from rbr_tpw.configure import DeployConfig
from rbr_tpw.drivers import Gen4Driver, L2Driver
from rbr_tpw.equations import decode_l2
from rbr_tpw.link import Link
from rbr_tpw.ncwrite import skew_jump
from rbr_tpw.rawbin import TFLAG_RESET_CLOCK, OffloadView, SegmentStart, resolve_time_arrays


def test_1_later_deployment_is_found_after_the_first_had_two_offloads(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=21, n_samples=1000)
    run(rig)
    fake.grow(10)
    run(rig)  # deployment A, offload_index 1
    fake.erase_and_enable(SYNC_S + 3600)
    fake.grow(200)
    run(rig)  # deployment B, offload_index 0
    fake.grow(10)
    run(rig)
    r2, r3 = _records(rig, 21)[2:]
    assert r3["deployment"]["stem"] == r2["deployment"]["stem"] and r3["deployment"]["download"] == "incremental"


def test_2_adjacent_corrected_reset_runs_do_not_block_the_file():
    period, n0, n1, n2 = 500, 100, 100, 50
    t = np.concatenate([1_790_000_000_000 + period * np.arange(n0),
                        946_684_805_000 + period * np.arange(n1),  # run 1 on the reset clock
                        946_684_805_000 + period * np.arange(n2)]).astype(np.int64)  # run 2
    tf = np.zeros(t.size, np.uint8)
    tf[n0:] |= TFLAG_RESET_CLOCK
    real1 = int(t[n0 - 1]) + 60_000
    real2 = real1 + period * (n1 - 1) - 2000  # run 1's end, corrected by offload 1, lands 2 s after run 2 began
    views = [OffloadView(n0 + 10, 1, real1 + 6000, (946_684_805_000 - real1) / 1000),
             OffloadView(t.size, 2, real2 + n2 * period + 1000, (946_684_805_000 - real2) / 1000)]
    t_utc, _, keep, _ = resolve_time_arrays(t, tf, None, views=views,
                                            starts=[SegmentStart(n0, 0), SegmentStart(n0 + n1, 1)])
    assert np.all(np.diff(t_utc[keep]) > 0)


def _header_refresh(rig, serial):
    fake = rig.add("usbmodem1", serial=serial, n_samples=1000)
    run(rig)
    image = bytearray(fake.image)
    struct.pack_into("<I", image, 32, 7)  # a header word changes, the data does not
    fake.set_image(bytes(image))
    fake.grow(10)
    return fake


def test_3a_rebuild_from_a_record_before_a_header_refresh(rig, monkeypatch, tmp_path):
    _no_skew(monkeypatch)
    _header_refresh(rig, 22)
    run(rig)
    first = sorted((rig.tmp / "raw").glob("22_*.json"))[0]
    cli.main([str(tmp_path / "out"), "--rebuild", str(first)])  # SystemExit: checksum does not match the record


def test_3b_crash_after_a_header_refresh_keeps_the_deployment(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = _header_refresh(rig, 23)
    real = cli._write_json

    def crash(path, obj):
        raise OSError("power lost before the record")

    monkeypatch.setattr(cli, "_write_json", crash)
    run(rig)
    monkeypatch.setattr(cli, "_write_json", real)
    fake.grow(10)
    run(rig)
    r0, r1 = _records(rig, 23)
    assert r1["deployment"]["stem"] == r0["deployment"]["stem"]


def test_6a_changes_against_a_dead_deployment_do_not_leak(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=24, n_samples=100)
    run(rig)
    fake.erase_and_enable(SYNC_S + 3600)
    fake.endtime = "20261231000000"
    fake.grow(50)
    run(rig)
    r1 = _records(rig, 24)[1]
    assert not any("since the previous offload of deployment" in w for w in r1["warnings"])
    with netCDF4.Dataset(rig.tmp / f"{r1['deployment']['stem']}.nc") as ds:
        assert "since the previous offload" not in ds.warnings


def test_6b_gen3_verified_message_counts_new_bytes(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem201101", cls=FakeConcerto3, n_samples=300)
    run(rig)
    fake.datasets[1] = fake.datasets[1] + easyparse_image(50, 6, 1_790_000_000_000 + 300 * 1000)
    run(rig)
    line = next(ln for ln in rig.console_text().splitlines() if "verified against deployment" in ln)
    assert f": {50 * (8 + 4 * 6)} new bytes" in line


# --- further tests for the review's findings (not the reviewer's reproductions)


def test_14_ruskin_stop_sync_erase_enable_then_the_new_deployment_reads_incrementally(rig, monkeypatch):
    """Plan test 14 as written, with the commands Ruskin sends (its enable sequence, which --configure repeats):
    stop, clock set, erase, enable. The old deployment's files are left alone; the new one reads incrementally
    from its second offload on (finding 1 again, through the real write path)."""
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", cls=FakeSoloWritable, serial=25, n_samples=1000, memory_follows=True)
    run(rig)
    fake.grow(10)
    run(rig)  # deployment A, two offloads
    # the third offload of A, then stop, sync, erase, enable. The clock set is in the sequence, but its 20 ms check
    # is loosened: this test is about the memory, and a shared CI runner measured 60 ms (macOS, 2026-09-28)
    run(rig, deploy=DeployConfig(clock_tolerance_s=5.0), assume_yes=True)
    a_stem = _records(rig, 25)[0]["deployment"]["stem"]
    a_files = {p.name: p.read_bytes() for p in [rig.tmp / f"{a_stem}.nc", rig.tmp / "raw" / f"{a_stem}.bin"]}
    assert fake.state["status"] == "logging" and len(fake.image) == 520  # a fresh header and sync marker
    fake.grow(300)
    run(rig)  # B, a new deployment: full
    fake.grow(20)
    fake.commands.clear()
    run(rig)  # B again: incremental
    recs = _records(rig, 25)
    assert [(r["deployment"]["stem"] == a_stem, r["deployment"]["download"]) for r in recs] == [
        (True, "full"), (True, "incremental"), (True, "incremental"), (False, "full"), (False, "incremental")]
    assert recs[3]["deployment"]["stem"] == recs[4]["deployment"]["stem"]
    assert {p: (rig.tmp / p if p.endswith(".nc") else rig.tmp / "raw" / p).read_bytes() for p in a_files} == a_files
    assert _reads(fake)[0] == "read data 1 1720 0" and all(int(c.split()[4]) >= 1720 for c in _reads(fake)[1:])


def test_6_a_reset_run_that_ended_before_offload_2_keeps_offload_1s_skew(rig, monkeypatch):
    """Two power losses with an offload between them: each run on the reset clock is re-timed by the offload
    that saw it (one offload only would drop the first run), and the correction table says which."""
    fake = rig.add("usbmodem1", serial=26, n_samples=100)
    fake.append_event(0x0A, 5)  # power loss 1: the clock restarts at 2000-01-01T00:00:05
    fake.grow(50)
    real1 = SYNC_S + 60  # s since 2000 when the logger really came back the first time
    real2 = real1 + 3600  # and the second time
    _skews(monkeypatch, [5 - real1, 5 - real2])
    run(rig)
    fake.append_event(0x0A, 5)  # power loss 2
    fake.grow(30)
    run(rig)
    (nc,) = rig.tmp.glob("26_*.nc")
    with netCDF4.Dataset(nc) as ds:
        t = ds["time"][:]
        assert len(t) == 180  # nothing dropped
        e2000 = 946_684_800_000
        assert t[100] == e2000 + 1000 * real1 and t[150] == e2000 + 1000 * real2
        assert list(ds["time_correction_offload"][:]) == [0, 1]
        assert list(ds["time_correction_start_index"][:]) == [100, 150]
        assert list(ds["time_correction_end_index"][:]) == [150, 180]
        assert "time_correction_offload" in ds["time"].comment


def test_6_an_offload_that_saw_the_reset_event_but_no_new_sample_times_the_new_run():
    """Its skew is on the clock that restarted, so it may time the new run, never the one before the reset."""
    period, n0, n1, n2 = 500, 10, 5, 5
    real = 1_790_000_000_000
    t = np.concatenate([real + period * np.arange(n0),
                        946_684_805_000 + period * np.arange(n1),  # run 1 on a reset clock (reset event 1)
                        946_684_805_000 + period * np.arange(n2)]).astype(np.int64)  # run 2 (reset event 2)
    tf = np.zeros(t.size, np.uint8)
    tf[n0:] |= TFLAG_RESET_CLOCK
    skew2 = (946_684_805_000 - (real + 3_600_000)) / 1000
    view = OffloadView(samples_seen=n0 + n1, events_seen=3, unix_ms=real + 3_600_000 + 1000, skew_s=skew2)
    t_utc, _, keep, notes = resolve_time_arrays(t, tf, None, views=[view],
                                                starts=[SegmentStart(n0, 1), SegmentStart(n0 + n1, 2)])
    assert keep.tolist() == [True] * n0 + [False] * n1 + [True] * n2  # run 1 dropped: no offload on its clock
    assert t_utc[n0 + n1] == real + 3_600_000


def test_11_duet_is_read_in_full_and_grows_one_file(rig, monkeypatch):
    """Sectioned headers (715 bytes on the duet) are not read incrementally yet; the file still grows."""
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem2101", cls=FakeDuet, n_samples=2000)
    run(rig)
    fake.grow(100)
    fake.commands.clear()
    run(rig)
    r0, r1 = _records(rig, "081500")
    assert r1["deployment"]["download"] == "full-verified" and r1["deployment"]["stem"] == r0["deployment"]["stem"]
    assert r1["deployment"]["segment"] == {**r1["deployment"]["segment"], "offset": len(fake.image) - 1200,
                                           "bytes": 1200}
    assert r1["deployment"]["header_bytes"] == L2Driver.header_length(fake.image) != 512
    assert _reads(fake)[0] == "read data 1 512 0"  # the full read's header block, not a tail check
    (nc,) = rig.tmp.glob("081500_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert list(ds["offload_samples"][:]) == [2000, 2100] and len(ds["time"]) == 2100


def test_11_decode_l2_set_end_byte():
    image = FakeDuet("/dev/null", n_samples=4).image
    d = decode_l2(image, 3)
    hl = L2Driver.header_length(image)
    body = hl + 8  # the time-sync event record follows the header
    assert d.set_end_byte.tolist() == [body + 12 * (i + 1) for i in range(4)]
    assert d.events[0].offset == hl and d.events[0].size == 8


def test_6c_skews_on_different_references_are_not_compared():
    base = {"offload_started": "2026-09-25T00:00:00.000Z",
            "clock_skew": {"n": 3, "skew_vs_host_s": 0.0, "uncertainty_s": 0.01}}
    with_ntp = {**base, "host_ntp": {"offset_s": 3.0, "uncertainty_s": 0.02}}  # host 3 s behind UTC
    later = {**base, "offload_started": "2026-09-25T01:00:00.000Z"}
    no_ntp = {**later, "host_ntp": {"error": "disabled"}}
    assert skew_jump(with_ntp, no_ntp) is None  # would read as a 3 s jump an hour apart
    jump = skew_jump(with_ntp, {**later, "host_ntp": {"offset_s": 3.0, "uncertainty_s": 0.02}})
    assert jump[0] == 0.0 and abs(jump[1] - (2.0 + 50e-6 * 3600 + 0.03 + 0.03)) < 1e-9


def test_6e_gen4_earlier_offloads_count_their_own_samples_and_events(monkeypatch, tmp_path):
    fake = FakeGen4(n_samples=20)
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link("/dev/cu.gen4")
    data = gen4.download(link, tmp_path / "part", "210000")
    snap = Gen4Driver(120).snapshot(link)
    key = next(k for k in data if k.endswith("/data"))
    rec_size = len(data[key]) // 20
    ident = {"model": "RBRconcerto3", "version": "2.1.0", "serial": "210000", "fwtype": 120}
    def info(n):
        return {"file": "raw/g.bin", "bytes": n, "sha256": "-"}

    early = {"offload_started": "2026-09-25T00:00:00.000Z", "id": ident, "snapshot_before": snap, "warnings": [],
             "datasets": {key: info(5 * rec_size), key.rsplit("/", 2)[0] + "/events": info(0)}}
    latest = {**early, "offload_started": "2026-09-25T01:00:00.000Z",
              "datasets": {k: info(len(v)) for k, v in data.items()}}
    Gen4Driver(120).write_netcdf(data, [early, latest], tmp_path / "g.nc")
    with netCDF4.Dataset(tmp_path / "g.nc") as ds:
        assert list(ds["offload_samples"][:]) == [5, 20]


def test_3_rebuild_the_latest_record_with_bytes_past_it(rig, monkeypatch, tmp_path):
    """A crash after the append left bytes past the latest record: --rebuild still works (the next offload
    truncates them)."""
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=27, n_samples=500)
    run(rig)
    fake.grow(10)
    run(rig)
    recs = sorted((rig.tmp / "raw").glob("27_*.json"))
    stem = json.loads(recs[-1].read_text())["deployment"]["stem"]
    with open(rig.tmp / "raw" / f"{stem}.bin", "ab") as f:
        f.write(b"\xaa" * 40)
    cli.main([str(tmp_path / "out"), "--rebuild", str(recs[-1])])
    with netCDF4.Dataset(tmp_path / "out" / f"{stem}.nc") as ds:
        assert len(ds["time"]) == 510


def test_3_header_change_is_recorded_once_and_the_image_only_grows(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = _header_refresh(rig, 28)
    run(rig)
    fake.grow(10)
    run(rig)
    r0, r1, r2 = _records(rig, 28)
    image = (rig.tmp / "raw" / f"{r0['deployment']['stem']}.bin").read_bytes()
    assert image[:512] != fake.image[:512] and image[512:] == fake.image[512:]  # the header as first downloaded
    assert r1["deployment"]["header_changed"] == [32, 33, 34, 35]
    assert bytes.fromhex(r1["deployment"]["header_hex"]) == fake.image[:512]
    assert r2["deployment"]["header_changed"] == [] and r2["deployment"]["header_hex"] == r1["deployment"]["header_hex"]
