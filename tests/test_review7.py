# ruff: noqa: F811
"""Seventh review (PR #13, follow-up adversarial review of 08fd3a9, 2026-09-28): the reviewer's reproductions
(assertions describe the intended behaviour; all but the full-download control failed on 08fd3a9), and tests
for the fixes."""

import datetime as dt
import struct

import netCDF4
import numpy as np
import pytest
import test_gen4 as g4_fixture
from fakelogger import SYNC_S, solo_image
from test_incremental import _no_skew, _records, _skews, run
from test_offload import rig  # noqa: F401

from rbr_tpw import cli
from rbr_tpw.rawbin import EPOCH2000_MS, TFLAG_RESET_CLOCK, OffloadView, SegmentStart, resolve_time_arrays


def set_offload_clock(monkeypatch, fake, unix_ms, skew_s=0):
    host = dt.datetime.fromtimestamp(unix_ms / 1000, dt.UTC)
    logger = host + dt.timedelta(seconds=skew_s)
    monkeypatch.setattr(cli, "utcnow", lambda: host)
    monkeypatch.setattr(fake, "now", lambda: logger)


@pytest.mark.parametrize("full_download", [False, True])
def test_redeployment_with_repeated_error_readings_is_a_new_deployment(rig, monkeypatch, full_download):
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=301, n_samples=0)
    error = struct.pack("<I", 0xF6000001)
    fake.set_image(solo_image(0, SYNC_S, 301) + struct.pack("<I", 0x20000000) * 1000 + error * 19_000)
    set_offload_clock(monkeypatch, fake, EPOCH2000_MS + (SYNC_S + 11_000) * 1000)
    run(rig)
    (old_nc,) = rig.tmp.glob("301_*.nc")
    saved_nc = old_nc.read_bytes()
    first = _records(rig, 301)[0]
    old_raw = rig.tmp / first["raw"]["file"]
    saved_raw = old_raw.read_bytes()

    # Erase+enable happened between offloads, and the new deployment has
    # acquired more samples than the old one. Its last 68 kB repeats the
    # same valid logger error code, while its initial time anchor differs.
    fake.erase_and_enable(SYNC_S + 86400)
    fake.set_image(fake.image + struct.pack("<I", 0x21000000) * 1000 + error * 20_000)
    set_offload_clock(monkeypatch, fake, EPOCH2000_MS + (SYNC_S + 86400 + 11_000) * 1000)
    run(rig, full_download=full_download)
    latest = _records(rig, 301)[1]
    assert latest["deployment"]["stem"] != first["deployment"]["stem"], latest["deployment"]
    assert old_nc.read_bytes() == saved_nc and old_raw.read_bytes() == saved_raw
    with netCDF4.Dataset(rig.tmp / f"{latest['deployment']['stem']}.nc") as ds:
        assert ds["time"][0] == EPOCH2000_MS + (SYNC_S + 86400) * 1000
        assert ds["temperature_raw"][0] == 0x21000000


@pytest.mark.parametrize("last_event,n_after,wait_s", [(0x0A, 1, 1), (0x06, 50, 3000)])
def test_reset_at_end_of_image_does_not_retime_the_previous_clock_run(
    rig, monkeypatch, last_event, n_after, wait_s
):
    fake = rig.add("usbmodem1", serial=302, n_samples=0)
    fake.set_image(solo_image(2, SYNC_S, 302, period_ms=60_000))
    original_replies = fake.replies

    def replies():
        result = original_replies()
        result["sampling"] = "sampling mode = continuous, period = 60000"
        return result

    monkeypatch.setattr(fake, "replies", replies)
    fake.append_event(0x0A, 5)
    fake.grow(n_after)
    real1 = SYNC_S + 120
    real2 = real1 + 3600
    _skews(monkeypatch, [5 - real1, 5 - real2])
    set_offload_clock(monkeypatch, fake, EPOCH2000_MS + (real1 + (n_after - 1) * 60 + 1) * 1000, 5 - real1)
    run(rig)
    (nc,) = rig.tmp.glob("302_*.nc")
    with netCDF4.Dataset(nc) as ds:
        first_times = ds["time"][:].copy()

    # A second power loss writes its restart event, but no sample has
    # followed it yet (e.g. reconnect before the next scheduled sample).
    fake.append_event(last_event, 5)
    if last_event == 0x06:  # restart_failed_rtc_invalid: sampling has not restarted
        fake.status = "stopped"
    set_offload_clock(monkeypatch, fake, EPOCH2000_MS + (real2 + wait_s) * 1000, 5 - real2)
    run(rig)
    with netCDF4.Dataset(nc) as ds:
        later_times = ds["time"][:].copy()
        assert np.array_equal(later_times, first_times), later_times[2] - first_times[2]
        assert ds["time_correction_offload"][:].tolist() == [0]


def test_new_gen4_dataset_gets_a_new_deployment_file(rig, monkeypatch):
    _no_skew(monkeypatch)
    fake = rig.add("gen4", cls=lambda port: g4_fixture.FakeGen4(n_samples=20))
    run(rig)
    first = _records(rig, 210000)[0]
    (old_nc,) = rig.tmp.glob("210000_*.nc")
    saved = old_nc.read_bytes()

    # Gen4 keeps one dataset per deployment, so the old dataset can still
    # be present when the next enable opens a second one.
    monkeypatch.setattr(g4_fixture, "ENABLE_MS", g4_fixture.ENABLE_MS + 86400_000)
    meta = g4_fixture.build_meta(dataset="dataset_02")
    data, _, _ = g4_fixture.build_data(30)
    events = g4_fixture.build_events()
    fake.objects.update({"dataset_02/meta": meta, "dataset_02/events": events,
                         "dataset_02/sch_ctd/data": data})
    original_handle = fake._handle

    def handle(cmd):
        custom = {
            "dataset": "dataset count=2 maxcount=20 list=dataset_01|dataset_02",
            "dataset dataset_01": "dataset dataset_01 status=closed schedulelist=sch_ctd "
                                  f"bytecount={len(fake.meta) + len(fake.data) + len(fake.events)} datatype=float32",
            "dataset dataset_02": "dataset dataset_02 status=open schedulelist=sch_ctd "
                                  f"bytecount={len(meta) + len(data) + len(events)} datatype=float32",
            "dataset dataset_02/meta": f"dataset dataset_02/meta bytecount={len(meta)}",
            "dataset dataset_02/events": f"dataset dataset_02/events bytecount={len(events)} eventcount=2",
            "dataset dataset_02/sch_ctd/data": f"dataset dataset_02/sch_ctd/data bytecount={len(data)} samplecount=30",
        }
        if cmd in custom:
            fake.commands.append(cmd)
            fake._reply(custom[cmd])
        else:
            original_handle(cmd)

    monkeypatch.setattr(fake, "_handle", handle)
    run(rig)
    latest = _records(rig, 210000)[1]
    assert latest["deployment"]["stem"] != first["deployment"]["stem"], latest["deployment"]
    assert old_nc.read_bytes() == saved
    with netCDF4.Dataset(rig.tmp / f"{latest['deployment']['stem']}.nc") as ds:
        assert len(ds["time"]) == 30


# --- further tests for the fixes


def test_1_a_replaced_start_under_an_identical_header_is_caught_by_the_first_block(rig, monkeypatch):
    """The first block is compared as well as the tail, so a memory that differs only near its start is another
    deployment even when the header and the data tail are the same."""
    _no_skew(monkeypatch)
    fake = rig.add("usbmodem1", serial=303, n_samples=0)
    error = struct.pack("<I", 0xF6000001)
    head = solo_image(0, SYNC_S, 303)
    fake.set_image(head + struct.pack("<I", 0x20000000) * 1000 + error * 19_000)
    run(rig)
    fake.set_image(head + struct.pack("<I", 0x21000000) * 1000 + error * 19_100)
    run(rig)
    r0, r1 = _records(rig, 303)
    assert r1["deployment"]["stem"] != r0["deployment"]["stem"] and r1["deployment"]["download"] == "full"
    assert r0["deployment"]["tail_check"] is None and r1["deployment"]["tail_check"] is None
    # readings start at byte 520 (header and time-sync record); they differ in their top byte, the fourth
    assert "differs from the deployment image at byte 523" in rig.console_text()


def test_2_one_offload_after_a_terminal_reset_does_not_time_the_run_before_it():
    """Also with a single offload (a change from main, deliberate): its skew was measured on the clock that
    restarted after the last sample, so the reset run before it has no offload on its clock and is dropped."""
    period, n0, n1 = 60_000, 5, 3
    real = 1_790_000_000_000
    t = np.concatenate([real + period * np.arange(n0), 946_684_805_000 + period * np.arange(n1)]).astype(np.int64)
    tf = np.zeros(t.size, np.uint8)
    tf[n0:] |= TFLAG_RESET_CLOCK
    starts = [SegmentStart(n0, 1), SegmentStart(n0 + n1, 2)]  # the second reset: after the last sample
    skew = (946_684_805_000 - (real + 7_200_000)) / 1000
    one = OffloadView(samples_seen=t.size, events_seen=3, unix_ms=real + 7_300_000, skew_s=skew)
    _, _, keep, notes = resolve_time_arrays(t, tf, None, views=[one], starts=starts)
    assert keep.tolist() == [True] * n0 + [False] * n1
    before = OffloadView(samples_seen=t.size, events_seen=2, unix_ms=real + 300_000 + (n1 - 1) * period + 1000,
                         skew_s=(946_684_805_000 - (real + 300_000)) / 1000)
    t_utc, _, keep, _ = resolve_time_arrays(t, tf, None, views=[before, one], starts=starts)
    assert keep.all() and t_utc[n0] == real + 300_000  # an offload before the reset times it
