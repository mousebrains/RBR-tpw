"""Fixes from the Config5 bench session (2026-09-26 evening): an RBRconcerto (fwtype 103, fw 1.460) that now and
then gives no reply at all to a command. Reads are re-sent once, as Ruskin does, and a clock-skew measurement
that fails does not abort the offload. (The concerto3 `status = on` case from the same session is in
test_offload.test_concerto3_reporting_on_off_channel_status.)"""

import json
import time

import netCDF4
import pytest
from fakelogger import FakeDuet, FakeSolo
from test_offload import rig  # noqa: F401  (a fixture)

from rbr_tpw import cli
from rbr_tpw import link as link_module
from rbr_tpw.link import Link, LinkError, LoggerError


@pytest.fixture
def fake_link(monkeypatch):
    def connect(fake):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
        return Link(fake.port)
    return connect


def test_query_resends_a_read_once_after_silence_but_command_does_not(fake_link):
    fake = FakeSolo("/dev/cu.fake", drop_every=2)  # every second command gets no reply
    link = fake_link(fake)
    fake.commands.append("pad")  # command 1, so the query below is command 2: dropped, re-sent as 3, answered
    t0 = time.monotonic()
    assert link.query("status", timeout=0.3)["status"] == "logging"
    assert 0.3 <= time.monotonic() - t0 < 1.5
    assert fake.dropped == ["status"] and fake.commands[-2:] == ["status", "status"]
    assert any(e.direction == "NOTE" and "re-sending" in e.text for e in link.transcript)
    with pytest.raises(LinkError, match="timeout"):
        link.command("status", timeout=0.3)  # command 4, dropped: the write path makes one attempt only
    assert fake.dropped == ["status", "status"]
    with pytest.raises(LoggerError):
        link.query("bogus", timeout=0.3)  # command 5, answered E0102: a logger error is not retried
    assert fake.commands.count("bogus") == 1


def test_offload_completes_on_a_logger_that_drops_a_skew_poll_and_a_settings_read(rig, monkeypatch):  # noqa: F811
    """SN060275 dropped the sixth `now` of the skew measurement; on 4944021 that ended the offload (nothing
    downloaded). Each dropped read costs one 3 s timeout, then the re-send is answered."""
    fake = rig.add("usbmodem101", cls=FakeDuet, n_samples=300, drop_first={"now", "channels"})
    cli.run(rig.settings(), once=True, port=None)
    text = rig.console_text()
    assert "OFFLOAD INCOMPLETE" not in text and "done with /dev/cu.usbmodem101: disconnect the logger" in text
    assert fake.dropped == ["now", "channels"]
    (nc,) = rig.tmp.glob("081500_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 300
    (rec,) = (rig.tmp / "raw").glob("081500_*.json")
    assert json.loads(rec.read_text())["clock_skew"]["n"] == 3  # the dropped poll was re-sent, not fatal


class MuteClock(FakeSolo):
    """Gives no reply to the first two `now` polls (both attempts of the first skew read), then answers."""

    def _handle(self, cmd):
        if cmd == "now" and self.commands.count("now") < 2:
            self.commands.append(cmd)
            return
        super()._handle(cmd)


def test_a_failed_skew_measurement_does_not_abort_the_offload(rig, monkeypatch):  # noqa: F811
    rig.add("usbmodem101", cls=MuteClock, n_samples=100)
    cli.run(rig.settings(), once=True, port=None)
    text = rig.console_text()
    assert "clock skew could not be measured" in text and "OFFLOAD INCOMPLETE" not in text
    (rec,) = (rig.tmp / "raw").glob("100689_*.json")
    r = json.loads(rec.read_text())
    assert r["clock_skew"]["n"] == 0 and "timeout" in r["clock_skew"]["error"]
    assert list(rig.tmp.glob("100689_*.nc"))


# --- the write lock hides channels: read the list unlocked, or complete it from the deployment header


def test_channel_status_forms():
    from rbr_tpw.solo import STATUS_NOT_STORED, channel_status
    assert channel_status("0") == 0 and channel_status("13") == 13 and channel_status("9") == 9
    assert channel_status("on") == 0 and channel_status("") == 0  # measured, or derived in EasyParse
    assert channel_status("on", stored_if_on=False) == STATUS_NOT_STORED  # derived in rawbin00
    assert channel_status("off") == STATUS_NOT_STORED and channel_status("off", stored_if_on=False) == STATUS_NOT_STORED
    assert channel_status("weird") is None


def test_duet_channel_list_is_read_unlocked_and_the_lock_restored(rig):  # noqa: F811
    """Locked, SN060275 hid its compensation thermistor (channel 9 of 9) and answered `status = on`; unlocked with
    Ruskin's key it listed all nine with numeric statuses. The thermistor is in every stored sample set."""
    fake = rig.add("usbmodem101", cls=FakeDuet, n_samples=200)
    cli.run(rig.settings(), once=True, port=None)
    assert "OFFLOAD INCOMPLETE" not in rig.console_text()
    assert fake.unlocks == 1 and fake.locked  # unlocked once for the channel list, then `lock on`
    (rec,) = (rig.tmp / "raw").glob("081500_*.json")
    snap = json.loads(rec.read_text())["snapshot_before"]
    assert snap["channels_read_unlocked"] is True
    assert [(c["type"], c["status_as_reported"], c["status"]) for c in snap["channels_all"]] == [
        ("temp12", "0", 0), ("pres21", "0", 0), ("temp05", "9", 9)]
    assert len(snap["channel_list"]) == 3
    (nc,) = rig.tmp.glob("081500_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 200 and "temperature03" in ds.variables  # the hidden thermistor, decoded


def test_locked_listing_is_completed_from_the_deployment_header(rig):  # noqa: F811
    """If the unlock is refused (wrong key, no `lock` command), the locked list has 2 of 3 channels: the header
    supplies the third, so the 3-word sample sets still decode, with a warning."""
    fake = rig.add("usbmodem101", cls=FakeDuet, n_samples=200)
    fake.refuse_unlock = True
    cli.run(rig.settings(), once=True, port=None)
    text = rig.console_text()
    assert "OFFLOAD INCOMPLETE" not in text and fake.locked
    (rec,) = (rig.tmp / "raw").glob("081500_*.json")
    snap = json.loads(rec.read_text())["snapshot_before"]
    assert snap["channels_read_unlocked"] is False
    assert [(c["type"], c["status_as_reported"]) for c in snap["channels_all"]] == [("temp12", "on"), ("pres21", "on")]
    (nc,) = rig.tmp.glob("081500_*.nc")
    with netCDF4.Dataset(nc) as ds:
        assert len(ds["time"]) == 200 and "temperature03" in ds.variables and "pressure" in ds.variables
    assert "channel 3 (temp05, status 9) is in the deployment header but the logger did not list it" in text


# --- a CPU reset with the clock intact re-anchors the samples that follow it


def test_samples_after_a_cpu_reset_event_are_timed_from_it():
    """RBRsolo SN076315 (bench 2026-09-26): battery down to 2.4 V, sampling stopped on 09-23, USB power 3.5 days
    later gave `CPU reset detected` (0x04, RTC still running) and sampling resumed. Counted from the last sync
    marker those samples would be dated 3.5 days early."""
    import struct

    from fakelogger import SYNC_S, solo_image

    from rbr_tpw.crc import crc16_ccitt
    from rbr_tpw.rawbin import EPOCH2000_MS, clock_segments, decode

    n_before, gap_s, n_after = 100, 3 * 86400, 7
    image = bytearray(solo_image(n_before, SYNC_S))
    body = bytes([0x04, 0xF7]) + struct.pack("<I", SYNC_S + n_before // 2 + gap_s)  # 500 ms period: n/2 s of data
    image += struct.pack(">H", crc16_ccitt(body)) + body
    image += (0.42 * (1 << 30) * __import__("numpy").ones(n_after)).astype("<u4").tobytes()
    d = decode(bytes(image), 1)
    assert d.raw.shape[0] == n_before + n_after and [e.type for e in d.events] == [0x01, 0x04]
    t_event = EPOCH2000_MS + 1000 * (SYNC_S + n_before // 2 + gap_s)
    assert d.time_ms[n_before - 1] == EPOCH2000_MS + 1000 * SYNC_S + 500 * (n_before - 1)  # unchanged before
    assert d.time_ms[n_before] == t_event and d.time_ms[-1] == t_event + 500 * (n_after - 1)  # re-anchored after
    assert len(set(clock_segments(d).tolist())) == 1  # the clock was not reset: one clock segment, one skew


# --- the lock handshake against a logger that drops replies


class DropsLockReplies(FakeDuet):
    """No reply to the first `lock OFF` (though it takes effect) and to the first `lock on`."""

    def _handle(self, cmd):
        if cmd.startswith("lock OFF = ") and not getattr(self, "_dropped_off", False):
            self._dropped_off = True
            self.commands.append(cmd)
            self._handle_lock(cmd)  # the logger acts on it ...
            self._out.clear()  # ... but its reply is lost
            self.dropped.append(cmd)
            return
        super()._handle(cmd)


def test_a_logger_that_drops_lock_replies_ends_up_locked(rig):  # noqa: F811
    fake = rig.add("usbmodem101", cls=DropsLockReplies, n_samples=100, drop_first={"lock on"})
    cli.run(rig.settings(), once=True, port=None)
    assert "OFFLOAD INCOMPLETE" not in rig.console_text()
    assert fake.locked  # `lock OFF` reply lost -> the tool re-locks anyway; `lock on` reply lost -> re-sent
    assert [c for c in fake.dropped if c.startswith("lock")] == [c for c in fake.dropped]  # exactly those two
    assert fake.commands.count("lock on") >= 2
    (rec,) = (rig.tmp / "raw").glob("081500_*.json")
    assert json.loads(rec.read_text())["snapshot_before"]["channels_read_unlocked"] is False
    assert list(rig.tmp.glob("081500_*.nc"))
