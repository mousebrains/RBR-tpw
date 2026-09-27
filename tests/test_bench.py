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
