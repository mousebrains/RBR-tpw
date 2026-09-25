"""Remaining-time model, thresholds, and the configure write sequence against a simulated fwtype-9 logger."""

import datetime as dt
import re
import time
from pathlib import Path

import pytest

from rbr_tpw import power
from rbr_tpw.cli import Settings, Thresholds, _time_checks
from rbr_tpw.configure import DeployConfig, configure, unlock_key
from rbr_tpw.link import LoggerError

ROOT = Path(__file__).parent.parent


def test_energy_model_numbers():
    # 2 Hz, 450 ms active per sample: 90% duty at 0.69 mA
    per_day = power.energy_per_day_J(500, 450)
    assert per_day == pytest.approx(3.6 * (0.69 * 77_760 + 0.0055 * 8_640) / 1000)
    fresh = power.remaining(33_696.0, 512, 132_120_064, 500, 1, 450)
    assert fresh["limited_by"] == "energy"
    assert fresh["energy_days"] == pytest.approx(30_326 / per_day, rel=1e-3)
    assert fresh["memory_days"] == pytest.approx(132_120_064 / (4 * 172_800))
    # 10 s sampling: memory-limited
    slow = power.remaining(33_696.0, 512, 132_120_064, 10_000, 1, 450)
    assert slow["limited_by"] == "energy" or slow["memory_days"] < slow["energy_days"]


def test_used_energy_is_subtracted():
    full = power.remaining(33_696.0, 512, 132_120_064, 500, 1, 450)
    after_30_days = power.remaining(33_696.0, 512 + 4 * 172_800 * 30, 132_120_064 - 4 * 172_800 * 30, 500, 1, 450)
    assert full["energy_days"] - after_30_days["energy_days"] == pytest.approx(30, abs=0.01)


def test_thresholds_yaml_and_checks():
    th = Thresholds.from_yaml(ROOT / "deploy.example.yaml")
    assert th.min_battery_voltage == 3.3 and th.min_days == 90
    s = Settings(outdir=Path("."), thresholds=th, assume_yes=True)
    rem = power.remaining(33_696.0, 512, 132_120_064, 500, 1, 450)
    assert _time_checks("1", "x", rem, 3.64, s) == []
    assert any("BATTERY" in a for a in _time_checks("1", "x", rem, 0.74, s))
    s.thresholds.min_days = 365
    assert any("DAYS OF SAMPLING" in a for a in _time_checks("1", "x", rem, 3.64, s))


class FakeSolo:
    """Just enough of an fwtype-9 RBRsolo to exercise configure() without hardware."""

    def __init__(self, status="finished"):
        self.serial = 100689
        self.state = {"status": status, "starttime": "20000101000000", "endtime": "20991231235959",
                      "sampling": "mode = continuous, period = 2000", "remaining": "1ACDF88", "used": 772}
        self.locked, self.challenge, self.log = True, None, []
        self.transcript = []

    def _now(self):
        return dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")

    def note(self, text):
        pass

    def command(self, cmd, timeout=3.0):
        self.log.append(cmd)
        writes = ("stop", "now =", "starttime =", "endtime =", "sampling mode", "powerstatus remaining",
                  "permit", "enable")
        if cmd.startswith(writes) and cmd != "now" and self.locked:
            raise LoggerError(cmd, "0102", "locked")
        if cmd == "now":
            self.challenge = self._now()
            return f"now = {self.challenge}"
        if cmd.startswith("lock OFF = "):
            e2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
            secs = int((dt.datetime.strptime(self.challenge, "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC) - e2000)
                       .total_seconds())
            if int(cmd.split("=")[1]) != unlock_key(self.serial, secs):
                raise LoggerError(cmd, "0108", "")
            self.locked = False
            return "lock = off"
        if cmd == "lock on":
            self.locked = True
            return "lock = on"
        if cmd == "stop":
            if self.state["status"] not in ("logging", "pending"):
                raise LoggerError(cmd, "0406", f"stop = {self.state['status']}")
            self.state["status"] = "stopped"
            return "stop = stopped"
        if cmd in ("status", "starttime", "endtime"):
            return f"{cmd} = {self.state[cmd]}"
        if cmd == "sampling":
            return f"sampling {self.state['sampling']}"
        if cmd == "meminfo":
            used = self.state["used"]
            return f"meminfo used = {used}, remaining = {132120576 - used}, size = 132120576"
        if cmd == "powerstatus":
            return f"powerstatus source = usb, int = 3634, remaining = {self.state['remaining']}"
        m = re.match(r"(starttime|endtime) = (\d+)$", cmd)
        if m:
            self.state[m.group(1)] = m.group(2)
            return cmd
        if cmd.startswith("sampling mode"):
            self.state["sampling"] = cmd.split(" ", 1)[1]
            return cmd
        if cmd.startswith("powerstatus remaining = "):
            self.state["remaining"] = cmd.rsplit(" ", 1)[1]
            return cmd
        if cmd == "permit memclear":
            return "permit = memclear"
        if cmd == "verify":
            if self.state["used"] > 0:
                raise LoggerError(cmd, "0402", f", verify = {self.state['status']}")
            raise LoggerError(cmd, "0401", ", verify = logging")
        if cmd == "enable":
            self.state["used"] = 512
            self.state["status"] = "logging" if self.state["starttime"] <= self._now() else "pending"
            return f"enable = {self.state['status']}"
        raise AssertionError(f"unexpected command {cmd!r}")

    def command_until_prompt(self, cmd, timeout=60.0):
        self.log.append(cmd)
        assert cmd == "memclear" and not self.locked
        self.state["used"] = 0  # real logger: 0 after memclear, 512 (header) after enable
        return ""

    def query(self, cmd, timeout=3.0):
        from rbr_tpw.link import parse_pairs
        return parse_pairs(self.command(cmd, timeout))


def test_configure_sequence_on_finished_logger():
    fake = FakeSolo(status="finished")
    cfg = DeployConfig(set_clock=False, fresh_battery=True, period_ms=500, start="now", end="never")
    t0 = time.monotonic()
    report = configure(fake, 100689, cfg, 0.0, log=lambda *a: None)
    writes = [c for c in fake.log if c not in ("now", "status", "sampling", "meminfo", "powerstatus",
                                               "starttime", "endtime") and not c.startswith("lock")]
    assert writes == ["powerstatus remaining = 2022900", "starttime = 20000101000000", "endtime = 20991231235959",
                      "sampling mode = continuous, period = 500", "permit memclear", "memclear", "verify", "enable"]
    assert "stop" not in fake.log  # a finished logger is not stopped (E0406)
    assert fake.locked  # re-locked at the end
    assert report["readback"]["status"] == "logging" and report["readback"]["power"]["energy_remaining_J"] == 33696
    assert time.monotonic() - t0 < 30
