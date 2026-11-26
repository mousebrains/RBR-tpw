"""--configure on the RBRduet (fwtype 102) and RBRconcerto (103), through the real Link: Ruskin 2.26.1's write
sequence for these models (configure.py docstring), only the settings that change, writes whose reply is lost
(both models drop a command now and then), the paced clock-skew probe, and the gap between commands."""

import threading
import time

import pytest
from fakelogger import FakeConcerto, FakeDuet
from test_offload import fast_skew, rig  # noqa: F401  (rig is a fixture)

from rbr_tpw import cli
from rbr_tpw import configure as cf
from rbr_tpw import link as link_module
from rbr_tpw.configure import ConfigError, DeployConfig, configure, period_problem, rate_period_ms, validate
from rbr_tpw.drivers import driver_for
from rbr_tpw.link import Link
from rbr_tpw.solo import measure_clock_skew


@pytest.fixture
def connect(monkeypatch):
    """A Link to `fake`, its driver's command gap applied as the CLI does, and configure()'s skew re-measurement
    answered from the fake's clock (logger minus host), so a clock set converges without waiting for ticks."""
    def make(fake):
        monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
        link = Link(fake.port)
        drv = driver_for(fake.fwtype)
        drv.serial = fake.serial
        link.min_gap_s = drv.command_gap_s

        def skew(link, reps=3, **k):
            return {"n": 3, "polls": 9, "skew_vs_host_s": fake.skew_s + fake.offset_s, "uncertainty_s": 0.003,
                    "spread_s": 0.001, "individual_s": [0.0] * 3, "measured_at": ""}

        monkeypatch.setattr(cf, "measure_clock_skew", skew)
        return link, drv
    return make


# The clock set's own precision is tested elsewhere; here a slow CI runner may deliver a write 60-100 ms late (macOS,
# c7ff901), which would take the set past the 20 ms tolerance and change the attempt count these tests assert.
LOOSE_CLOCK_S = 0.5


def run_configure(link, drv, fake, cfg):
    if cfg.clock_tolerance_s == DeployConfig.clock_tolerance_s:
        cfg.clock_tolerance_s = LOOSE_CLOCK_S
    return configure(link, fake.serial, cfg, 0.0, log=lambda *a: None, fwtype=fake.fwtype, driver=drv)


def writes(fake):
    """The state-changing commands the logger carried out, without the unlock handshake."""
    return [w for w in fake.writes if not w.startswith("lock")]


# --- the RBRduet


def test_duet_enable_writes_what_ruskin_writes(connect):
    """SN081500 on 2026-09-16 and 09-22 (Ruskin): stop, the clock, the UTC offset (the clock write leaves it
    `unknown`), `enable erasememory = true`. The period, start and end are already as wanted: not written."""
    fake = FakeDuet("/dev/cu.d", n_samples=500)
    link, drv = connect(fake)
    used = len(fake.image)
    report = run_configure(link, drv, fake, DeployConfig(period_ms=500))
    w = writes(fake)
    assert w[0] == "stop" and w[1].startswith("now = ")
    assert w[2:] == ["permit settings", "settings offsetfromutc = 0.0", "enable erasememory = true"]
    steps = [x["step"] for x in report["steps"]]
    assert steps == ["stop_sent", "stop", "set_clock", "offsetfromutc", "schedule", "verify", "erase_sent",
                     "enable_sent", "erase", "enable"]
    assert report["readback"]["offsetfromutc"] == "+0.00"
    erase = next(x for x in report["steps"] if x["step"] == "erase")
    assert erase["used_before"] == used + 8  # the stop event
    assert 0 < erase["meminfo"]["used"] < used  # a fresh header and time-sync marker
    assert report["readback"]["status"] == "logging" and fake.locked
    assert report["readback"]["power"]["battery_voltage_V"] == 3.61
    assert report["readback"]["power"]["energy_remaining_J"] != report["readback"]["power"]["energy_remaining_J"]


def test_duet_cleared_and_left_stopped(connect, monkeypatch):
    """The reset plan for the duets: memory cleared, clock set, 2 Hz, left stopped. Without an enable the erase is
    `permit memclear` + `memclear` (L3 reference 3.9, Ruskin's PhysicalL2.eraseMemory)."""
    monkeypatch.setattr(cf, "IDLE_SAVE_S", 0.0)
    fake = FakeDuet("/dev/cu.d", n_samples=500)
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(period_ms=500, enable=False))
    w = writes(fake)
    assert w[0] == "stop" and w[1].startswith("now = ")
    assert w[2:] == ["permit settings", "settings offsetfromutc = 0.0", "permit memclear", "memclear"]
    assert fake.image == b"" and report["readback"]["status"] == "disabled"
    assert report["readback"]["meminfo"]["used"] == 0
    assert [x["step"] for x in report["steps"]][-3:] == ["erase_sent", "erase", "verify"]


def test_utc_offset_is_zeroed_after_the_clock_and_only_when_it_is_not(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=50)
    link, drv = connect(fake)
    run_configure(link, drv, fake, DeployConfig())
    w = writes(fake)
    i = w.index("permit settings")
    assert w[i - 1].startswith("now = ") and w[i + 1] == "settings offsetfromutc = 0.0"
    assert fake.state["offsetfromutc"] == "+0.00"
    j = fake.commands.index("permit settings")  # the permit is for the next command: no unlock between them
    assert fake.commands[j + 1] == "settings offsetfromutc = 0.0"
    # no clock write and an offset of +0.00: nothing to write
    fake2 = FakeDuet("/dev/cu.d2", n_samples=50)
    link2, drv2 = connect(fake2)
    run_configure(link2, drv2, fake2, DeployConfig(set_clock=False))
    assert "permit settings" not in writes(fake2)
    # no clock write, but an offset that is not zero (local time): zeroed
    fake3 = FakeDuet("/dev/cu.d3", n_samples=50)
    fake3.state["offsetfromutc"] = "-8.00"
    link3, drv3 = connect(fake3)
    run_configure(link3, drv3, fake3, DeployConfig(set_clock=False))
    assert "settings offsetfromutc = 0.0" in writes(fake3) and fake3.state["offsetfromutc"] == "+0.00"


def test_duet_battery_counter_is_refused_before_any_write(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=50)
    link, drv = connect(fake)
    with pytest.raises(ConfigError, match="battery counter of fwtype 102"):
        run_configure(link, drv, fake, DeployConfig(fresh_battery=True))
    assert writes(fake) == []


def test_duet_period_below_its_limit_is_refused_before_any_write(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=50)  # userperiodlimit = 500, as both duets report
    link, drv = connect(fake)
    with pytest.raises(ConfigError, match="userperiodlimit"):
        run_configure(link, drv, fake, DeployConfig(period_ms=250))
    assert writes(fake) == []


# --- the RBRconcerto


def test_concerto_fast_period_and_a_gated_enable(connect):
    """SN060275 on 2026-09-13 (Ruskin): `sampling period = 250`, then the erasing enable with the memory format,
    answered `enable = gated` (twist activation on, twisted off): ready to deploy."""
    fake = FakeConcerto("/dev/cu.c", n_samples=300)
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(period_ms=250))
    w = writes(fake)
    assert w[0] == "stop" and w[1].startswith("now = ")
    assert w[2:] == ["permit settings", "settings offsetfromutc = 0.0", "sampling period = 250",
                     "enable erasememory = true, memformat = RAWBIN00"]
    back = report["readback"]
    assert back["status"] == "gated" and back["sampling"]["period"] == "250"
    assert back["sampling"]["gate"] == "twistactivation" and back["twistactivation"]["state"] == "on"
    assert back["status"] in cf.READY_STATUSES


def test_concerto_six_hertz_and_its_limit(connect):
    fake = FakeConcerto("/dev/cu.c", n_samples=30)  # userperiodlimit = 167, as SN060275/060276 report
    link, drv = connect(fake)
    run_configure(link, drv, fake, DeployConfig(period_ms=rate_period_ms(6)))
    assert "sampling period = 167" in writes(fake)
    fake2 = FakeConcerto("/dev/cu.c2", n_samples=30)
    link2, drv2 = connect(fake2)
    with pytest.raises(ConfigError, match="167 ms"):
        run_configure(link2, drv2, fake2, DeployConfig(period_ms=125))
    assert writes(fake2) == []


def test_concerto_twist_activation_off_logs_at_once(connect):
    fake = FakeConcerto("/dev/cu.c", n_samples=30)
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(twist_activation=False))
    assert "twistactivation state = off" in writes(fake)
    assert report["readback"]["status"] == "logging" and report["readback"]["sampling"]["gate"] == "none"


def test_twist_activation_on_a_duet_is_refused(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=30)
    link, drv = connect(fake)
    with pytest.raises(ConfigError, match="twist activation"):
        run_configure(link, drv, fake, DeployConfig(twist_activation=True))
    assert writes(fake) == []


# --- lost replies


def test_writes_whose_reply_is_lost_are_read_back(connect):
    """A dropped command (never received) is sent again; a write carried out whose reply was lost is not."""
    fake = FakeConcerto("/dev/cu.c", n_samples=30)
    fake.drop_once_prefixes = {"stop"}  # not received: sent again
    fake.lose_reply_once_prefixes = {"sampling period = ", "enable erasememory"}  # done, reply lost: not again
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig(period_ms=250))
    w = writes(fake)
    assert w.count("stop") == 1 and fake.dropped == ["stop"]
    assert w.count("sampling period = 250") == 1
    assert w.count("enable erasememory = true, memformat = RAWBIN00") == 1
    assert report["readback"]["status"] == "gated" and report["readback"]["sampling"]["period"] == "250"
    notes = [e.text for e in link.transcript if e.direction == "NOTE"]
    assert any("'stop' did not take effect; sending it once more" in n for n in notes)
    assert any("'sampling period = 250' took effect; its reply was lost" in n for n in notes)
    assert any("but it took effect: status gated" in n for n in notes)


def test_a_lost_clock_write_is_measured_not_sent_again(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=30)
    fake.skew_s = 5.0
    fake.drop_once_prefixes = {"now = "}
    link, drv = connect(fake)
    report = run_configure(link, drv, fake, DeployConfig())
    clk = next(x for x in report["steps"] if x["step"] == "set_clock")
    assert clk["ok"] and len(clk["history"]) == 2  # attempt 1 lost (still +5 s), attempt 2 set it
    assert abs(clk["history"][0]["skew_vs_utc_s"] - 5.0) < 0.1
    assert sum(1 for c in fake.commands if c.startswith("now = ")) == 2


def test_a_lost_unlock_is_sent_again(connect):
    fake = FakeDuet("/dev/cu.d", n_samples=30)
    link, drv = connect(fake)
    original = fake._handle_lock
    lost = []

    def lose_first_key(cmd):
        if cmd.startswith("lock OFF = ") and not lost:
            lost.append(cmd)
            return True  # swallowed: no reply
        return original(cmd)

    fake._handle_lock = lose_first_key
    run_configure(link, drv, fake, DeployConfig())
    assert lost and any("lock OFF: no reply" in e.text for e in link.transcript if e.direction == "NOTE")
    assert writes(fake)[-1] == "enable erasememory = true"


# --- the paced skew probe and the command gap


def test_paced_probe_survives_dropped_polls(monkeypatch):
    """Unpaced, a dropped `now` costs a 3 s re-send and a bracket can span it (+/-1.68 s on SN081500, 2026-10-19).
    Paced, a drop costs 0.4 s and no bracket spans it."""
    fake = FakeDuet("/dev/cu.d", n_samples=10, drop_every=60, drop_first={"now"})  # 1.7 % (unpaced: 1.5 %) + 1st
    fake.skew_s = 0.25
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link(fake.port)
    link.min_gap_s = 0.010
    skew = measure_clock_skew(link, reps=3, clock=driver_for(102).clock_now, pace_s=0.010, max_seconds=30)
    # n = 3 within 30 s (4-7 s locally); no wall-clock bound here: CI runners are slower
    assert skew["n"] == 3 and skew["dropped_polls"] >= 1 and fake.dropped
    # The tick lies inside every bracket, so the reported half-width must cover the true skew. A bracket spanning a
    # dropped poll (0.1 s timeout + 0.3 s drain) would be >= 0.4 s wide; the unpaced probe's was +/-1.68 s (SN081500).
    # No tighter bound: a slow CI runner widens every bracket (macOS: ~30 ms, against ~7 ms locally).
    assert abs(skew["skew_vs_host_s"] - 0.25) <= skew["uncertainty_s"] + 0.002
    assert skew["uncertainty_s"] < 0.15


def test_commands_wait_for_the_gap_after_a_reply(monkeypatch):
    fake = FakeDuet("/dev/cu.d", n_samples=10)
    monkeypatch.setattr(link_module, "open_serial", lambda port, baudrate: fake)
    link = Link(fake.port)
    link.min_gap_s = 0.010
    for _ in range(5):
        link.query("status")
    t = link.transcript
    gaps = [(b.t_ns - a.t_ns) / 1e9 for a, b in zip(t, t[1:], strict=False)
            if a.direction == "RX" and b.direction == "TX"]
    assert len(gaps) == 4 and min(gaps) >= 0.0095


# --- the period rules


@pytest.mark.parametrize(("period", "fwtype", "limit", "ok"), [
    (500, 9, None, True), (250, 9, None, False), (2000, 0, None, True), (1500, 9, None, False),
    (500, 102, 500, True), (250, 102, 500, False), (167, 103, 167, True), (125, 103, 167, False),
    (250, 103, 167, True), (170, 103, 167, False), (63, 104, 31, True), (31, 104, 31, True),
    (15000, 103, 167, True), (1500, 103, 167, False), (250, None, None, True), (170, None, None, False),
])
def test_period_rules(period, fwtype, limit, ok):
    assert (period_problem(period, fwtype, limit) is None) == ok


def test_rates_round_half_up():
    assert [rate_period_ms(r) for r in (2, 4, 6, 8, 16, 24, 32)] == [500, 250, 167, 125, 63, 42, 31]


def test_the_command_line_accepts_a_fast_period_checked_per_logger_later():
    assert validate(DeployConfig(period_ms=167), 500, fwtype=None)[2] == 167
    with pytest.raises(ConfigError):
        validate(DeployConfig(period_ms=167), 500, fwtype=9)


# --- the CLI end to end


def test_cli_configures_a_duet_and_offloads_it_incrementally_afterwards(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    fake = rig.add("usbmodemD", cls=FakeDuet, n_samples=400)

    def skew(link, reps=3, **k):
        return {"n": 3, "polls": 9, "skew_vs_host_s": fake.skew_s + fake.offset_s, "uncertainty_s": 0.003,
                "spread_s": 0.001, "individual_s": [0.0] * 3, "measured_at": ""}

    monkeypatch.setattr(cf, "measure_clock_skew", skew)
    cli.run(rig.settings(deploy=DeployConfig(period_ms=500, clock_tolerance_s=LOOSE_CLOCK_S), assume_yes=True),
            once=True, port=None)
    text = rig.console_text()
    assert "CONFIGURE FAILED" not in text and "NOT READY" not in text
    assert "memory erased by the enable" in text and "enable erasememory = true" in fake.writes
    # what was offloaded before the erase is saved: one deployment, 400 samples
    (nc0,) = rig.tmp.glob("081500_*.nc")
    # log on, offload twice: a new deployment, then only its new bytes
    fake.grow(50)
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    fake.grow(20)
    held = len(fake.image) - 240
    fake.commands.clear()
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    reads = [c for c in fake.commands if c.startswith("read data")]
    assert reads[0] == f"read data 1 {held} 0" and all(int(c.split()[4]) >= held for c in reads[1:])
    assert len(list(rig.tmp.glob("081500_*.nc"))) == 2 and nc0.exists()


def test_cli_a_gated_concerto_is_ready_to_deploy(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    fake = rig.add("usbmodemC", cls=FakeConcerto, n_samples=200)

    def skew(link, reps=3, **k):
        return {"n": 3, "polls": 9, "skew_vs_host_s": fake.skew_s + fake.offset_s, "uncertainty_s": 0.003,
                "spread_s": 0.001, "individual_s": [0.0] * 3, "measured_at": ""}

    monkeypatch.setattr(cf, "measure_clock_skew", skew)
    cli.run(rig.settings(deploy=DeployConfig(period_ms=250, clock_tolerance_s=LOOSE_CLOCK_S), assume_yes=True),
            once=True, port=None)
    text = rig.console_text()
    assert "CONFIGURE FAILED" not in text and "NOT READY" not in text
    assert "enabled and gated: it samples only while twisted on" in text


def test_a_table_read_locked_once_is_not_a_channel_change(rig, monkeypatch):  # noqa: F811
    """SN081500, 2026-11-26: its unlock reply was lost at one offload, so that table was read locked (the hidden
    thermistor left out, `status = on`); the next offload read it unlocked. That is not a change of the logger."""
    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    fake = rig.add("usbmodemD", cls=FakeDuet, n_samples=200)
    fake.refuse_unlock = True
    cli.run(rig.settings(), once=True, port=None)
    fake.refuse_unlock = False
    fake.grow(10)
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    assert "channel table" not in rig.console_text()
    # a real change still shows: the thermistor's calibration in the unlocked view of both
    fake.COMP = [("c0", 3.39e-3), *fake.COMP[1:]]
    fake.grow(10)
    time.sleep(1.05)
    cli.run(rig.settings(), once=True, port=None)
    assert "channel table (types, statuses or calibration coefficients)" in rig.console_text()


# --- sample timing at a period that is not a whole divisor of 1000 ms


def test_a_sub_second_period_is_an_exact_rate():
    """SN060275 set to `period = 167` sampled 5.99985 Hz (3543 samples in 590.5 s by its memory growth,
    2026-11-26), as the L3 reference says sub-second periods do: sample k is at k * 1000 / 6 ms, not k * 167."""
    from rbr_tpw.rawbin import sample_offsets_ms

    assert sample_offsets_ms(167, 7).tolist() == [0, 167, 333, 500, 667, 833, 1000]
    assert sample_offsets_ms(167, 3544)[-1] == 590_500  # 3543 periods; k * 167 would give 591,681 ms
    assert sample_offsets_ms(53, 20)[-1] == 1000  # the reference's example: 53 ms is 19 Hz
    for p in (500, 250, 125, 1000, 15000):  # unchanged where the period divides 1000 or is >= 1 s
        assert sample_offsets_ms(p, 5).tolist() == [p * k for k in range(5)]


def test_l2_decode_times_a_6_hz_concerto_at_6_hz():
    from fakelogger import FakeConcerto

    from rbr_tpw.equations import decode_l2

    fake = FakeConcerto("/dev/null", n_samples=0)
    fake.state["period"] = 167
    fake.image = fake.fresh_image(843_696_000)  # 2026-09-25T00:00:00Z
    fake.grow(6 * 600 + 1)
    d = decode_l2(fake.image, 4)
    assert d.time_ms[-1] - d.time_ms[0] == 600_000  # ten minutes, not 601.2 s
    assert set(map(int, d.time_ms[1:] - d.time_ms[:-1])) == {166, 167}


def test_read_only_serials_are_offloaded_and_never_written(rig, monkeypatch):  # noqa: F811
    """Loggers held for diagnostics: --configure skips them, the offload still saves their memory."""
    from fakelogger import FakeSoloWritable

    monkeypatch.setattr(cli, "measure_clock_skew", fast_skew({"lock": threading.Lock(), "now": 0, "max": 0}))
    held = rig.add("usbmodemH", cls=FakeSoloWritable, serial=76585, n_samples=50, fwtype=0)
    other = rig.add("usbmodemO", cls=FakeDuet, n_samples=50)
    s = rig.settings(deploy=DeployConfig(enable=False, clock_tolerance_s=LOOSE_CLOCK_S), assume_yes=True,
                     read_only={76585})
    monkeypatch.setattr(cf, "IDLE_SAVE_S", 0.0)
    monkeypatch.setattr(cf, "measure_clock_skew", lambda link, reps=3, **k: {
        "n": 3, "polls": 9, "skew_vs_host_s": 0.0, "uncertainty_s": 0.003, "spread_s": 0.001, "individual_s": [0.0] * 3,
        "measured_at": ""})
    cli.run(s, once=True, port=None)
    assert not any(c.startswith(("stop", "now =", "permit", "memclear", "enable", "starttime =", "sampling mode"))
                   for c in held.commands)
    assert list(rig.tmp.glob("*76585_*.nc")) and "76585 is on the --read-only list" in rig.console_text()
    assert "memclear" in other.writes  # the others are configured as usual
