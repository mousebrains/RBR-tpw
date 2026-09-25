from pathlib import Path

import pytest

from rbr_tpw.configure import FAR_FUTURE, PAST, ConfigError, DeployConfig, validate

ROOT = Path(__file__).parent.parent


def test_example_yaml_loads():
    cfg = DeployConfig.from_yaml(ROOT / "deploy.example.yaml")
    assert cfg.period_ms == 500 and cfg.start == "now" and cfg.end == "never" and not cfg.fresh_battery
    assert validate(cfg, 1000) == (PAST, FAR_FUTURE, 500)


def test_rate_and_unknown_keys(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("schedule:\n  rate_hz: 1\n  start: 2026-10-01T00:00:00Z\n")
    cfg = DeployConfig.from_yaml(p)
    assert cfg.period_ms == 1000
    assert validate(cfg, 500)[0] == "20261001000000"
    p.write_text("sample_rate: 2\n")
    with pytest.raises(ConfigError):
        DeployConfig.from_yaml(p)


@pytest.mark.parametrize("period, ok", [(500, True), (1000, True), (60000, True), (250, False), (1500, False)])
def test_period_rules(period, ok):
    cfg = DeployConfig(period_ms=period)
    if ok:
        assert validate(cfg, 500)[2] == period
    else:
        with pytest.raises(ConfigError):
            validate(cfg, 500)


def test_time_rules():
    with pytest.raises(ConfigError):  # naive time: ambiguous zone
        validate(DeployConfig(start="2026-10-01T00:00:00"), 500)
    with pytest.raises(ConfigError):  # end in the past
        validate(DeployConfig(end="2020-01-01T00:00:00Z"), 500)
    with pytest.raises(ConfigError):  # end before start
        validate(DeployConfig(start="2030-01-02T00:00:00Z", end="2030-01-01T00:00:00Z"), 500)
    assert validate(DeployConfig(start="2030-01-01T08:00:00+08:00"), 500)[0] == "20300101000000"


def test_unlock_key_matches_ruskin():
    """Keys Ruskin 2.26.1 sent to SN100689 on 2026-09-25 right after reading `now`."""
    import datetime as dt

    from rbr_tpw.configure import unlock_key
    e2000 = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)

    def secs(s):
        return int((dt.datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=dt.UTC) - e2000).total_seconds())

    assert unlock_key(100689, secs("20260925191209")) == 829003196
    assert unlock_key(100689, secs("20000101000101")) == 2444104072
    assert unlock_key(100689, secs("20260925191310")) == 2333840883
