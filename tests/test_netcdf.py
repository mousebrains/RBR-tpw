"""NetCDF writer: content round-trip and CF compliance (IOOS compliance-checker cf:1.11 grammar;
the files declare CF-1.13, for which no checker exists yet)."""

import io
from contextlib import redirect_stdout
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from rbr_tpw.ncwrite import write_netcdf
from rbr_tpw.rawbin import hexfloat
from rbr_tpw.solo import sha256

DATA = Path(__file__).parent / "data"
# Recorded logger data is kept out of the public repository.
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="tests/data (recorded logger data) not present")


@pytest.fixture
def nc_path(tmp_path: Path) -> Path:
    image = (DATA / "100689_20260925_read_data_with_crc.bin").read_bytes()[:-2]
    coeffs = {f"c{i}": hexfloat(h) for i, h in enumerate(("3B639047", "B9845460", "36209858", "B3AD3E1B"))}
    record = {
        "port": "/dev/cu.usbmodem101",
        "offload_started": "2026-09-25T19:23:00.000Z",
        "id": {"model": "RBRsolo", "version": "1.000", "serial": "100689", "fwtype": 9},
        "host_ntp": {"server": "time.apple.com", "offset_s": 0.0002, "uncertainty_s": 0.025},
        "clock_skew": {"n": 3, "skew_vs_host_s": 0.5, "uncertainty_s": 0.003, "spread_s": 0.001,
                       "measured_at": "2026-09-25T19:22:58.000+00:00"},
        "snapshot_before": {
            "status": "logging", "sampling": {"mode": "continuous", "period": "500"},
            "channel_list": [{"type": "temp02", "equation": "tmp", "calibration_datetime": "20160212143123",
                              "coefficients": coeffs}],
            "meminfo": {"used": 6524, "remaining": 132114052, "size": 132120576},
            "power": {"source": "usb", "int_raw": "793", "remaining_raw": "1ACDF88", "battery_voltage_V": 0.793,
                      "energy_remaining_J": 28106.632, "energy_nominal_J": 33696.0,
                      "energy_remaining_fraction": 28106.632 / 33696.0}},
        "raw": {"file": "raw/test.bin", "bytes": len(image), "sha256": sha256(image)},
        "warnings": [],
    }
    path = tmp_path / "100689_test.nc"
    write_netcdf(image, record, path)
    return path


def test_contents(nc_path: Path):
    with netCDF4.Dataset(nc_path) as nc:
        assert nc.Conventions == "CF-1.13"
        assert nc.instrument_serial_number == "100689"
        assert nc.clock_skew_s == pytest.approx(0.5 - 0.0002)
        assert nc.battery_voltage_V == pytest.approx(0.793)
        t = nc["time"][:]
        assert len(t) == 1501 and np.all(np.diff(t) == 500)
        temp = nc["temperature"][:]
        assert 15 < temp.min() and temp.max() < 35
        assert list(nc["event_type"][:]) == [1]


# Known false positive: the checker only accepts the literal "CF-1.11".
_CF_FALSE_POSITIVES = ('Conventions global attribute does not contain "CF-1.11"',)


def test_cf_compliance(nc_path: Path):
    cc = pytest.importorskip("compliance_checker.runner")
    cc.CheckSuite.load_all_available_checkers()
    buf = io.StringIO()
    with redirect_stdout(buf):
        cc.ComplianceChecker.run_checker(str(nc_path), checker_names=["cf:1.11"], verbose=0,
                                         criteria="strict", output_filename="-", output_format="text")
    problems = []
    for line in buf.getvalue().splitlines():
        s = line.strip()
        if not s.startswith("*"):
            continue
        msg = s.lstrip("*").strip()
        if not msg or "potential issues" in msg.lower() or any(fp in msg for fp in _CF_FALSE_POSITIVES):
            continue
        problems.append(msg)
    assert not problems, "\n".join(problems) + "\n\n" + buf.getvalue()


def test_rtc_reset_segment_is_retimed(tmp_path: Path):
    """SN100689 on 2026-09-25: USB unplugged 19:50:52.9-19:52:27.8 UTC with a non-powering battery.
    The logger restarted on a reset clock (event 0x0A at 2000-01-01) with no later sync event."""
    import json
    record = json.loads((DATA / "100689_20260925_after_unplug.json").read_text())
    image = (DATA / "100689_20260925_after_unplug.bin").read_bytes()
    d, warnings, t = write_netcdf(image, record, tmp_path / "x.nc")
    assert d.rtc_reset and len(t) == 4421
    jumps = np.flatnonzero(np.diff(t) != 500)
    assert list(jumps) == [4407]
    assert t[4407] == 1790365794500  # 19:49:54.5, last sample written before the power cut
    assert t[4408] == 1790365947348  # 19:52:27.348 = logger 2000-01-01T00:00:00 minus the offload skew
    with netCDF4.Dataset(tmp_path / "x.nc") as nc:
        assert list(nc["time_flag"][-13:]) == [6] * 13 and nc["time_flag"][4407] == 0
        assert nc.clock_reset_detected == "yes"
        assert list(nc["event_sample_index"][:]) == [0, 4408]
