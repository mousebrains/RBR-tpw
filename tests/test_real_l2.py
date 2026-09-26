"""Real RBRduet/RBRconcerto memory through the decoder and NetCDF writer, compared with Ruskin 2.26.1's own
results for the same downloads. The images are anonymised (serial number zeroed; see real_l2/extract.py):

- duet: the first 240 sample sets of an RBRduet, with its time-sync anchor (0xF5) and 11 error words (0xF6);
  its pressure channel has the extra header word after n0.
- concerto_twist: a whole 64-sample RBRconcerto record with twist-activation and disable events (0xF3).
- concerto_reset: a whole 736-sample RBRconcerto record on a clock that restarted at 2000-01-01.
"""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from rbr_tpw.equations import parse_l2_header
from rbr_tpw.ncwrite import write_netcdf
from rbr_tpw.rawbin import FLAG_ERROR_CODE, TFLAG_RESET_CLOCK

HERE = Path(__file__).parent / "real_l2"
NAMES = ["duet", "concerto_twist", "concerto_reset"]


@pytest.fixture(params=NAMES)
def fx(request, tmp_path):
    """(name, image, Ruskin's results, the NetCDF written from the image, its channel variables by order)."""
    name = request.param
    blob = json.loads((HERE / f"{name}.json").read_text())
    image = (HERE / f"{name}.bin").read_bytes()
    write_netcdf(image, blob["record"], tmp_path / f"{name}.nc")
    types = {c["index"]: c["type"] for c in blob["record"]["snapshot_before"]["channel_list"]}
    with netCDF4.Dataset(tmp_path / f"{name}.nc") as nc:
        var = {order: next((v for v in nc.variables.values() if getattr(v, "rbr_channel_type", None) == t
                            and "calibration_equation" in v.ncattrs()), None) for order, t in types.items()}
        yield name, image, blob["ruskin"], nc, var


def test_the_serial_number_is_zeroed_and_the_header_is_valid(fx):
    _, image, _, _, _ = fx
    assert parse_l2_header(image).fields["serial"] == 0  # parse_l2_header also checks the header CRC


def test_logger_times_match_ruskin(fx):
    _, _, ruskin, nc, _ = fx
    assert nc["logger_time"][:].tolist() == ruskin["tstamp"]


def test_values_match_ruskin(fx):
    _, _, ruskin, nc, var = fx
    for order, theirs in zip(ruskin["stored_order"], ruskin["values"], strict=True):
        assert var[order] is not None, f"no calibrated variable for channel {order}"
        theirs = np.array(theirs, dtype=float)  # null (Ruskin stored no value) -> NaN
        ours = var[order][:].filled(np.nan)
        assert np.array_equal(np.isnan(ours), np.isnan(theirs)), var[order].name
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-12, equal_nan=True, err_msg=var[order].name)


def test_header_coefficients_pair_as_ruskin_reads_them(fx):
    """Each coefficient word in the header goes to the name Ruskin gives it (the pressure channels' extra word
    after n0 included)."""
    _, _, ruskin, _, var = fx
    for order, coeffs in ruskin["coefficients"].items():
        v = var[int(order)]
        assert v is not None, f"no calibrated variable for channel {order}"
        for k, value in coeffs.items():
            if value in ("value", ""):
                continue
            assert v.getncattr(f"calibration_{k}") == pytest.approx(float(value), rel=1e-12), (v.name, k)


def test_events_match_ruskin(fx):
    _, _, ruskin, nc, _ = fx
    ours = list(zip(nc["event_type"][:].tolist(), nc["event_sample_index"][:].tolist(), strict=True))
    assert ours == [(typ, index - 1) for _, typ, index in ruskin["events"]]
    assert nc["event_time"][:].tolist() == [t for t, _, _ in ruskin["events"]]


def test_error_words_are_flagged_where_ruskin_lists_errors(fx):
    _, _, ruskin, nc, var = fx
    ours = {(int(i), order) for order, v in var.items()
            for i in np.flatnonzero(nc[f"{v.name}_flag"][:] & FLAG_ERROR_CODE)}
    assert ours == {(index - 1, order) for index, order in ruskin["errors"]}


def test_sample_times(fx):
    """A set clock is written as the logger kept it; a reset one is re-timed by Ruskin's drift."""
    name, _, ruskin, nc, _ = fx
    t = np.array(ruskin["tstamp"])
    if name == "concerto_reset":
        assert nc.clock_reset_detected == "yes" and np.all(nc["time_flag"][:] & TFLAG_RESET_CLOCK)
        assert nc["time"][:].tolist() == (t - ruskin["drift_ms"]).tolist()
    else:
        assert nc["time"][:].tolist() == t.tolist() and not np.any(nc["time_flag"][:] & TFLAG_RESET_CLOCK)
