"""Decoder regression tests against data Ruskin 2.26.1 decoded (truth) and a raw logger transfer."""

from pathlib import Path

import numpy as np
import pytest

from rbr_tpw.crc import check_appended
from rbr_tpw.link import parse_pairs
from rbr_tpw.rawbin import decode, hexfloat, tmp_equation

DATA = Path(__file__).parent / "data"
# Recorded logger data is kept out of the public repository.
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="tests/data (recorded logger data) not present")
# SN100689 factory calibration as reported by `calibration 1` (hex IEEE-754 singles)
C_100689 = tuple(hexfloat(h) for h in ("3B639047", "B9845460", "36209858", "B3AD3E1B"))


def test_hex_coefficients_match_ruskin():
    ruskin = (0.0034723447170108557, -2.5239866226911545e-4, 2.393053364357911e-6, -8.067237189379739e-8)
    assert C_100689 == ruskin


@needs_data
def test_july_file_matches_ruskin():
    """Blob from 100689_20260720_1742.rsk: 24 samples after an RTC reset."""
    image = (DATA / "100689_20260720_rsk_download.bin").read_bytes()
    truth = np.loadtxt(DATA / "100689_20260720_rsk_data.csv", delimiter=",")
    d = decode(image, 1)
    assert d.header.serial == 100689 and d.header.period_ms == 500 and d.header.length == 512
    assert [(e.type, e.seconds, e.sample_index) for e in d.events] == [(1, 10, 0), (2, 21, 23)]
    assert d.rtc_reset  # sync event is before the header's enable time
    t, bad = tmp_equation(d.raw[:, 0], C_100689)
    assert len(t) == len(truth) == 24 and not bad.any()
    assert np.max(np.abs(t - truth[:, 1])) < 1e-12
    # Ruskin anchored these at the header start time; we keep the logger's (reset) clock.
    assert np.all(np.diff(d.time_ms) == 500)
    assert d.time_ms[0] == 946_684_810_000


@needs_data
def test_live_transfer_crc_and_decode():
    """`read data 1 6524 0` from SN100689 on 2026-09-25, with its trailing CRC."""
    block = (DATA / "100689_20260925_read_data_with_crc.bin").read_bytes()
    assert check_appended(block)
    assert not check_appended(block[:-1] + bytes([block[-1] ^ 1]))
    d = decode(block[:-2], 1)
    assert d.header.trailer_event == (0x04, d.header.trailer_event[1])  # CPU reset record in the header
    assert d.events[0].type == 0x01 and d.events[0].sample_index == 0
    assert d.trailing_bytes == 0 and not d.rtc_reset
    t, bad = tmp_equation(d.raw[:, 0], C_100689)
    assert not bad.any() and 15 < np.nanmin(t) and np.nanmax(t) < 35  # bench, room temperature


def test_parse_pairs():
    assert parse_pairs("meminfo used = 8012, remaining = 132112068, size = 132120576") == {
        "used": "8012", "remaining": "132112068", "size": "132120576"}
    assert parse_pairs("channel 1 type = temp02, equation = tmp")["type"] == "temp02"
    assert parse_pairs("now = 20260925192846") == {"now": "20260925192846"}
