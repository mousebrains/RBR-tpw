"""Calibration equations (L3 ref section 7) and the L2 sectioned-header memory layout (RBRduet/RBRconcerto)."""

import os
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.equations import (
    decode_l2,
    evaluate,
    header_coefficients,
    parse_l2_header,
    ratio,
    ruskin_coefficient,
)
from rbr_tpw.rawbin import EPOCH2000_MS, TFLAG_NO_ANCHOR, tmp_equation

RSK_DIR = os.environ.get("RBR_TPW_RSK_DIR")  # a folder of Ruskin .rsk files (private data; tests skip without it)
# SN081500 temperature calibration (Ruskin coefficients table)
TMP = {"c0": 0.0034740728, "c1": -2.5209197e-4, "c2": 2.478486e-6, "c3": -8.594369e-8}
# L3 ref rev L section 7.3.3 example coefficients (pres19)
PRES = {"c0": 0.2346, "c1": 120.9873, "c2": 2.7356, "c3": 0.7, "x0": 9.983, "x1": 0.2003, "x2": 0.2943,
        "x3": 0.0721, "x4": 0.1049, "x5": 21.29, "n0": 2}


def words(r):
    """Voltage ratios -> signed 32-bit readings as stored."""
    return np.round(np.asarray(r) * 2**30).astype(np.int64).astype(np.int32).view(np.uint32)


def test_ratio_is_signed():
    assert ratio(np.array([0xFFFDD680], np.uint32))[0] == pytest.approx(-141_696 / 2**30)
    assert ratio(np.array([1 << 29], np.uint32))[0] == 0.5


def test_tmp_matches_the_solo_decoder():
    raw = words(np.linspace(0.3, 0.6, 50))
    ours, bad, problems = evaluate(raw, [{"index": 1, "equation": "tmp", "status": 0, "coefficients": TMP}])
    solo, _ = tmp_equation(raw, tuple(TMP[f"c{i}"] for i in range(4)))
    assert not problems and not bad.any() and np.array_equal(ours[:, 0], solo)


def test_corr_pres2_follows_section_7_3_3():
    r_t, r_p = np.array([0.40, 0.45]), np.array([0.10, 0.20])
    channels = [{"index": 1, "type": "pres19", "equation": "corr_pres2", "status": 0, "coefficients": PRES},
                {"index": 2, "type": "temp09", "equation": "tmp", "status": 0, "coefficients": TMP}]
    values, bad, problems = evaluate(np.column_stack([words(r_p), words(r_t)]), channels)
    assert not problems and not bad.any()
    temp, _ = tmp_equation(words(r_t), tuple(TMP[f"c{i}"] for i in range(4)))
    rp = ratio(words(r_p))
    praw = PRES["c0"] + PRES["c1"] * rp + PRES["c2"] * rp**2 + PRES["c3"] * rp**3
    dt = temp - PRES["x5"]
    expect = PRES["x0"] + (praw - PRES["x0"] - PRES["x1"] * dt - PRES["x2"] * dt**2 - PRES["x3"] * dt**3) / (
        1 + PRES["x4"] * dt)
    assert np.allclose(values[:, 0], expect, rtol=0, atol=1e-12)


def test_corr_cond_uses_temperature_and_pressure_channels():
    cond = {"c0": 0.03, "c1": 154.6, "x0": 2.57e-4, "x1": -9.71e-6, "x2": 6e-7, "x3": 15.0, "x4": 10.0,
            "n0": "2", "n1": "3"}
    channels = [{"index": 1, "equation": "corr_cond", "status": 0, "coefficients": cond},
                {"index": 2, "equation": "tmp", "status": 0, "coefficients": TMP},
                {"index": 3, "equation": "lin", "status": 0, "coefficients": {"c0": 0.0, "c1": 100.0}}]
    r = np.array([[0.25, 0.4, 0.2]])
    values, bad, problems = evaluate(words(r), channels)
    t, p = values[0, 1], values[0, 2]
    craw = 0.03 + 154.6 * ratio(words(r))[0, 0]
    expect = (craw - 2.57e-4 * (t - 15.0)) / (1 - 9.71e-6 * (t - 15.0) + 6e-7 * (p - 10.0))
    assert not problems and values[0, 0] == pytest.approx(expect, abs=1e-12)


def test_value_references_errors_and_problems():
    pres = {**PRES, "n0": "value"}
    channels = [{"index": 1, "equation": "corr_pres2", "status": 0, "coefficients": pres},
                {"index": 2, "equation": "deri_seapres", "status": 4, "coefficients": {"n0": 1, "n1": -1}},
                {"index": 3, "equation": "tmp", "status": 9, "coefficients": TMP},
                {"index": 4, "equation": "mystery", "status": 0, "coefficients": {}},
                {"index": 5, "equation": "corr_pres2", "status": 0, "coefficients": {**PRES, "n0": 2}}]
    raw = np.column_stack([words([0.1, 0.1]), words([0.4, 0.4]), words([0.1, 0.1]), words([0.1, 0.1])])
    raw[1, 1] = 0xF6094E28  # error code on the temperature channel
    values, bad, problems = evaluate(raw, channels, defaults={"temperature": PRES["x5"]})
    rp = ratio(words([0.1]))[0]
    praw = PRES["c0"] + PRES["c1"] * rp + PRES["c2"] * rp**2 + PRES["c3"] * rp**3
    assert values[0, 0] == pytest.approx(praw, abs=1e-12)  # dT = 0 with the default temperature
    assert bad[1, 1] and not bad[0, 1]
    assert set(problems) == {2, 3}  # unknown equation; reference to a channel that is not stored
    assert "not implemented" in problems[2] and "not stored" in problems[3]
    assert bad[:, 2:].all()


def _l2_image(readings, events, period=500, logger_time=843_000_000):
    """A minimal version-1009 L2 memory image: sections 1-3, CRC, then events and readings."""
    s2 = bytearray(b"\xff" * 503)
    s2[0] = 2
    s2[1:3] = struct.pack("<H", 503)
    for rel, v in ((3, 3220), (7, 81500), (11, logger_time), (15, 0), (19, 3_155_759_999), (23, period),
                   (27, 2), (31, 2), (35, 9600), (39, 0x103)):
        struct.pack_into("<I", s2, rel, v)
    ch = b"temp12" + struct.pack("<HIB", 0, 558_177_718, 4) + struct.pack("<4f", *(TMP[f"c{i}"] for i in range(4)))
    s3 = bytes([3]) + struct.pack("<H", 3 + 1 + 2 + len(ch)) + bytes([1]) + struct.pack("<H", 6) + ch
    length = 9 + len(s2) + len(s3) + 2
    s1 = bytes([1]) + struct.pack("<H", 9) + struct.pack("<I", 1009) + struct.pack("<H", length)
    hdr = s1 + bytes(s2) + s3
    body = b""
    for item in readings_and_events(readings, events):
        body += item
    return hdr + struct.pack(">H", crc16_ccitt(hdr)) + body


def readings_and_events(readings, events):
    """events: {position: (marker, type, seconds, ms, info)} inserted before readings[position]."""
    for i in range(len(readings) + 1):
        if i in events:
            marker, etype, seconds, ms, info = events[i]
            if marker == 0xF7:
                rec = bytes([etype, 0xF7]) + struct.pack("<I", seconds)
            else:
                rec = bytes([etype, marker]) + struct.pack("<IHBB", seconds, ms, 3 if marker == 0xF3 else 0, info)
            yield struct.pack(">H", crc16_ccitt(rec)) + rec
        if i < len(readings):
            yield struct.pack("<I", int(readings[i]))


def test_decode_l2_layout_and_timing():
    raw = words(np.linspace(0.4, 0.5, 6))
    raw[1] = 0xF601E7A1
    events = {0: (0xF5, 0x01, 843_000_010, 250, 0),  # 0xF5 time sync, with ms
              4: (0xF3, 0x18, 843_000_200, 500, 1)}  # 0xF3 twist started, info bit 0: anchors the next sample
    image = _l2_image(raw, events)
    hdr = parse_l2_header(image)
    assert hdr.version == 1009 and hdr.period_ms == 500 and hdr.serial == 81500 and hdr.logger_time == 843_000_000
    (ch,) = hdr.fields["channels"]
    assert ch["type"] == "temp12" and ch["status"] == 0
    coeffs = header_coefficients(ch, ["c0", "c1", "c2", "c3"])
    assert coeffs == {k: ruskin_coefficient(v) for k, v in TMP.items()}
    d = decode_l2(image, 1)
    assert list(d.raw[:, 0]) == list(raw) and d.trailing_bytes == 0
    assert [(e.type, e.sample_index) for e in d.events] == [(0x01, 0), (0x18, 4)]
    t0 = EPOCH2000_MS + 1000 * 843_000_010 + 250
    t4 = EPOCH2000_MS + 1000 * 843_000_200 + 500
    assert list(d.time_ms) == [t0, t0 + 500, t0 + 1000, t0 + 1500, t4, t4 + 500]
    assert d.flags[1, 0] == 1 and not (d.time_flags & TFLAG_NO_ANCHOR).any()


def test_decode_l2_rejects_a_corrupt_header():
    image = bytearray(_l2_image(words([0.4]), {0: (0xF5, 0x01, 843_000_010, 0, 0)}))
    image[40] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        parse_l2_header(bytes(image))


RUSKIN_FILES = ["081500_20260923_0955.rsk", "060276_20260912_0957.rsk"]  # an RBRduet and an RBRconcerto


@pytest.mark.skipif(not RSK_DIR, reason="set RBR_TPW_RSK_DIR to a folder of Ruskin .rsk files")
@pytest.mark.parametrize("name", RUSKIN_FILES)
def test_matches_ruskin(name):
    """Memory image and Ruskin's values from the same .rsk: counts, times, events and values must agree."""
    path = next(Path(RSK_DIR).rglob(name), None)
    if path is None:
        pytest.skip(f"{name} not under RBR_TPW_RSK_DIR")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    image = b"".join(r[0] for r in con.execute("select data from downloads order by deploymentID, part"))
    channels = []
    for idx, status, eq, cid in con.execute(
            "select ic.channelOrder, ic.channelStatus, cal.equation, cal.calibrationID from instrumentChannels ic "
            "left join calibrations cal on cal.channelOrder = ic.channelOrder order by ic.channelOrder"):
        coeffs = dict(con.execute("select key, value from coefficients where calibrationID = ?", (cid,)).fetchall())
        channels.append({"index": idx, "status": status, "equation": eq, "coefficients": coeffs})
    cols = [r[1] for r in con.execute("pragma table_info(data)") if r[1].startswith("channel")]
    data = np.array(con.execute(f"select tstamp, {','.join(cols)} from data order by tstamp").fetchall(), float)
    d = decode_l2(image, len(cols))
    assert len(d.time_ms) == len(data) and np.array_equal(d.time_ms, data[:, 0].astype(np.int64))
    ruskin_events = con.execute("select type, sampleIndex from events where type < 256 order by tstamp, rowid")
    assert [(e.type, e.sample_index + 1) for e in d.events] == [tuple(r) for r in ruskin_events]
    values, bad, problems = evaluate(d.raw, channels)
    assert not problems
    assert np.array_equal(bad, np.isnan(data[:, 1:]))
    assert np.nanmax(np.abs(values - data[:, 1:])) < 1e-12
