"""Gen3 EasyParse ("calbin00") decoding: synthetic round trips, real header/event bytes from Ruskin 2.26.1's serial log,
and (if RBR_TPW_RSK_DIR points at Ruskin .rsk files) dataset 1 checked against Ruskin's own values."""

import os
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from rbr_tpw.crc import check_appended, crc16_ccitt
from rbr_tpw.easyparse import decode_easyparse, decode_events, event_ok, parse_deployment_header, record_dtype

# SN233442 (RBRconcerto3, firmware 1.162), 2026-09-12: first 128 bytes of `readdata size = 256, offset = 0, dataset = 2`
HEADER_128 = bytes.fromhex(
    "010900d4070000b404024e00680000008a040000e28f0300524252636f6e636572746f3300ffffff23004c332d4d31312d4631352d42"
    "454331312d4f50312d47312d53435431322d535031310000003fc57d000800000003c3000100000068733832000000007f1319bc1f00"
    "00000400000003014500e09304000a0000000100")
# SN233442, 2026-09-12 13:49:54: complete `readdata size = 112, offset = 0, dataset = 0` reply (7 events + transfer CRC)
EVENTS_112 = bytes.fromhex(
    "13b216f4407a5297a0010000ffffffff88df19f4407a5297a0010000ffffffff96951af4b8e35297a0010000ffffffffffd718f470ef"
    "5297a0010000ffffffff2e8c1bf4b8dd5397a0010000ffffffffda9b19f4957a6197a0010000ffffffffe8901af4187a6197a0010000"
    "ffffffff3b7a")
# Ruskin's `events` table for the same download (233442_20260912_1349.rsk): (tstamp, type)
EVENTS_RUSKIN = [(1789245160000, 22), (1789245160000, 25), (1789245187000, 26), (1789245190000, 24),
                 (1789245251000, 27), (1789246143125, 25), (1789246143000, 26)]


def make_records(t_ms, values) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    rec = np.zeros(len(t_ms), dtype=record_dtype(values.shape[1]))
    rec["t"] = t_ms
    rec["v"] = values if values.shape[1] > 1 else values[:, 0]
    return rec.tobytes()


def make_event(ms: int, etype: int, payload: int = 0xFFFFFFFF) -> bytes:
    body = bytes([etype, 0xF4]) + struct.pack("<QI", ms, payload)
    return struct.pack(">H", crc16_ccitt(body)) + body


def test_round_trip_with_error_codes():
    n, nchan = 50, 6
    t = 1_789_245_190_000 + 1000 * np.arange(n)
    v = np.tile(np.arange(nchan, dtype=np.float32) + 0.5, (n, 1))
    data = bytearray(make_records(t, v))
    err = 0xFF810010  # L3 error 16: channel value outside reasonable range
    struct.pack_into("<I", data, 7 * 32 + 8 + 4 * 2, err)  # sample 7, channel 3
    ep = decode_easyparse(bytes(data) + b"\x01\x02\x03", nchan)
    assert list(ep.time_ms) == list(t) and ep.trailing_bytes == 3
    assert np.isnan(ep.values[7, 2]) and ep.error_codes[7, 2] == err
    assert int(np.isnan(ep.values).sum()) == 1 and int((ep.error_codes != 0).sum()) == 1
    k = 7 * nchan + 2
    assert np.array_equal(np.delete(ep.values.ravel(), k), np.delete(v.astype(float).ravel(), k))


def test_single_channel_and_wrong_nchan():
    t = 1_789_245_190_000 + 500 * np.arange(20)
    data = make_records(t, np.full((20, 1), 12.25))
    ep = decode_easyparse(data, 1)
    assert ep.values.shape == (20, 1) and np.all(ep.values == 12.25)
    with pytest.raises(ValueError, match="timestamps decrease"):
        decode_easyparse(make_records(t, np.full((20, 6), 12.25)), 5)


def test_synthetic_events_and_bad_crc():
    good = make_event(1_789_245_160_000, 0x22, 16544) + make_event(1_789_245_170_000, 0x23, 471680)
    bad = bytearray(make_event(1_789_245_180_000, 0x27, 5))
    bad[5] ^= 1
    events, nbad = decode_events(good + bytes(bad))
    assert events == [(1_789_245_160_000, 0x22, 16544), (1_789_245_170_000, 0x23, 471680)] and nbad == 1
    ep = decode_easyparse(make_records([1], [[1.0]]), 1, data0=good)
    assert len(ep.events) == 2 and ep.bad_events == 0


def test_real_events_match_ruskin():
    assert check_appended(EVENTS_112)  # the transfer's own CRC
    data0 = EVENTS_112[:-2]
    assert all(event_ok(data0[i : i + 16]) for i in range(0, 112, 16))
    events, nbad = decode_events(data0)
    assert nbad == 0 and [(ms, t) for ms, t, _ in events] == EVENTS_RUSKIN


def test_real_header_fields():
    h = parse_deployment_header(HEADER_128)
    assert h["sections"] == {1: 9, 2: 78, 3: 195}
    assert (h["header_version"], h["header_length"]) == (2004, 1204)
    assert (h["fwtype"], h["firmware_version"], h["serial"]) == (104, 1162, 233442)
    assert h["model"] == "RBRconcerto3" and h["part_number"] == "L3-M11-F15-BEC11-OP1-G1-SCT12-SP11"
    assert h["dataset_format_name"] == "calbin00" and h["period_ms"] == 31 and h["status_code"] == 4  # gated
    assert (h["start_s2000"], h["end_s2000"]) == (0, 3_155_759_999)  # 2000-01-01, 2099-12-31T23:59:59
    assert h["enabled_s2000"] == 842_560_360  # 2026-09-12T20:32:40, the first event above
    assert 946_684_800_000 + 1000 * h["enabled_s2000"] == EVENTS_RUSKIN[0][0]


RSK_DIR = os.environ.get("RBR_TPW_RSK_DIR")


def _easyparse_rsk_files():
    if not RSK_DIR:
        return []
    out = []
    for f in sorted(Path(RSK_DIR).rglob("*.rsk")):
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        try:
            if con.execute("select type from dbInfo").fetchone()[0] == "EPdesktop":
                out.append(f)
        finally:
            con.close()
    return out


@pytest.mark.skipif(not RSK_DIR, reason="set RBR_TPW_RSK_DIR to a folder of Ruskin .rsk files")
def test_dataset1_matches_ruskin_values():
    files = _easyparse_rsk_files()
    assert files, f"no EasyParse .rsk files under {RSK_DIR}"
    for f in files:
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        blob = b"".join(r[0] for r in con.execute("select data from downloads order by part"))
        cols = [r[1] for r in con.execute("pragma table_info(data)") if r[1].startswith("channel")]
        rows = np.array(con.execute(f"select tstamp, {', '.join(cols)} from data order by tstamp").fetchall(),
                        dtype=np.float64).reshape(-1, len(cols) + 1)
        con.close()
        ep = decode_easyparse(blob, len(cols))
        assert ep.trailing_bytes == 0 and len(ep.time_ms) == len(rows), f.name
        assert np.array_equal(ep.time_ms, rows[:, 0].astype(np.int64)), f.name
        assert np.array_equal(ep.values, rows[:, 1:], equal_nan=True), f.name
