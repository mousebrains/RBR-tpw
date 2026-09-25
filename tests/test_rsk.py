"""rbr-rsk2nc against small synthetic .rsk files that follow the layout of Ruskin 2.26.1's (RSK schema 2.19.0)."""

import sqlite3
import struct
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.rawbin import EPOCH2000_MS, hexfloat, tmp_equation
from rbr_tpw.rsk import convert, main

C_100689 = tuple(hexfloat(h) for h in ("3B639047", "B9845460", "36209858", "B3AD3E1B"))
T0_S = 843_696_000  # 2026-09-25T00:00:00Z in seconds since 2000
PERIOD = 500

SCHEMA = """
create table dbInfo (version, type);
create table instruments (instrumentID integer primary key, serialID, model, firmwareVersion, firmwareType,
                          partNumber);
create table deployments (deploymentID integer primary key, instrumentID, comment, loggerStatus, loggerTimeDrift,
                          timeOfDownload, name, sampleSize);
create table schedules (scheduleID integer primary key, instrumentID, mode, gate);
create table continuous (continuousID integer primary key, scheduleID, samplingPeriod);
create table epochs (deploymentID integer primary key, startTime, endTime);
create table appSettings (deploymentID integer primary key, ruskinVersion);
create table channels (channelID integer primary key, shortName, longName, units, longNamePlainText,
                       unitsPlainText, isMeasured, isDerived);
create table instrumentChannels (instrumentID, channelID, channelOrder, channelStatus);
create table calibrations (calibrationID integer primary key, channelOrder, instrumentID, type, tstamp, equation);
create table coefficients (calibrationID, key, value);
create table instrumentSensors (instrumentID, sensorID, channelOrder, serialID, details);
create table parameterKeys (parameterID, key, value);
create table downloads (deploymentID, part, offset, data blob);
create table events (deploymentID, tstamp, type, sampleIndex, channelIndex, notes);
create table errors (deploymentID, tstamp, type, sampleIndex, channelOrder);
"""


def make_rsk(path: Path, *, model, serial, fwtype, kind, channels, t_ms, values, drift_ms, tod_ms,
             events=(), errors=(), image=None):
    """channels: (shortName, longName, units, status, derived, equation, {coefficient: value}) in channel order."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.execute("insert into dbInfo values ('2.19.0', ?)", (kind,))
    con.execute("insert into instruments values (1, ?, ?, '1.000', ?, '')", (serial, model, fwtype))
    con.execute("insert into deployments values (1, 1, '', 'logging', ?, ?, ?, ?)", (drift_ms, tod_ms, path.name,
                                                                                     len(t_ms)))
    con.execute("insert into schedules values (1, 1, 'continuous', 'none')")
    con.execute("insert into continuous values (1, 1, ?)", (PERIOD,))
    con.execute("insert into epochs values (1, 946684800000, 4102444799000)")
    con.execute("insert into appSettings values (1, '2.26.1')")
    stored = []
    for order, (short, long, units, status, derived, eq, coeffs) in enumerate(channels, 1):
        con.execute("insert into channels values (?, ?, ?, ?, ?, ?, ?, ?)",
                    (order, short, long, units, long, units, int(not derived), int(derived)))
        con.execute("insert into instrumentChannels values (1, ?, ?, ?)", (order, order, status))
        con.execute("insert into calibrations values (?, ?, 1, 'factory', 1455287483000, ?)", (order, order, eq))
        con.executemany("insert into coefficients values (?, ?, ?)", [(order, k, repr(v)) for k, v in coeffs.items()])
        if not status & 0x04:
            stored.append(f"channel{order:02d}")
    con.execute(f"create table data (tstamp bigint, {', '.join(c + ' double' for c in stored)}, primary key (tstamp))")
    rows = [(int(t), *(None if np.isnan(x) else float(x) for x in row)) for t, row in zip(t_ms, values, strict=True)]
    con.executemany(f"insert into data values ({', '.join('?' * (len(stored) + 1))})", rows)
    con.executemany("insert into events values (1, ?, ?, ?, null, '')", events)
    con.executemany("insert into errors values (1, ?, ?, ?, ?)", errors)
    if image is not None:  # Ruskin stores the download in 68000-byte parts
        con.executemany("insert into downloads values (1, ?, ?, ?)",
                        [(i // 68000 + 1, i, image[i : i + 68000]) for i in range(0, len(image), 68000)])
    con.commit()
    con.close()
    return path


def solo_image(n: int, sync_s: int, enable_s: int, serial=100689) -> tuple[bytes, np.ndarray]:
    """fwtype-9 memory: 512-byte header, a time-sync event, then n single-channel readings."""
    hdr = bytearray(b"\xff" * 512)
    for off, v in ((0, 512), (4, 1000), (8, serial), (12, enable_s), (16, enable_s), (20, 3_155_759_999),
                   (24, PERIOD), (40, 0)):
        struct.pack_into("<I", hdr, off, v)
    body = bytes([0x01, 0xF7]) + struct.pack("<I", sync_s)
    event = struct.pack(">H", crc16_ccitt(body)) + body
    raw = (np.linspace(0.40, 0.45, n) * (1 << 30)).astype("<u4")
    return bytes(hdr) + event + raw.tobytes(), raw


SOLO_CHANNEL = ("temp02", "Temperature", "°C", 0, False, "tmp", {f"c{i}": c for i, c in enumerate(C_100689)})


def test_solo_is_decoded_and_matches_ruskin(tmp_path, cf_problems):
    image, raw = solo_image(40, T0_S, T0_S)
    t = EPOCH2000_MS + 1000 * T0_S + PERIOD * np.arange(40)
    temp, _ = tmp_equation(raw, C_100689)
    rsk = make_rsk(tmp_path / "100689_x.rsk", model="RBRsolo", serial=100689, fwtype=9, kind="full",
                   channels=[SOLO_CHANNEL], t_ms=t, values=temp[:, None], drift_ms=-12, tod_ms=int(t[-1]) + 5000,
                   events=[(int(t[0]), 1, 1)], image=image)
    out = tmp_path / "x.nc"
    r = convert(rsk, out)
    assert r.route == "decode" and r.samples == 40 and not r.warnings
    assert "logger times identical" in r.note
    with netCDF4.Dataset(out) as nc:
        assert list(nc["temperature_raw"][:]) == list(raw)
        assert np.max(np.abs(nc["temperature"][:] - temp)) < 1e-12
        assert list(nc["time"][:]) == list(t)
        assert nc.clock_skew_vs_host_s == pytest.approx(-0.012) and np.isnan(nc.clock_skew_s)
        assert nc.instrument_serial_number == "100689" and nc.conversion_route == "decode"
        assert nc.raw_file == "100689_x.rsk (downloads table)"
    assert not cf_problems(out)


def test_solo_ruskin_values_route(tmp_path):
    image, raw = solo_image(10, T0_S, T0_S)
    t = EPOCH2000_MS + 1000 * T0_S + PERIOD * np.arange(10)
    temp, _ = tmp_equation(raw, C_100689)
    rsk = make_rsk(tmp_path / "s.rsk", model="RBRsolo", serial=100689, fwtype=9, kind="full",
                   channels=[SOLO_CHANNEL], t_ms=t, values=temp[:, None], drift_ms=0, tod_ms=int(t[-1]),
                   image=image)
    r = convert(rsk, tmp_path / "s.nc", force_values=True)
    assert r.route == "values" and r.samples == 10
    with netCDF4.Dataset(tmp_path / "s.nc") as nc:
        assert "temperature_raw" not in nc.variables


CONCERTO = [
    ("cond06", "Conductivity", "mS/cm", 0, False, "corr_cond", {"c0": 0.03, "n0": 3}),
    ("temp09", "Temperature", "°C", 0, False, "tmp", {"c0": 0.0034}),
    ("pres24", "Pressure", "dbar", 0, False, "corr_pres2", {"x0": 10.1}),
    ("pres08", "Sea pressure", "dbar", 4, True, "deri_seapres", {}),  # derived, not stored
    ("temp10", "Temperature", "°C", 9, False, "tmp", {}),  # hidden, stored
]


def _concerto_values(n):
    return np.column_stack([np.full(n, 42.9), np.full(n, 15.0), np.full(n, 20.0), np.full(n, 16.0)])


def test_values_route(tmp_path, cf_problems):
    n = 20
    t = 1_790_000_000_000 + 1000 * np.arange(n)
    v = _concerto_values(n)
    v[3, 1] = np.nan
    rsk = make_rsk(tmp_path / "060275_x.rsk", model="RBRconcerto", serial=60275, fwtype=103, kind="EPdesktop",
                   channels=CONCERTO, t_ms=t, values=v, drift_ms=None, tod_ms=int(t[-1]) + 1000,
                   events=[(int(t[5]) - 400, 0x19, -1), (int(t[5]), 307, 6)],
                   errors=[(int(t[7]), 9, 8, 3)])
    out = tmp_path / "y.nc"
    r = convert(rsk, out)
    assert r.route == "values" and r.samples == n
    assert any("no clock drift" in w for w in r.warnings)
    with netCDF4.Dataset(out) as nc:
        assert {"conductivity", "temperature", "pressure", "temperature05"} <= set(nc.variables)
        assert "sea_pressure" not in nc.variables and "temperature_raw" not in nc.variables
        assert nc["temperature"].standard_name == "sea_water_temperature"
        assert "standard_name" not in nc["temperature05"].ncattrs() and nc["temperature05"].rbr_channel_status == 9
        assert nc["conductivity"].units == "mS cm-1" and nc["conductivity"].calibration_n0 == 3
        assert list(np.flatnonzero(nc["temperature_flag"][:])) == [3]
        assert list(np.flatnonzero(nc["pressure_flag"][:])) == [7]
        assert list(nc["event_type"][:]) == [0x19, 307] and list(nc["event_sample_index"][:]) == [5, 5]
        assert "ruskin_event_307" in nc["event_type"].flag_meanings
        assert "pres08 (Sea pressure, derived)" in nc.ruskin_channels_not_stored
        assert nc.instrument_serial_number == "060275" and np.isnan(nc.clock_skew_s)
    assert not cf_problems(out)


def test_values_route_retimes_a_reset_clock(tmp_path):
    """Whole record on a clock that restarted at 2000-01-01; Ruskin's drift says how far behind it is."""
    n = 10
    t = EPOCH2000_MS + 300_000 + 1000 * np.arange(n)
    true_first = 1_789_171_855_000
    drift = int(t[0]) - true_first
    rsk = make_rsk(tmp_path / "r.rsk", model="RBRconcerto", serial=60275, fwtype=103, kind="full",
                   channels=CONCERTO, t_ms=t, values=_concerto_values(n), drift_ms=drift,
                   tod_ms=int(t[-1]) + 2000)
    r = convert(rsk, tmp_path / "r.nc")
    assert r.samples == n and r.t_first == true_first
    with netCDF4.Dataset(tmp_path / "r.nc") as nc:
        assert list(nc["time_flag"][:]) == [6] * n
        assert list(nc["logger_time"][:]) == list(t)
        assert nc.clock_reset_detected == "yes"


def test_empty_file_writes_nothing(tmp_path, capsys):
    rsk = make_rsk(tmp_path / "e.rsk", model="RBRduet", serial=81015, fwtype=102, kind="full",
                   channels=CONCERTO[:2], t_ms=[], values=np.empty((0, 2)), drift_ms=0, tod_ms=0,
                   image=b"\x00" * 1000)
    main([str(tmp_path / "out"), str(rsk)])
    assert "Ruskin's data table is empty" in capsys.readouterr().out
    assert not (tmp_path / "out" / "e.nc").exists()


def test_cli_keeps_folders_and_skips_existing(tmp_path, capsys):
    image, raw = solo_image(4, T0_S, T0_S)
    t = EPOCH2000_MS + 1000 * T0_S + PERIOD * np.arange(4)
    temp, _ = tmp_equation(raw, C_100689)
    (tmp_path / "in" / "a").mkdir(parents=True)
    make_rsk(tmp_path / "in" / "a" / "s.rsk", model="RBRsolo", serial=100689, fwtype=9, kind="full",
             channels=[SOLO_CHANNEL], t_ms=t, values=temp[:, None], drift_ms=0, tod_ms=int(t[-1]),
             events=[(int(t[0]), 1, 1)], image=image)
    main([str(tmp_path / "out"), str(tmp_path / "in")])
    assert (tmp_path / "out" / "a" / "s.nc").exists()
    main([str(tmp_path / "out"), str(tmp_path / "in")])
    assert "exists, skipped" in capsys.readouterr().out
