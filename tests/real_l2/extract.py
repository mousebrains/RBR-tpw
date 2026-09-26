"""Make an anonymised real-data fixture from a Ruskin .rsk of an RBRduet/RBRconcerto (fwtype 102/103).

    uv run python tests/real_l2/extract.py FILE.rsk NAME [--samples K]

Writes NAME.bin: the logger's memory image as Ruskin downloaded it, with the serial number (header byte 16,
its only occurrence) zeroed, the header CRC recomputed, and optionally only the first K sample sets. Writes
NAME.json: a minimal offload record for ncwrite.write_netcdf, and Ruskin's own results for those samples
(logger times, values, coefficients, events other than Ruskin's own type 307, and errors) to compare with.
Nothing else from the .rsk (file names, Ruskin parameters, hashes of the source) is kept.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from rbr_tpw import rsk
from rbr_tpw.crc import crc16_ccitt
from rbr_tpw.equations import decode_l2, parse_l2_header

SERIAL_OFFSET = 16  # uint32 little-endian in the deployment header (section 2)


def anonymise(image: bytes, serial: int) -> bytes:
    length = parse_l2_header(image).length
    img = bytearray(image)
    if struct.unpack_from("<I", img, SERIAL_OFFSET)[0] != serial:
        raise SystemExit(f"the serial number is not at header byte {SERIAL_OFFSET}")
    struct.pack_into("<I", img, SERIAL_OFFSET, 0)
    struct.pack_into(">H", img, length - 2, crc16_ccitt(bytes(img[: length - 2])))
    if struct.pack("<I", serial) in img or str(serial).encode() in img:
        raise SystemExit("the serial number occurs elsewhere in the image too")
    return bytes(img)


def truncate(image: bytes, nstored: int, k: int) -> bytes:
    """The shortest prefix that decodes to exactly k sample sets (a whole number of body words)."""
    length = parse_l2_header(image).length
    for end in range(length + 4 * nstored * k, len(image) + 1, 4):
        if len(decode_l2(image[:end], nstored).time_ms) == k:
            return image[:end]
    raise SystemExit(f"cannot cut at {k} sample sets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rsk", type=Path)
    ap.add_argument("name")
    ap.add_argument("--samples", type=int)
    args = ap.parse_args()
    con = rsk.connect(args.rsk)
    r = rsk.read_rsk(args.rsk, con)
    if r.fwtype not in (102, 103):
        raise SystemExit(f"fwtype {r.fwtype}: only RBRduet/RBRconcerto (102/103)")
    image = anonymise(rsk.read_download(con), r.serial)
    t, v = rsk.read_values(con, r)
    k = args.samples or len(t)
    whole = k >= len(t)  # then every event is kept, including those after the last sample
    if not whole:
        image = truncate(image, len(r.stored), k)
        t, v = t[:k], v[:k]

    chans = [{"index": c.order, "type": c.short_name, "equation": c.equation, "status": c.status,
              "coefficients": dict(c.coefficients), "calibration_datetime": ""} for c in r.channels]
    full = rsk.make_record(r, "decode", image)
    record = {
        "offload_started": full["offload_started"], "offload_finished": full["offload_started"],
        "id": {"model": r.model, "version": r.firmware, "serial": "000000", "fwtype": r.fwtype},
        "host_ntp": {}, "clock_skew": full["clock_skew"],
        "snapshot_before": {"status": r.status, "sampling": {"mode": r.mode, "period": str(r.period_ms)},
                            "channel_list": [c for c in chans if not c["status"] & 0x04], "channels_all": chans,
                            "settings": {}},
        "raw": {"file": f"{args.name}.bin", "bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()},
        "warnings": [],
    }
    events = [[int(e[0]), int(e[1]), int(e[2])] for e in con.execute(
        "select tstamp, type, sampleIndex from events where type != 307 order by rowid") if whole or e[2] <= k]
    errors = [[int(e[0]), int(e[1])] for e in con.execute(
        "select sampleIndex, channelOrder from errors order by rowid") if e[0] <= k]
    ruskin = {"tstamp": t.tolist(), "stored_order": [c.order for c in r.stored],
              "values": [[None if np.isnan(x) else float(x) for x in col] for col in v.T],
              "coefficients": {str(c.order): dict(c.coefficients) for c in r.stored},
              "drift_ms": r.drift_ms, "events": events, "errors": errors}
    out = Path(__file__).parent
    (out / f"{args.name}.bin").write_bytes(image)
    (out / f"{args.name}.json").write_text(json.dumps(
        {"about": f"{r.model} (fwtype {r.fwtype}) memory from a Ruskin {r.ruskin_version} .rsk; serial zeroed; "
                  f"{k} sample sets. Made by extract.py.", "record": record, "ruskin": ruskin}, indent=1) + "\n")
    print(f"{args.name}: {len(image)} bytes, {k} sample sets, {len(events)} events, {len(errors)} errors")


if __name__ == "__main__":
    main()
