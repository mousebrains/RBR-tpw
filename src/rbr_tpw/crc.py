"""CRC-16/CCITT as used by RBR's `read data` transfers and event records.

Polynomial 0x1021, seed 0xFFFF, MSB-first, no final XOR (a.k.a. CRC-16/CCITT-FALSE,
which is what binascii.crc_hqx computes). The logger appends the CRC big-endian,
so the CRC over data+crc is 0. Verified on a real RBRsolo (fwtype 9) download
on 2026-09-25.
"""

import binascii


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    return binascii.crc_hqx(data, crc)


def check_appended(data_with_crc: bytes) -> bool:
    """True if the trailing two bytes are a valid big-endian CRC of what precedes them."""
    return len(data_with_crc) >= 2 and crc16_ccitt(data_with_crc) == 0
