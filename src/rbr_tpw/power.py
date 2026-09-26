"""Remaining sampling time: memory-limited (exact) and energy-limited (modelled).

Energy model for the RBRsolo T (fwtype 9), from Ruskin 2.26.1 constants:
  - battery: 1 x AA Li-SOCl2, 3.6 V, 33,696 J nominal, 30,326 J derated (Ruskin `Battery`
    LITHIUM_THIONYL_CHLORIDE_AA);
  - active current 0.69 mA while a sample is taken, for sampling at 2 Hz or slower (Ruskin
    `SL2PowerValues.SOLO_T_L100_R350`, "intermittent"; this matches the solo's channel latency of
    100 ms and read time of 350 ms);
  - sleep current 0.0055 mA (Ruskin `SL2PowerValues.SLEEP`).
How Ruskin combines these is not verified. The formula below (active current x (latency +
read time) per sample, plus sleep current for the rest of the day) is our own and should be checked
against Ruskin's "estimated end" for a few sampling periods.

Derating is proportional: usable energy = counter x DERATED_J / NOMINAL_J (90%), so a cell whose
counter was pro-rated to a fraction f of a new cell (configure.DeployConfig.battery_days_used) gets
f of the derated capacity.

Voltage says little about remaining capacity for Li-SOCl2. The discharge curve is flat near 3.6 V
until close to exhaustion, so a voltage threshold only catches dead, disconnected or nearly exhausted
cells.
"""

from __future__ import annotations

import math

CELL_V = 3.6
NOMINAL_J = 33_696.0
DERATED_J = 30_326.0
ACTIVE_MA = 0.69
SLEEP_MA = 0.0055
HEADER_BYTES = 512
BYTES_PER_READING = 4


def energy_per_day_J(period_ms: int, active_ms: float) -> float:
    samples = 86_400_000 / period_ms
    active_s = min(samples * active_ms / 1000, 86_400.0)
    return CELL_V * (ACTIVE_MA * active_s + SLEEP_MA * (86_400.0 - active_s)) / 1000


def memory_days(remaining_bytes: int, period_ms: int, nchan: int, bytes_per_sample: int | None = None) -> float:
    return remaining_bytes / ((bytes_per_sample or BYTES_PER_READING * nchan) * 86_400_000 / period_ms)


def remaining(counter_J: float, used_bytes: int, remaining_bytes: int, period_ms: int, nchan: int,
              active_ms: float, bytes_per_sample: int | None = None, energy_model: bool = True) -> dict:
    """Days of sampling left at `period_ms`, limited by memory or by energy.

    The logger's energy counter does not appear to fall during a deployment (it was unchanged over
    10 days of logging on SN100685), so the modelled energy for the samples already in memory is
    subtracted from it. The energy model is the RBRsolo T's; for other loggers, or without a
    counter (fwtype 0), only the memory limit is computed.
    """
    bps = bytes_per_sample or BYTES_PER_READING * nchan
    m_days = memory_days(remaining_bytes, period_ms, nchan, bps)
    if not energy_model or not math.isfinite(counter_J):
        return {"energy_per_day_J": math.nan, "energy_used_this_deployment_J": math.nan,
                "energy_usable_J": math.nan, "energy_days": math.nan, "memory_days": m_days, "days": m_days,
                "limited_by": "memory (energy not modelled)", "derating": "proportional"}
    per_day = energy_per_day_J(period_ms, active_ms)
    samples = max(0, used_bytes - HEADER_BYTES) / bps
    used_this_deployment = per_day * samples * period_ms / 86_400_000
    usable = counter_J * DERATED_J / NOMINAL_J - used_this_deployment
    e_days = max(0.0, usable) / per_day if math.isfinite(usable) else math.nan
    limit = "energy" if e_days < m_days else "memory"
    return {
        "energy_per_day_J": per_day,
        "energy_used_this_deployment_J": used_this_deployment,
        "energy_usable_J": usable,
        "energy_days": e_days,
        "memory_days": m_days,
        "days": min(e_days, m_days),
        "limited_by": limit,
        "derating": "proportional",  # records from before 2026-09-25 lack this key: fixed 3,370 J then
    }
