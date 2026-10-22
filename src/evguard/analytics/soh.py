"""Capacity-based State of Health (SOH) calculation and health state categorization.

Mathematical Foundation:
    SOH (%) = (Current Usable Capacity / Reference Initial Capacity) * 100

Reference Capacity Justification:
    - Absolute Nominal Standard: 2.0 Ah (Manufacturer rated capacity for 18650 cells
      in NASA battery aging benchmarks). Using nominal capacity enables uniform
      pack-level comparisons.
    - Initial Cycle Baseline: In cells with manufacturing variations, the cycle 1
      capacity can optionally be used as the cell-specific reference.

Note:
    Calculated SOH represents physical electrochemical capacity ratio and is
    strictly distinguished from machine learning cluster classifications or
    statistical trend predictions.
"""


import numpy as np
import pandas as pd

from evguard.utils.config import EOL_SOH_THRESHOLD_PCT, NOMINAL_CAPACITY_AH
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_soh(
    usable_capacity: float | None,
    reference_capacity: float = NOMINAL_CAPACITY_AH,
) -> float:
    """Calculate capacity-based State of Health (%) safely for a single scalar.

    Parameters
    ----------
    usable_capacity : float, optional
        Current measured discharge capacity in Ampere-hours (Ah).
    reference_capacity : float
        Nominal or initial reference capacity in Ah. Must be strictly positive.

    Returns
    -------
    float
        State of Health percentage (e.g. 95.4%). Returns NaN if input is invalid.
    """
    if reference_capacity <= 0:
        raise ValueError(f"Reference capacity must be strictly positive, got {reference_capacity}")

    if usable_capacity is None or np.isnan(usable_capacity):
        return float(np.nan)

    if usable_capacity < 0:
        logger.warning(f"Negative usable capacity encountered ({usable_capacity} Ah). Flagged as 0.0 SOH.")
        return 0.0

    soh = (usable_capacity / reference_capacity) * 100.0
    return round(float(soh), 2)


def calculate_soh_series(
    capacity_series: pd.Series | np.ndarray,
    reference_capacity: float = NOMINAL_CAPACITY_AH,
) -> pd.Series:
    """Vectorized calculation of capacity-based State of Health (%)."""
    if reference_capacity <= 0:
        raise ValueError(f"Reference capacity must be strictly positive, got {reference_capacity}")

    s = pd.Series(capacity_series).copy()
    # Mask invalid values
    valid_mask = s.notna() & (s >= 0)
    soh_series = pd.Series(np.nan, index=s.index)
    soh_series[valid_mask] = (s[valid_mask] / reference_capacity) * 100.0
    return soh_series.round(2)


def classify_health_state(soh_pct: float) -> str:
    """Classify battery health into engineering operational stages.

    Categories:
    - 'HEALTHY' (SOH >= 90%): Optimal operational condition, minimal internal impedance.
    - 'MODERATE_WEAR' (80% <= SOH < 90%): Normal degradation; within safe EV operational envelope.
    - 'END_OF_LIFE' (SOH < 80%): Reached standard EV retirement threshold (candidate for 2nd-life/recycling).
    - 'INVALID_OR_ANOMALY': SOH is NaN, negative, or implausibly high (>130%).
    """
    if soh_pct is None or np.isnan(soh_pct):
        return "UNKNOWN"
    if soh_pct < 0 or soh_pct > 130.0:
        return "INVALID_OR_ANOMALY"
    if soh_pct >= 90.0:
        return "HEALTHY"
    if soh_pct >= EOL_SOH_THRESHOLD_PCT:
        return "MODERATE_WEAR"
    return "END_OF_LIFE"
