"""Degradation kinetics, fade rate analysis, and Remaining Useful Life (RUL) estimation."""


import numpy as np
import pandas as pd

from evguard.utils.config import EOL_SOH_THRESHOLD_PCT
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def compute_degradation_summary(battery_df: pd.DataFrame) -> dict:
    """Compute cycle life degradation metrics for a single battery time-series.

    Calculates:
    - Initial measured capacity and initial SOH
    - Current/final capacity and current SOH
    - Total capacity loss (Ah and %)
    - Average fade rate per 100 cycles
    - Estimated cycles until End-of-Life (80% SOH) via robust linear regression
    """
    valid = battery_df[battery_df.get("is_plausible", True) == True].sort_values("cycle_number")  # noqa: E712
    if len(valid) < 2:
        return {"error": "Insufficient valid cycle records for degradation kinetics analysis."}

    batt_id = str(valid["battery_id"].iloc[0])
    cycles = valid["cycle_number"].values
    capacities = valid["discharge_capacity_ah"].values
    soh_values = valid["soh_pct"].values

    init_cap = float(capacities[0])
    final_cap = float(capacities[-1])
    init_soh = float(soh_values[0])
    final_soh = float(soh_values[-1])
    total_cycles = int(cycles[-1] - cycles[0])

    cap_loss_ah = max(0.0, init_cap - final_cap)
    soh_loss_pct = max(0.0, init_soh - final_soh)

    # Fade rate per 100 cycles
    fade_per_100_cycles = (soh_loss_pct / total_cycles * 100.0) if total_cycles > 0 else 0.0

    # Linear fit to estimate Remaining Useful Life (RUL) to 80% SOH
    # soh = slope * cycle + intercept
    slope, intercept = np.polyfit(cycles, soh_values, 1)

    rul_cycles = None
    eol_cycle_estimate = None
    if slope < -1e-5:
        # 80.0 = slope * target_cycle + intercept  =>  target_cycle = (80.0 - intercept) / slope
        target_cycle = (EOL_SOH_THRESHOLD_PCT - intercept) / slope
        if target_cycle > cycles[-1]:
            rul_cycles = int(round(target_cycle - cycles[-1]))
            eol_cycle_estimate = int(round(target_cycle))
        else:
            rul_cycles = 0  # Already reached or passed EOL
            eol_cycle_estimate = int(round(target_cycle))

    # Thermal correlation (if temperature measurements are available)
    temp_corr = None
    if "avg_temperature_c" in valid.columns:
        valid_temps = valid["avg_temperature_c"].dropna()
        if len(valid_temps) > 5 and valid_temps.std() > 1e-4:
            temp_corr = float(valid["soh_pct"].corr(valid["avg_temperature_c"]))

    return {
        "battery_id": batt_id,
        "total_cycles_recorded": total_cycles + 1,
        "initial_capacity_ah": round(init_cap, 4),
        "current_capacity_ah": round(final_cap, 4),
        "initial_soh_pct": round(init_soh, 2),
        "current_soh_pct": round(final_soh, 2),
        "total_capacity_loss_ah": round(cap_loss_ah, 4),
        "total_soh_loss_pct": round(soh_loss_pct, 2),
        "fade_rate_pct_per_100_cycles": round(fade_per_100_cycles, 3),
        "regression_slope_pct_per_cycle": round(float(slope), 5),
        "estimated_eol_cycle": eol_cycle_estimate,
        "remaining_useful_life_cycles": rul_cycles,
        "temperature_soh_correlation": round(temp_corr, 3) if temp_corr is not None else None,
        "is_eol_reached": final_soh < EOL_SOH_THRESHOLD_PCT,
    }


def compute_fleet_degradation_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze and rank all batteries in the dataset by degradation rate and current health."""
    results: list[dict] = []
    for _batt_id, group in df.groupby("battery_id"):
        summary = compute_degradation_summary(group)
        if "error" not in summary:
            results.append(summary)

    ranking_df = pd.DataFrame(results)
    if not ranking_df.empty:
        ranking_df.sort_values(by="current_soh_pct", ascending=True, inplace=True)
    return ranking_df
