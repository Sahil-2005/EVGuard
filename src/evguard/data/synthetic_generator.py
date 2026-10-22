"""Deterministic synthetic battery aging dataset generator for EVGuard demo mode.

Generates realistic battery cycle metrics modeled after standard 18650 Li-ion cells
from the NASA Battery Aging dataset (nominal capacity ~2.0 Ah, discharge cutoff ~2.7V).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from evguard.utils.config import (
    DEMO_DATA_FILE,
    NOMINAL_CAPACITY_AH,
)
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_battery_data(
    output_path: Path | None = None,
    num_cycles_per_battery: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic battery aging time-series for 4 simulated batteries.

    Batteries simulated:
    - BAT_SIM_01: Normal nominal aging (room temp ~24C, gradual SEI growth).
    - BAT_SIM_02: Thermal stress aging (higher operating temp ~38C, faster capacity loss).
    - BAT_SIM_03: Fast-discharge wear (moderate degradation with resting recovery bursts).
    - BAT_SIM_04: End-of-Life transition (crosses the 80% SOH threshold around cycle 90).

    Includes subtle physical phenomena:
    - Capacity regeneration / self-recovery after rest periods.
    - Impedance (internal resistance) rise as electrolyte degrades.
    - Controlled sensor glitches to demonstrate data validation & anomaly flagging.
    """
    if output_path is None:
        output_path = DEMO_DATA_FILE

    rng = np.random.default_rng(seed)
    records = []
    base_time = datetime(2025, 1, 15, 8, 0, 0)

    battery_profiles = [
        {"id": "BAT_SIM_01", "init_cap": 2.02, "fade_rate": 0.0022, "temp_mean": 27.5, "temp_std": 1.5, "stress": "normal"},
        {"id": "BAT_SIM_02", "init_cap": 1.98, "fade_rate": 0.0036, "temp_mean": 38.0, "temp_std": 2.8, "stress": "thermal"},
        {"id": "BAT_SIM_03", "init_cap": 2.05, "fade_rate": 0.0025, "temp_mean": 30.0, "temp_std": 1.8, "stress": "moderate"},
        {"id": "BAT_SIM_04", "init_cap": 1.92, "fade_rate": 0.0042, "temp_mean": 33.5, "temp_std": 2.2, "stress": "accelerated"},
    ]

    for prof in battery_profiles:
        batt_id = prof["id"]
        init_cap = prof["init_cap"]
        fade_rate = prof["fade_rate"]
        temp_mean = prof["temp_mean"]
        curr_cap = init_cap

        for cycle in range(1, num_cycles_per_battery + 1):
            timestamp = base_time + timedelta(hours=cycle * 3)

            # Degradation kinetics with non-linear accelerated wear near EOL
            linear_fade = fade_rate * (1.0 + (cycle / num_cycles_per_battery) ** 1.5)
            # Occasional capacity regeneration after resting (observed in NASA datasets)
            rest_rebound = 0.008 if (cycle % 18 == 0) else 0.0
            noise = rng.normal(0.0, 0.0025)

            curr_cap = max(0.5, curr_cap - linear_fade + rest_rebound + noise)

            # Capacity-based State of Health (%)
            calculated_soh = (curr_cap / NOMINAL_CAPACITY_AH) * 100.0

            # Internal resistance rises as cell degrades (from ~0.06 up to 0.35 Ohm)
            degrade_ratio = max(0.0, (init_cap - curr_cap) / init_cap)
            internal_r = 0.065 + (degrade_ratio * 0.28) + rng.normal(0.0, 0.003)

            # Voltages during CC-CV discharge
            avg_v = 3.65 - (degrade_ratio * 0.25) + rng.normal(0.0, 0.02)
            min_v = 2.70 - (degrade_ratio * 0.20) + rng.normal(0.0, 0.015)
            max_v = 4.20 + rng.normal(0.0, 0.01)

            # Operational temperatures
            avg_temp = temp_mean + (degrade_ratio * 4.5) + rng.normal(0.0, prof["temp_std"])
            max_temp = avg_temp + rng.uniform(3.0, 7.5)

            # Discharge duration (seconds) ~ capacity / current
            discharge_time_s = int(curr_cap * 1800 + rng.normal(0, 30))

            records.append({
                "battery_id": batt_id,
                "cycle_number": cycle,
                "timestamp": timestamp.isoformat(),
                "discharge_capacity_ah": round(float(curr_cap), 4),
                "soh_pct": round(float(calculated_soh), 2),
                "avg_voltage_v": round(float(avg_v), 3),
                "min_voltage_v": round(float(min_v), 3),
                "max_voltage_v": round(float(max_v), 3),
                "avg_temperature_c": round(float(avg_temp), 2),
                "max_temperature_c": round(float(max_temp), 2),
                "internal_resistance_ohm": round(float(internal_r), 4),
                "discharge_duration_s": max(300, discharge_time_s),
                "data_source": "SIMULATED_DEMO",
                "stress_condition": prof["stress"],
            })

    df = pd.DataFrame(records)

    # Inject controlled anomalies to test validation layer (properly flagged later)
    # 1. Implausible high temperature sensor fault
    df.loc[df["cycle_number"] == 77, "max_temperature_c"] = 98.5
    # 2. Implausible low voltage reading
    df.loc[df["cycle_number"] == 115, "min_voltage_v"] = 1.15
    # 3. Duplicate measurement for test battery
    dup_row = df.iloc[[10]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic demo dataset created at {output_path} with {len(df)} records.")
    return df
