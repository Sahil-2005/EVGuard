"""Data validation, anomaly flagging, and feature engineering for battery records."""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from evguard.utils.config import (
    DATA_QUALITY_REPORT_FILE,
    EOL_SOH_THRESHOLD_PCT,
    NOMINAL_CAPACITY_AH,
    PLAUSIBLE_RANGES,
)
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def validate_and_preprocess(
    df: pd.DataFrame,
    report_output_path: Path = DATA_QUALITY_REPORT_FILE,
) -> Tuple[pd.DataFrame, Dict]:
    """Validate schema, detect duplicates, flag physical anomalies, and engineer features.

    Does NOT silently delete records; instead flags them transparently with
    'is_plausible' and 'anomaly_reason' columns.
    """
    logger.info("Starting battery dataset validation and quality assessment...")
    initial_rows = len(df)
    quality_metrics: Dict = {
        "total_records_input": initial_rows,
        "duplicate_records_removed": 0,
        "missing_values_per_column": {},
        "anomaly_counts": {},
        "plausible_records_count": 0,
        "anomalous_records_count": 0,
        "batteries_analyzed": [],
    }

    # 1. Schema check
    required_cols = ["battery_id", "cycle_number", "discharge_capacity_ah"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset missing mandatory schema columns: {missing_required}")

    # 2. Transparent duplicate handling (deduplicate on battery_id + cycle_number)
    dup_mask = df.duplicated(subset=["battery_id", "cycle_number"], keep="first")
    num_duplicates = int(dup_mask.sum())
    quality_metrics["duplicate_records_removed"] = num_duplicates
    if num_duplicates > 0:
        logger.warning(f"Detected {num_duplicates} duplicate battery cycle records. Preserving first occurrence.")
        df = df[~dup_mask].copy()
    else:
        df = df.copy()

    # 3. Missing values analysis
    missing_counts = df.isnull().sum().to_dict()
    quality_metrics["missing_values_per_column"] = {k: int(v) for k, v in missing_counts.items() if v > 0}

    # 4. Physical Plausibility Flagging (Do not silently drop)
    df["is_plausible"] = True
    df["anomaly_reason"] = "VALID"

    # Rule A: Capacity bounds
    cap_min, cap_max = PLAUSIBLE_RANGES["capacity_ah"]
    bad_cap_mask = (df["discharge_capacity_ah"] < cap_min) | (df["discharge_capacity_ah"] > cap_max) | df["discharge_capacity_ah"].isna()
    if bad_cap_mask.any():
        df.loc[bad_cap_mask, "is_plausible"] = False
        df.loc[bad_cap_mask, "anomaly_reason"] = "IMPLAUSIBLE_CAPACITY"

    # Rule B: Temperature bounds (if present)
    if "max_temperature_c" in df.columns:
        temp_min, temp_max = PLAUSIBLE_RANGES["temperature_c"]
        bad_temp_mask = (df["max_temperature_c"] < temp_min) | (df["max_temperature_c"] > temp_max)
        if bad_temp_mask.any():
            df.loc[bad_temp_mask, "is_plausible"] = False
            df.loc[bad_temp_mask, "anomaly_reason"] = "IMPLAUSIBLE_TEMPERATURE"

    # Rule C: Voltage bounds (if present)
    if "min_voltage_v" in df.columns:
        v_min, v_max = PLAUSIBLE_RANGES["voltage_v"]
        bad_v_mask = (df["min_voltage_v"] < v_min) | (df["min_voltage_v"] > v_max)
        if bad_v_mask.any():
            df.loc[bad_v_mask, "is_plausible"] = False
            df.loc[bad_v_mask, "anomaly_reason"] = "IMPLAUSIBLE_VOLTAGE"

    anomalies_count = int((~df["is_plausible"]).sum())
    plausible_count = int(df["is_plausible"].sum())
    quality_metrics["plausible_records_count"] = plausible_count
    quality_metrics["anomalous_records_count"] = anomalies_count
    quality_metrics["anomaly_counts"] = df["anomaly_reason"].value_counts().to_dict()

    # 5. Feature Engineering
    # Calculate SOH % if missing or refresh with standard definition
    if "soh_pct" not in df.columns or df["soh_pct"].isnull().any():
        df["soh_pct"] = (df["discharge_capacity_ah"] / NOMINAL_CAPACITY_AH) * 100.0

    # Ensure internal resistance exists (or estimate from voltage drop / capacity if needed)
    if "internal_resistance_ohm" not in df.columns:
        # Fallback approximation for external datasets without impedance
        df["internal_resistance_ohm"] = 0.08 + (100.0 - df["soh_pct"].clip(lower=0, upper=100)) * 0.002

    # Derived degradation features
    df["is_below_eol"] = df["soh_pct"] < EOL_SOH_THRESHOLD_PCT

    if "max_temperature_c" in df.columns and "avg_temperature_c" in df.columns:
        df["thermal_gradient_c"] = (df["max_temperature_c"] - df["avg_temperature_c"]).round(2)
    else:
        df["thermal_gradient_c"] = 0.0

    # Sort deterministically
    df.sort_values(by=["battery_id", "cycle_number"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    quality_metrics["batteries_analyzed"] = sorted(df["battery_id"].unique().tolist())
    quality_metrics["total_records_processed"] = len(df)

    # Save quality summary report
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(quality_metrics, f, indent=2)

    logger.info(
        f"Validation complete: {plausible_count} plausible records, "
        f"{anomalies_count} flagged anomalies, {num_duplicates} duplicates resolved."
    )
    return df, quality_metrics
