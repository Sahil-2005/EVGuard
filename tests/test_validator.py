"""Unit tests for schema validation, transparent deduplication, and anomaly flagging."""

import pandas as pd
import pytest

from evguard.data.validator import validate_and_preprocess


class TestValidator:
    """Test suite for data preprocessing and validation rules."""

    def test_missing_mandatory_schema_columns(self):
        """Test dataset without required columns raises ValueError."""
        invalid_df = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(ValueError) as exc:
            validate_and_preprocess(invalid_df)
        assert "missing mandatory schema columns" in str(exc.value)

    def test_transparent_duplicate_detection(self):
        """Test duplicates by (battery_id, cycle_number) are identified and recorded."""
        df = pd.DataFrame({
            "battery_id": ["BATT_01", "BATT_01", "BATT_01"],
            "cycle_number": [1, 1, 2],
            "discharge_capacity_ah": [2.0, 2.0, 1.95],
            "max_temperature_c": [28.0, 28.0, 29.0],
            "min_voltage_v": [2.7, 2.7, 2.7],
        })
        cleaned_df, report = validate_and_preprocess(df)

        assert report["duplicate_records_removed"] == 1
        assert len(cleaned_df) == 2

    def test_physical_anomaly_flagging(self):
        """Test implausible temperatures and voltages are flagged without silent deletion."""
        df = pd.DataFrame({
            "battery_id": ["BATT_01", "BATT_01", "BATT_01"],
            "cycle_number": [1, 2, 3],
            "discharge_capacity_ah": [2.0, 1.95, 1.90],
            "max_temperature_c": [28.0, 99.5, 30.0],  # Cycle 2 has implausible temp
            "min_voltage_v": [2.7, 2.7, 1.1],         # Cycle 3 has implausible voltage
        })
        cleaned_df, report = validate_and_preprocess(df)

        assert len(cleaned_df) == 3  # Not silently deleted!
        assert cleaned_df.loc[cleaned_df["cycle_number"] == 1, "is_plausible"].iloc[0] == True  # noqa: E712
        assert cleaned_df.loc[cleaned_df["cycle_number"] == 2, "is_plausible"].iloc[0] == False  # noqa: E712
        assert cleaned_df.loc[cleaned_df["cycle_number"] == 2, "anomaly_reason"].iloc[0] == "IMPLAUSIBLE_TEMPERATURE"
        assert cleaned_df.loc[cleaned_df["cycle_number"] == 3, "is_plausible"].iloc[0] == False  # noqa: E712
        assert cleaned_df.loc[cleaned_df["cycle_number"] == 3, "anomaly_reason"].iloc[0] == "IMPLAUSIBLE_VOLTAGE"
        assert report["anomalous_records_count"] == 2

    def test_feature_engineering_outputs(self):
        """Test derived features like is_below_eol and soh_pct are generated."""
        df = pd.DataFrame({
            "battery_id": ["BATT_01", "BATT_01"],
            "cycle_number": [1, 2],
            "discharge_capacity_ah": [2.0, 1.5],  # 1.5 Ah = 75% SOH < 80% EOL
            "avg_temperature_c": [25.0, 26.0],
            "max_temperature_c": [28.0, 31.0],
            "min_voltage_v": [2.7, 2.6],
        })
        cleaned_df, _ = validate_and_preprocess(df)

        assert "is_below_eol" in cleaned_df.columns
        assert "thermal_gradient_c" in cleaned_df.columns
        assert cleaned_df.loc[0, "is_below_eol"] == False  # noqa: E712
        assert cleaned_df.loc[1, "is_below_eol"] == True  # noqa: E712
        assert cleaned_df.loc[1, "thermal_gradient_c"] == 5.0
