"""Integration tests for end-to-end data generation, loader, and pipeline."""

import pandas as pd

from evguard.data.loader import load_dataset
from evguard.data.synthetic_generator import generate_synthetic_battery_data
from evguard.data.validator import validate_and_preprocess


class TestPipelineIntegration:
    """Test suite for data loading, synthetic generation, and end-to-end flow."""

    def test_synthetic_generator_reproducibility(self, tmp_path):
        """Test deterministic generation creates expected rows and batteries."""
        out_file = tmp_path / "test_demo.csv"
        df = generate_synthetic_battery_data(output_path=out_file, num_cycles_per_battery=20, seed=42)

        assert out_file.exists()
        assert len(df) > 80  # 4 batteries * 20 cycles + duplicates
        assert set(df["battery_id"].unique()) == {"BAT_SIM_01", "BAT_SIM_02", "BAT_SIM_03", "BAT_SIM_04"}
        assert "discharge_capacity_ah" in df.columns
        assert "soh_pct" in df.columns

    def test_loader_csv_fallback(self, tmp_path):
        """Test loader reads CSV correctly."""
        test_csv = tmp_path / "simple_batt.csv"
        df_sample = pd.DataFrame({
            "battery_id": ["B01", "B01"],
            "cycle_number": [1, 2],
            "discharge_capacity_ah": [2.0, 1.95],
        })
        df_sample.to_csv(test_csv, index=False)

        loaded_df = load_dataset(test_csv)
        assert len(loaded_df) == 2
        assert "battery_id" in loaded_df.columns

    def test_full_preprocessing_cycle(self, tmp_path):
        """Test full validation pipeline on synthetic data."""
        raw_df = generate_synthetic_battery_data(output_path=tmp_path / "demo.csv", num_cycles_per_battery=10)
        validated_df, report = validate_and_preprocess(raw_df, report_output_path=tmp_path / "report.json")

        assert len(validated_df) == 40  # 4 * 10 after 1 duplicate removed
        assert (tmp_path / "report.json").exists()
        assert report["total_records_processed"] == 40
        assert "is_plausible" in validated_df.columns
