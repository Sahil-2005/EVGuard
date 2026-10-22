"""Unit tests for K-Means battery cycle clustering and evaluation."""

import numpy as np
import pandas as pd
import pytest

from evguard.models.clustering import BatteryClusterModel
from evguard.models.evaluator import generate_cluster_diagnostics


@pytest.fixture
def sample_battery_data():
    """Create deterministic sample battery dataset for clustering tests."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "battery_id": ["B01"] * 30 + ["B02"] * 30,
        "cycle_number": list(range(1, 31)) * 2,
        "discharge_capacity_ah": np.linspace(2.0, 1.4, n),
        "soh_pct": np.linspace(100.0, 70.0, n),
        "avg_temperature_c": np.random.uniform(25.0, 38.0, n),
        "max_temperature_c": np.random.uniform(30.0, 44.0, n),
        "avg_voltage_v": np.linspace(3.7, 3.4, n),
        "internal_resistance_ohm": np.linspace(0.06, 0.25, n),
        "is_plausible": [True] * n,
    })


class TestClustering:
    """Test suite for machine learning clustering logic and metrics."""

    def test_model_fitting_and_clusters_count(self, sample_battery_data):
        """Test model fits successfully and creates 3 clusters."""
        model = BatteryClusterModel(n_clusters=3, random_state=42)
        model.fit(sample_battery_data)

        assert model.is_fitted
        assert len(model.evaluation_metrics["cluster_profiles"]) == 3
        assert "silhouette_score" in model.evaluation_metrics
        assert "davies_bouldin_index" in model.evaluation_metrics
        assert model.evaluation_metrics["silhouette_score"] > -1.0

    def test_prediction_annotations(self, sample_battery_data):
        """Test prediction appends cluster_id and cluster_label columns."""
        model = BatteryClusterModel(n_clusters=3, random_state=42)
        model.fit(sample_battery_data)
        annotated = model.predict(sample_battery_data)

        assert "cluster_id" in annotated.columns
        assert "cluster_label" in annotated.columns
        assert set(annotated["cluster_id"].unique()).issubset({0, 1, 2})
        assert all(isinstance(lbl, str) for lbl in annotated["cluster_label"])

    def test_elbow_curve_computation(self, sample_battery_data):
        """Test computation of inertia across multiple k values."""
        model = BatteryClusterModel(n_clusters=3, random_state=42)
        model.fit(sample_battery_data)
        elbow = model.compute_elbow_curve(sample_battery_data, max_k=4)

        assert set(elbow.keys()) == {2, 3, 4}
        # Inertia should be strictly decreasing as k increases
        assert elbow[2] >= elbow[3] >= elbow[4]

    def test_cluster_diagnostics_generation(self, sample_battery_data):
        """Test diagnostic summaries by cluster."""
        model = BatteryClusterModel(n_clusters=3, random_state=42)
        model.fit(sample_battery_data)
        annotated = model.predict(sample_battery_data)

        diag = generate_cluster_diagnostics(annotated)
        assert len(diag) == 3
        for _cluster_name, info in diag.items():
            assert "record_count" in info
            assert "record_percentage" in info
            assert "metrics" in info
