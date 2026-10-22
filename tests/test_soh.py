"""Unit tests for capacity-based State of Health (SOH) analytics."""

import numpy as np
import pandas as pd
import pytest

from evguard.analytics.soh import (
    calculate_soh,
    calculate_soh_series,
    classify_health_state,
)


class TestSOHCalculations:
    """Test suite for SOH calculation accuracy and input boundaries."""

    def test_nominal_capacity_soh(self):
        """Test standard 2.0 Ah capacity yields exactly 100% SOH."""
        assert calculate_soh(2.0, reference_capacity=2.0) == 100.0
        assert calculate_soh(1.8, reference_capacity=2.0) == 90.0
        assert calculate_soh(1.6, reference_capacity=2.0) == 80.0

    def test_zero_and_negative_capacity(self):
        """Test graceful handling of zero and negative measured capacity."""
        assert calculate_soh(0.0, reference_capacity=2.0) == 0.0
        # Negative capacity should be caught and returned as 0.0 safely
        assert calculate_soh(-0.5, reference_capacity=2.0) == 0.0

    def test_none_and_nan_handling(self):
        """Test NaN or None input produces NaN without throwing exception."""
        assert np.isnan(calculate_soh(None, reference_capacity=2.0))
        assert np.isnan(calculate_soh(np.nan, reference_capacity=2.0))

    def test_invalid_reference_capacity(self):
        """Test non-positive reference capacity raises explicit ValueError."""
        with pytest.raises(ValueError):
            calculate_soh(1.8, reference_capacity=0.0)

        with pytest.raises(ValueError):
            calculate_soh(1.8, reference_capacity=-2.0)

    def test_vectorized_soh_series(self):
        """Test pandas series vectorized SOH calculation."""
        caps = pd.Series([2.0, 1.9, 1.6, np.nan, -0.2])
        res = calculate_soh_series(caps, reference_capacity=2.0)

        assert res.iloc[0] == 100.0
        assert res.iloc[1] == 95.0
        assert res.iloc[2] == 80.0
        assert np.isnan(res.iloc[3])
        assert np.isnan(res.iloc[4])  # Negative capacity masked


class TestHealthClassification:
    """Test battery operational health tier classification."""

    def test_healthy_state(self):
        assert classify_health_state(95.5) == "HEALTHY"
        assert classify_health_state(90.0) == "HEALTHY"

    def test_moderate_wear_state(self):
        assert classify_health_state(89.9) == "MODERATE_WEAR"
        assert classify_health_state(80.0) == "MODERATE_WEAR"

    def test_end_of_life_state(self):
        assert classify_health_state(79.9) == "END_OF_LIFE"
        assert classify_health_state(65.0) == "END_OF_LIFE"

    def test_invalid_and_anomalous_states(self):
        assert classify_health_state(np.nan) == "UNKNOWN"
        assert classify_health_state(None) == "UNKNOWN"
        assert classify_health_state(-10.0) == "INVALID_OR_ANOMALY"
        assert classify_health_state(150.0) == "INVALID_OR_ANOMALY"
