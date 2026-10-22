"""Evaluation utilities and cluster profiling diagnostics for EVGuard."""

from typing import Dict
import pandas as pd


def generate_cluster_diagnostics(clustered_df: pd.DataFrame) -> Dict:
    """Compute detailed summary statistics for each identified battery cluster.

    Generates mean, min, max, standard deviation, and percentage of cycles
    per cluster to explain operational regimes safely.
    """
    if "cluster_label" not in clustered_df.columns:
        return {"error": "DataFrame is missing cluster_label column"}

    cols_to_profile = [
        "soh_pct",
        "discharge_capacity_ah",
        "avg_temperature_c",
        "max_temperature_c",
        "internal_resistance_ohm",
    ]
    present_cols = [c for c in cols_to_profile if c in clustered_df.columns]

    diagnostics = {}
    total_records = len(clustered_df)

    for label, group in clustered_df.groupby("cluster_label"):
        group_size = len(group)
        pct_of_total = round((group_size / total_records) * 100.0, 1)

        profile = {
            "record_count": group_size,
            "record_percentage": pct_of_total,
            "metrics": {},
        }

        for col in present_cols:
            profile["metrics"][col] = {
                "mean": round(float(group[col].mean()), 3),
                "std": round(float(group[col].std()), 3),
                "min": round(float(group[col].min()), 3),
                "max": round(float(group[col].max()), 3),
            }

        diagnostics[str(label)] = profile

    return diagnostics
