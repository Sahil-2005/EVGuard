"""Central configuration and hardware-tailored constants for EVGuard."""

import os
import sys
from pathlib import Path

# Base Paths (resolve dynamically relative to this file)
# config.py is at: <PROJECT_ROOT>/src/evguard/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "evguard" / "utils"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Files
DEMO_DATA_FILE = DEMO_DATA_DIR / "battery_aging_demo.csv"
PROCESSED_PARQUET_DIR = PROCESSED_DATA_DIR / "battery_cycles.parquet"
KMEANS_MODEL_FILE = ARTIFACTS_DIR / "kmeans_model.joblib"
KMEANS_METRICS_FILE = ARTIFACTS_DIR / "clustering_metrics.json"
DATA_QUALITY_REPORT_FILE = REPORTS_DIR / "data_quality_summary.json"

# Hardware Constraints (Optimized for dual-core i3, 12GB RAM, Windows 64-bit)
SPARK_MASTER = "local[2]"
SPARK_DRIVER_MEMORY = "2g"
SPARK_SHUFFLE_PARTITIONS = "2"
SPARK_DRIVER_HOST = "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = "127.0.0.1"

# JDK 17 preferred path on this system
PREFERRED_JDK_17 = Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")

# NASA Li-ion 18650 Battery Domain Defaults
NOMINAL_CAPACITY_AH = 2.0  # Reference initial rated capacity for standard 18650 cells (e.g. B0005, B0006)
EOL_SOH_THRESHOLD_PCT = 80.0  # Industry standard End-of-Life criterion for EV battery packs (80% SOH)

# Physical Plausibility Thresholds for Validation
PLAUSIBLE_RANGES = {
    "voltage_v": (2.0, 4.5),       # Typical 18650 cutoff ~2.5V, charge max 4.25V
    "temperature_c": (0.0, 75.0),  # Safe operational range; >60°C indicates high thermal stress
    "capacity_ah": (0.2, 2.5),     # Usable discharge capacity in Ampere-hours
    "internal_resistance_ohm": (0.01, 1.0), # Normal healthy ~0.05-0.1 Ohm, degraded ~0.2-0.5+ Ohm
}

# Machine Learning Defaults
KMEANS_N_CLUSTERS = 3
KMEANS_RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "discharge_capacity_ah",
    "soh_pct",
    "avg_temperature_c",
    "max_temperature_c",
    "avg_voltage_v",
    "internal_resistance_ohm",
]
