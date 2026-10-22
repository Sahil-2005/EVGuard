"""Dataset loader supporting both CSV format and NASA MATLAB .mat files."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from evguard.utils.config import DEMO_DATA_FILE, NOMINAL_CAPACITY_AH, RAW_DATA_DIR
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(source_path: str | Path | None = None) -> pd.DataFrame:
    """Load battery dataset from a specified path (CSV or .mat).

    If no path is specified or the file does not exist, falls back gracefully to the
    synthetic demo dataset.
    """
    if source_path is None:
        source_path = DEMO_DATA_FILE

    path = Path(source_path)

    if not path.exists():
        logger.warning(f"File {path} does not exist. Checking for raw files or fallback demo.")
        # Check raw directory
        raw_files = list(RAW_DATA_DIR.glob("*.csv")) + list(RAW_DATA_DIR.glob("*.mat"))
        if raw_files:
            path = raw_files[0]
            logger.info(f"Discovered raw dataset file: {path}")
        else:
            logger.info(f"Falling back to default demo dataset: {DEMO_DATA_FILE}")
            path = DEMO_DATA_FILE

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_data(path)
    elif suffix == ".mat":
        return load_matlab_data(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .csv, .mat")


def load_csv_data(file_path: Path) -> pd.DataFrame:
    """Load tabular battery cycle data from a CSV file."""
    logger.info(f"Loading CSV data from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded {len(df)} records from {file_path.name}")
    return df


def load_matlab_data(file_path: Path) -> pd.DataFrame:
    """Parse NASA Li-ion Battery Aging MATLAB (.mat) files.

    Safely inspects dynamic keys in the .mat file (e.g. 'B0005', 'B0006', 'B0018')
    without hardcoding brittle assumptions. Extracts discharge cycles, capacity,
    measured voltages, and temperatures.
    """
    logger.info(f"Inspecting MATLAB .mat structure: {file_path}")
    try:
        mat_dict = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        logger.error(f"Failed to read MATLAB file with SciPy: {e}")
        raise

    # Find the battery key (exclude MATLAB metadata keys like '__header__', '__version__', '__globals__')
    candidate_keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if not candidate_keys:
        raise ValueError(f"No battery struct found in {file_path}. Keys present: {list(mat_dict.keys())}")

    batt_key = candidate_keys[0]
    batt_obj = mat_dict[batt_key]
    logger.info(f"Found root battery object: '{batt_key}'")

    if not hasattr(batt_obj, "cycle"):
        raise ValueError(f"Root object '{batt_key}' does not contain expected 'cycle' field.")

    cycles = batt_obj.cycle
    records: list[dict] = []
    discharge_cycle_counter = 0

    # cycles can be a list or 1D ndarray of cycle objects
    cycle_list = cycles if isinstance(cycles, (list, np.ndarray)) else [cycles]

    for item in cycle_list:
        cycle_type = getattr(item, "type", "unknown")
        if str(cycle_type).strip().lower() != "discharge":
            continue

        discharge_cycle_counter += 1
        data_block = getattr(item, "data", None)
        if data_block is None:
            continue

        # Extract scalar Capacity if available
        cap = getattr(data_block, "Capacity", None)
        if isinstance(cap, (list, np.ndarray)) and len(cap) > 0:
            cap_val = float(cap[0])
        elif isinstance(cap, (int, float, np.floating)):
            cap_val = float(cap)
        else:
            cap_val = np.nan

        # Extract time-series arrays for voltage and temperature during this discharge
        voltages = getattr(data_block, "Voltage_measured", np.array([]))
        temps = getattr(data_block, "Temperature_measured", np.array([]))

        avg_v = float(np.mean(voltages)) if len(voltages) > 0 else np.nan
        min_v = float(np.min(voltages)) if len(voltages) > 0 else np.nan
        max_v = float(np.max(voltages)) if len(voltages) > 0 else np.nan
        avg_temp = float(np.mean(temps)) if len(temps) > 0 else np.nan
        max_temp = float(np.max(temps)) if len(temps) > 0 else np.nan

        # Calculate SOH based on reference capacity
        soh = (cap_val / NOMINAL_CAPACITY_AH) * 100.0 if not np.isnan(cap_val) else np.nan

        records.append({
            "battery_id": batt_key,
            "cycle_number": discharge_cycle_counter,
            "discharge_capacity_ah": round(cap_val, 4) if not np.isnan(cap_val) else np.nan,
            "soh_pct": round(soh, 2) if not np.isnan(soh) else np.nan,
            "avg_voltage_v": round(avg_v, 3) if not np.isnan(avg_v) else np.nan,
            "min_voltage_v": round(min_v, 3) if not np.isnan(min_v) else np.nan,
            "max_voltage_v": round(max_v, 3) if not np.isnan(max_v) else np.nan,
            "avg_temperature_c": round(avg_temp, 2) if not np.isnan(avg_temp) else np.nan,
            "max_temperature_c": round(max_temp, 2) if not np.isnan(max_temp) else np.nan,
            "data_source": "NASA_MATLAB_MAT",
        })

    df = pd.DataFrame(records)
    logger.info(f"Extracted {len(df)} discharge cycles from MATLAB structure '{batt_key}'")
    return df
