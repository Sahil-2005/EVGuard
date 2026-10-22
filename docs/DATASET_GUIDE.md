# EVGuard Dataset Guide

## 1. Overview
EVGuard supports two data modes:
1. **Official NASA Li-ion Battery Aging Dataset** (MATLAB `.mat` or pre-extracted `.csv`)
2. **Deterministic Synthetic Demo Dataset** (`data/demo/battery_aging_demo.csv`)

This architecture guarantees that the application runs and tests successfully out of the box without requiring internet downloads, while providing seamless support for actual experimental NASA data.

---

## 2. NASA Li-ion Battery Aging Dataset

### Source Information
- **Publisher**: NASA Ames Prognostics Center of Excellence (PCoE)
- **Portal Link**: [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- **Cell Chemistry**: Commercial 18650 Lithium Nickel Cobalt Aluminum Oxide (LiNiCoAlO2 / NCA) cells
- **Nominal Rated Capacity**: 2.0 Ah
- **Nominal Voltage**: 3.7 V (Charging up to 4.2 V, discharging down to 2.7 V, 2.5 V, 2.2 V)

### Benchmark Battery IDs
- **B0005**: Cycled at 24°C ambient room temperature (CC-CV charging at 1.5A to 4.2V, 2A discharge to 2.7V cutoff).
- **B0006**: Cycled at 24°C with standard CC-CV discharge (faster fade rate).
- **B0007**: Cycled at 24°C with accelerated discharge conditions.
- **B0018**: Cycled at 24°C with higher cutoff voltage (2.5V).

### MATLAB `.mat` File Hierarchy
When downloaded from NASA, files are named `B0005.mat`, `B0006.mat`, etc.
SciPy (`scipy.io.loadmat`) dynamically inspects the file as follows:
```
B0005.mat
└── B0005 (struct)
    └── cycle (1D array of structs)
        ├── type: 'charge' | 'discharge' | 'impedance'
        ├── ambient_temperature: 24
        ├── time: [year, month, day, hour, minute, second]
        └── data (struct)
            ├── Voltage_measured: [array of discharge voltage samples]
            ├── Current_measured: [array of discharge current samples]
            ├── Temperature_measured: [array of thermocouple readings (°C)]
            ├── Current_load: [load current]
            ├── Voltage_load: [load voltage]
            ├── Time: [elapsed seconds]
            └── Capacity: [scalar float usable discharge capacity (Ah)]
```
EVGuard's `src/evguard/data/loader.py` automatically detects and filters the `discharge` cycles, extracting the usable capacity and computing aggregated cycle statistics (`avg_voltage_v`, `min_voltage_v`, `avg_temperature_c`, `max_temperature_c`).

### How to Use Real NASA Data
1. Download `B0005.mat` (or `B0006.mat`, `B0018.mat`) from NASA's open data repository.
2. Place the file directly inside `data/raw/` (e.g. `data/raw/B0005.mat`).
3. Run the pipeline:
   ```bash
   .\.venv\Scripts\python.exe run.py --pipeline
   ```
4. EVGuard automatically discovers the file in `data/raw/`, parses the cycles, and updates the dashboard.

---

## 3. Synthetic Demo Dataset

The demo dataset (`data/demo/battery_aging_demo.csv`) is generated deterministically (`seed=42`) using empirical battery degradation models:

### Simulated Battery Profiles
| Battery Identifier | Initial Capacity | Thermal Baseline | Degradation Characteristics |
|---|---|---|---|
| **BAT_SIM_01** | 2.02 Ah | 27.5°C | Normal nominal aging, moderate SEI growth |
| **BAT_SIM_02** | 1.98 Ah | 38.0°C | High thermal stress, accelerated capacity fade |
| **BAT_SIM_03** | 2.05 Ah | 30.0°C | Fast-discharge wear with rest-period capacity recovery bursts |
| **BAT_SIM_04** | 1.92 Ah | 33.5°C | Accelerated aging cell crossing 80% EOL early |

### Physical Phenomena Included
1. **Capacity Self-Recovery**: Periodic capacity rebound bursts after rest intervals (a well-documented electrochemical relaxation phenomenon in lithium cells).
2. **Impedance Rise**: Internal resistance increases as the electrolyte decomposes and electrodes degrade.
3. **Controlled Irregularities**:
   - Implausible temperature glitch (98.5°C) at cycle 77.
   - Implausible low voltage reading (1.15V) at cycle 115.
   - 1 duplicate measurement record.
   *These test and prove the effectiveness of the transparent validation and auditing layer.*

### Dataset Schema
| Column Name | Data Type | Description |
|---|---|---|
| `battery_id` | String | Unique cell identifier (e.g. `BAT_SIM_01`) |
| `cycle_number` | Integer | Sequential discharge cycle count |
| `timestamp` | ISO 8601 String | Simulation telemetry timestamp |
| `discharge_capacity_ah` | Float | Measured usable discharge capacity in Ah |
| `soh_pct` | Float | Capacity-based State of Health percentage |
| `avg_voltage_v` | Float | Mean cell voltage during discharge (V) |
| `min_voltage_v` | Float | Cutoff discharge voltage (V) |
| `max_voltage_v` | Float | Pre-discharge resting voltage (V) |
| `avg_temperature_c` | Float | Average cell temperature during cycle (°C) |
| `max_temperature_c` | Float | Peak cell temperature during discharge (°C) |
| `internal_resistance_ohm` | Float | Estimated internal cell impedance (Ω) |
| `discharge_duration_s` | Integer | Total duration of the discharge cycle in seconds |
| `data_source` | String | Data lineage marker (`SIMULATED_DEMO` or `NASA_MATLAB_MAT`) |
| `stress_condition` | String | Environmental condition label (`normal`, `thermal`, etc.) |
