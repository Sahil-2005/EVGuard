# EVGuard: Electric Vehicle Battery Health Analysis Using Big Data Techniques

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Apache PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24.1-purple.svg)](https://plotly.com/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)](https://pytest.org/)

**EVGuard** is a Big Data and machine learning framework designed for Electric Vehicle (EV) battery health diagnostics and degradation prognosis. Built directly on **Apache PySpark**, **Spark SQL**, **Scikit-Learn**, and **Streamlit**, it processes electrochemical telemetry, evaluates capacity-based **State of Health (SOH)**, predicts Remaining Useful Life (RUL), and classifies multi-dimensional aging regimes using **K-Means clustering**.

Optimized specifically for commodity dual-core consumer hardware (Intel Core i3, 12 GB RAM, Windows 64-bit). No GPU, Docker, external database, or paid APIs are required.

---

## ⚡ Key Features

1. **Big Data Processing (Apache PySpark & Spark SQL)**
   - Hardware-tuned local execution: `master="local[2]"`, `driver_memory="2g"`, `shuffle_partitions="2"`.
   - Spark SQL queries and window functions (`F.lag`, `F.first`) to compute cycle-over-cycle degradation deltas and cumulative capacity fade.
   - Clean Apache Parquet columnar serialization with Snappy compression.
2. **Standardized State of Health (SOH) Analysis**
   - Capacity-based formulation: $\text{SOH (\%)} = (\text{Usable Capacity} / \text{Nominal Capacity}) \times 100$.
   - Nominal 2.0 Ah 18650 cell benchmark; handles invalid/negative capacities safely.
   - Distinguishes ground-truth calculated SOH from statistical predictions.
3. **Kinetics & Remaining Useful Life (RUL)**
   - Linear regression capacity fade rate (% loss per 100 cycles).
   - Dynamic cycle estimation until the standard 80% End-of-Life (EOL) retirement boundary.
4. **Machine Learning (K-Means Clustering)**
   - Unsupervised clustering on normalized feature space (`StandardScaler`, `random_state=42`).
   - Validated via Silhouette Score, Davies-Bouldin Index, and multi-$k$ Elbow curves.
   - Grounded engineering regime labels (Tier 1 Nominal, Tier 2 Moderate Wear, Tier 3 Degraded EOL).
5. **Interactive Streamlit & Plotly Dashboard**
   - 4 Dedicated modules:
     1. *Battery Overview & SOH* (KPI metrics, health status gauge, discharge telemetry table).
     2. *Degradation Trends & RUL* (SOH vs cycles, 80% EOL boundary, impedance/thermal scatter).
     3. *K-Means Machine Learning* (3D interactive scatter, cluster profiles, elbow curve).
     4. *Spark Big Data & Quality* (Spark config, Spark SQL results, anomaly audit log).
6. **Transparent Data Quality & Auditing**
   - Detects schema mismatches, resolves duplicates transparently, and flags physical sensor anomalies (e.g., implausible voltage or temperature readings) without silent record deletion.

---

## 💻 Hardware Requirements & Optimization

Tested and tuned for:
- **Operating System**: Windows 10/11 64-bit
- **CPU**: Intel Core i3-1005G1 (2 Cores, 4 Threads) or compatible CPU
- **RAM**: 8 GB to 12 GB
- **Java**: Java JDK 17 (Eclipse Adoptium Temurin-17 recommended)
- **Python**: Python 3.11 (64-bit)

---

## 📁 Project Directory Structure

```
EVGuard/
├── .venv/                      # Isolated Python 3.11 virtual environment
├── app/
│   └── app.py                  # Streamlit interactive multi-section dashboard
├── src/
│   └── evguard/
│       ├── __init__.py
│       ├── analytics/          # SOH calculation, classification, degradation kinetics & RUL
│       │   ├── degradation.py
│       │   └── soh.py
│       ├── bigdata/            # PySpark session factory & Spark SQL processing pipeline
│       │   ├── pipeline.py
│       │   └── spark_session.py
│       ├── data/               # Loaders, synthetic demo generator, schema validation
│       │   ├── loader.py
│       │   ├── synthetic_generator.py
│       │   └── validator.py
│       ├── models/             # K-Means clustering, scaling, diagnostics, and evaluation
│       │   ├── clustering.py
│       │   └── evaluator.py
│       └── utils/              # Project paths, physical bounds config, and logging
│           ├── config.py
│           └── logger.py
├── data/
│   ├── demo/                   # Deterministic synthetic demo dataset (battery_aging_demo.csv)
│   ├── processed/              # Cleaned Apache Parquet data (battery_cycles.parquet)
│   └── raw/                    # Destination for NASA .mat or .csv files
├── artifacts/                  # Serialized KMeans model (.joblib) and metrics (.json)
├── reports/                    # Generated data quality summary reports
├── logs/                       # Application logs
├── tests/                      # 22 Automated pytest unit & integration tests
├── scripts/
│   ├── check_env.py            # Environment & Spark session verification script
│   └── run_pipeline.py         # End-to-end Big Data CLI pipeline runner
├── docs/
│   ├── ARCHITECTURE.md         # System architecture and Mermaid dataflow diagram
│   ├── DATASET_GUIDE.md        # NASA dataset instructions and schema documentation
│   └── PROJECT_REPORT.md       # Full college mini-project report template
├── .gitignore
├── .python-version
├── pyproject.toml              # Ruff and Pytest configuration
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project documentation
└── run.py                      # Root command center entrypoint
```

---

## 🚀 Quickstart & Usage

### 1. Environment Verification
Verify that Python 3.11, Java 17, and PySpark are properly recognized:
```powershell
.\.venv\Scripts\python.exe run.py --check-env
```

### 2. Run the Big Data Pipeline
Execute the end-to-end pipeline (synthetic demo generation, data validation, PySpark Spark SQL processing, SOH degradation kinetics, and K-Means training):
```powershell
.\.venv\Scripts\python.exe run.py --pipeline
```

To force-rebuild the synthetic demo dataset:
```powershell
.\.venv\Scripts\python.exe run.py --pipeline --force-demo
```

### 3. Launch the Interactive Dashboard
Start the Streamlit dashboard on `http://localhost:8501`:
```powershell
.\.venv\Scripts\python.exe run.py --dashboard
```
Or simply run:
```powershell
.\.venv\Scripts\python.exe run.py
```
*(If the pipeline has not been executed yet, `run.py` will automatically run it before starting the dashboard).*

### 4. Run Automated Tests
Execute the comprehensive test suite (22 tests covering SOH formulas, boundaries, validation, Spark SQL, and clustering):
```powershell
.\.venv\Scripts\python.exe run.py --test
```

### 5. Run Code Quality Linter
Verify code style and formatting with Ruff:
```powershell
.\.venv\Scripts\python.exe run.py --lint
```

---

## 📊 Using the NASA Battery Aging Dataset

EVGuard natively supports the official **NASA Ames Prognostics Center of Excellence Li-ion Battery Aging Dataset**:
1. Download `B0005.mat`, `B0006.mat`, or `B0018.mat` from the [NASA Data Portal](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).
2. Place the downloaded `.mat` file in `data/raw/` (e.g. `data/raw/B0005.mat`).
3. Run the pipeline:
   ```powershell
   .\.venv\Scripts\python.exe run.py --pipeline
   ```
4. EVGuard's SciPy-based loader will dynamically inspect the struct, extract discharge cycles, and update all analytics.
*(Refer to [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) for complete details).*

---

## 🛠️ Troubleshooting & Technical Notes

### Java JDK 17 Configuration
- Apache PySpark 3.5 requires Java 8, 11, or 17. Java 21 can produce reflection errors on Windows.
- EVGuard automatically detects and points `JAVA_HOME` to `C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot` if present, ensuring complete compatibility.

### Windows PySpark Worker Binding
- On Windows, PySpark may default to the Windows Store Python stub if environment variables are unset.
- EVGuard explicitly registers `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to the active `.venv\Scripts\python.exe` and binds the driver to `127.0.0.1` to prevent socket timeout issues.

### Parquet Serialization on Windows
- If Hadoop `winutils.exe` is absent on Windows, native Hadoop file permission calls can throw warnings.
- EVGuard configures `org.apache.hadoop.fs.RawLocalFileSystem` and includes an automatic PyArrow serialization fallback to guarantee reliable Parquet output.

---

## 📜 License & Academic Integrity
Developed as a Big Data Analytics (BDA) College Mini-Project. Code is open for educational and research purposes.
