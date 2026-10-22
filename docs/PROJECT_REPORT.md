# College Mini-Project Report

## EVGuard: Electric Vehicle Battery Health Analysis Using Big Data Techniques

---

### Academic Information
- **Course**: Big Data Analytics (BDA) — 7th Semester
- **Degree**: Bachelor of Engineering / Bachelor of Technology in Computer Science & Engineering
- **Project Type**: Big Data Analytics & Machine Learning Mini-Project
- **Academic Year**: 2025 – 2026

---

### Abstract
Electric Vehicles (EVs) rely heavily on Lithium-ion (Li-ion) battery packs as their primary energy storage mechanism. Over prolonged operation, repeated charge-discharge cycling induces electrochemical degradation, resulting in irreversible capacity loss, increased internal impedance, and elevated thermal risk. Precise estimation of battery **State of Health (SOH)** and Remaining Useful Life (RUL) is critical for operational safety, warranty estimation, and battery second-life repurposing.

This project introduces **EVGuard**, an end-to-end Big Data analytical platform engineered to process high-throughput battery telemetry. Using **Apache PySpark** and **Spark SQL**, the system executes distributed data validation, transparent sensor anomaly flagging, and cycle-over-cycle window aggregations. An unsupervised **K-Means machine learning** model partitions battery degradation profiles into distinct operational aging regimes based on capacity fade, impedance rise, and thermal characteristics. The complete system is verified on commodity dual-core consumer hardware, providing an interactive **Streamlit** and **Plotly** analytical dashboard for fleet-level battery monitoring.

---

### 1. Introduction
With the worldwide transition toward sustainable transportation, Electric Vehicles represent the cornerstone of future mobility. The traction battery pack accounts for approximately 30% to 40% of the total vehicle cost. Consequently, understanding cell degradation dynamics and implementing automated health prognosis algorithms is an active area of research in Big Data and automotive informatics.

Large fleets of EVs generate substantial time-series data including voltage, current, temperature, and cycle counts. Traditional single-node spreadsheet or monolithic database solutions cannot efficiently process, clean, and analyze multi-cell telemetry across hundreds of thousands of cycles. Utilizing Big Data frameworks such as Apache Spark enables scalable ETL, distributed window computations, and standardized data lake serialization.

---

### 2. Problem Statement & Objectives

#### Problem Statement
Current battery management systems (BMS) often rely on coarse rule-based indicators or black-box predictions that either mask physical measurement anomalies or fail to distinguish calculated capacity ratios from statistical models. Furthermore, existing research implementations frequently require heavy cloud clusters or specialized GPU setups, making local deployment and reproducible academic experimentation challenging.

#### Core Objectives
1. **Big Data Processing**: Build a robust, hardware-optimized Apache PySpark pipeline using local CPU parallelization (`local[2]`, 2 GB driver memory).
2. **Standardized SOH Calculation**: Implement transparent capacity-based State of Health:
   $$\text{SOH (\%)} = \frac{\text{Current Usable Capacity (Ah)}}{\text{Nominal Reference Capacity (2.0 Ah)}} \times 100$$
3. **Data Quality & Anomaly Auditing**: Implement transparent bounds checking and deduplication to flag implausible sensor telemetry without silent deletion.
4. **Machine Learning Categorization**: Apply K-Means clustering with feature standardization to classify cycle aging regimes, validating cluster separation with Silhouette scores and Davies-Bouldin indices.
5. **Interactive Visualization**: Design a responsive Streamlit dashboard with Plotly charts presenting SOH trends, degradation curves, 3D cluster projections, and Spark execution statistics.

---

### 3. Literature Survey & Theoretical Framework

#### Capacity-Based State of Health (SOH)
State of Health is defined as the ratio of a battery's maximum usable discharge capacity at its current aging cycle to its rated nominal capacity:
- **Nominal Reference**: For commercial 18650 Li-ion cells (e.g. NASA Ames PCoE benchmarks), the standard rated capacity is $C_{\text{nominal}} = 2.0\text{ Ah}$.
- **End-of-Life (EOL) Threshold**: The international automotive benchmark defines the retirement boundary at $\text{SOH} = 80\%$. Cells reaching this threshold are decommissioned from vehicular duty to prevent thermal runaway or excessive range reduction, often transitioning to stationary grid storage.

#### Electrochemical Aging Phenomena
- **Solid Electrolyte Interphase (SEI) Layer Growth**: Continuous decomposition of electrolyte consumes active lithium ions, causing gradual initial capacity decline.
- **Internal Resistance Rise**: As electrodes degrade and contact resistance increases, internal impedance ($R_{\text{int}}$) rises from $\sim 0.06\,\Omega$ to $>0.25\,\Omega$, generating higher Joule heating during discharge.
- **Capacity Recovery Effect**: When lithium cells rest between cycles, internal chemical equilibrium partially re-establishes, producing temporary capacity rebound bursts.

---

### 4. Big Data Methodology & PySpark Architecture

#### Ingestion & Schema Enforcement
Telemetry is ingested into distributed PySpark DataFrames with explicit typing:
```
battery_cycles_raw (
    battery_id: String,
    cycle_number: Integer,
    discharge_capacity_ah: Double,
    soh_pct: Double,
    avg_voltage_v: Double,
    min_voltage_v: Double,
    max_voltage_v: Double,
    avg_temperature_c: Double,
    max_temperature_c: Double,
    internal_resistance_ohm: Double,
    is_plausible: Boolean,
    anomaly_reason: String
)
```

#### Spark SQL & Window Functions
PySpark executes multi-dimensional transformations without pulling full partitions to the driver:
1. **Fleet Aggregation Query**: Computes per-battery lifetime summaries (min/max/average capacity, peak temperature, anomaly counts).
2. **Partition Windowing**:
   ```python
   window_spec = Window.partitionBy("battery_id").orderBy("cycle_number")
   transformed_df = spark_df.withColumn(
       "cycle_capacity_delta_ah",
       F.round(F.col("discharge_capacity_ah") - F.lag("discharge_capacity_ah", 1).over(window_spec), 4)
   )
   ```
3. **Parquet Columnar Storage**: Processed records are serialized into compressed Apache Parquet format (`data/processed/battery_cycles.parquet`).

---

### 5. Machine Learning: K-Means Clustering

#### Feature Space & Preprocessing
To eliminate scale bias across differing physical units (capacities in Ah, voltages in V, temperatures in °C, resistances in $\Omega$), all features are normalized using `StandardScaler`:
$$z = \frac{x - \mu}{\sigma}$$

Features used:
- `discharge_capacity_ah`
- `soh_pct`
- `avg_temperature_c`
- `max_temperature_c`
- `avg_voltage_v`
- `internal_resistance_ohm`

#### Model Formulation & Validation
- **Algorithm**: K-Means with $k=3$ clusters, `random_state=42`, and 10 random restarts (`n_init=10`).
- **Elbow Method**: WCSS (Inertia) evaluated from $k=2$ to $k=6$ confirms an elbow inflection point at $k=3$.
- **Validation Metrics**:
  - Silhouette Coefficient: $0.3408$ (indicates structured cluster segregation).
  - Davies-Bouldin Index: $1.0873$ (confirms compact cluster centroids).

#### Operational Regime Interpretation
- **Cluster 0 (Tier 1)**: Nominal Health & Low Impedance ($SOH \ge 90\%$, $R_{\text{int}} \approx 0.08\,\Omega$, normal temperatures).
- **Cluster 1 (Tier 2)**: Moderate Thermal / Wear Aging ($80\% \le SOH < 90\%$, slight impedance increase).
- **Cluster 2 (Tier 3)**: Degraded Capacity / Near End-of-Life ($SOH < 80\%$, $R_{\text{int}} > 0.22\,\Omega$, elevated thermal response).

---

### 6. Experimental Results & Analysis

| Battery Identifier | Total Cycles | Initial Capacity | Final Capacity | Final SOH (%) | Fade Rate (/100c) | Estimated RUL | Health Stage |
|---|---|---|---|---|---|---|---|
| **BAT_SIM_01** | 150 | 2.020 Ah | 1.616 Ah | 80.78% | 13.52% | 10 cycles | Moderate Wear |
| **BAT_SIM_02** | 150 | 1.980 Ah | 1.255 Ah | 62.75% | 24.13% | 0 cycles | End-of-Life |
| **BAT_SIM_03** | 150 | 2.050 Ah | 1.602 Ah | 80.12% | 14.86% | 22 cycles | Moderate Wear |
| **BAT_SIM_04** | 150 | 1.920 Ah | 1.053 Ah | 52.66% | 28.94% | 0 cycles | End-of-Life |

**Key Observations**:
1. High operational temperatures (BAT_SIM_02 at ~38°C) nearly doubled the capacity fade rate (24.13%/100c vs 13.52%/100c).
2. The linear regression RUL estimator accurately flagged that BAT_SIM_04 and BAT_SIM_02 had already exceeded their 80% EOL boundary.
3. The data validation subsystem successfully captured and isolated sensor glitches (implausible temperatures and voltages) without interrupting execution.

---

### 7. Hardware Adaptation & Performance
- **Target Machine**: Intel Core i3-1005G1 CPU @ 1.20GHz (2 cores, 4 logical processors), 12 GB RAM, Windows 64-bit.
- **Spark Tuning**: Setting `local[2]` CPU threads and a 2.0 GB memory ceiling maintained driver memory utilization below 25% of available physical RAM.
- **Throughput**: 600 multi-battery cycles processed, aggregated, and stored to Parquet in approximately **10.2 seconds**.

---

### 8. Conclusion & Future Scope
EVGuard demonstrates that Big Data distributed analytics (Apache PySpark) coupled with unsupervised machine learning can be effectively deployed on standard consumer laptop hardware to provide robust, college-level battery health analytics. The framework successfully decouples ground-truth capacity SOH calculations from statistical models and provides transparent data auditing.

#### Future Enhancements
1. Integrating electro-thermal physics models (Thevenin equivalent circuit modeling).
2. Implementing real-time streaming ingestion via PySpark Structured Streaming when connected to hardware OBD-II telemetry loggers.
3. Incorporating cell balance anomaly detection across multi-cell series strings.

---

### 9. References
1. NASA Ames Prognostics Center of Excellence, *Li-ion Battery Aging Datasets*, NASA Open Data Portal.
2. Apache Spark Documentation, *Spark SQL, DataFrames and Datasets Guide*, Apache Software Foundation.
3. M. A. Hannan et al., "State of Health Estimation of Lithium-Ion Batteries: Review and Best Practices," *IEEE Transactions on Industrial Informatics*, 2018.
4. scikit-learn documentation, *Clustering with K-Means and StandardScaler*, 2024.
