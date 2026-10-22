"""CLI Pipeline runner for EVGuard Big Data processing and analytics."""

import argparse
import sys
from pathlib import Path

# Add project root and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evguard.analytics.degradation import compute_fleet_degradation_rankings
from evguard.bigdata.pipeline import run_spark_pipeline
from evguard.bigdata.spark_session import get_spark_session, stop_spark_session
from evguard.data.loader import load_dataset
from evguard.data.synthetic_generator import generate_synthetic_battery_data
from evguard.data.validator import validate_and_preprocess
from evguard.models.clustering import train_and_evaluate_clustering
from evguard.utils.config import DEMO_DATA_FILE
from evguard.utils.logger import get_logger

logger = get_logger("evguard.pipeline")


def main():
    parser = argparse.ArgumentParser(description="EVGuard Big Data Battery Health Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to battery dataset (CSV or .mat). Defaults to demo.")
    parser.add_argument("--force-demo", action="store_true", help="Force generation and use of synthetic demo data.")
    args = parser.parse_args()

    print("=" * 70)
    print(" EVGuard: Electric Vehicle Battery Health Analysis Pipeline ")
    print(" Big Data Processing with Apache PySpark, Spark SQL & Machine Learning ")
    print("=" * 70)

    # 1. Dataset Resolution & Loading
    target_path = Path(args.data_path) if args.data_path else None
    if args.force_demo or (target_path is None and not DEMO_DATA_FILE.exists()):
        print("\n[Step 1/5] Generating deterministic synthetic demo battery dataset...")
        raw_df = generate_synthetic_battery_data(DEMO_DATA_FILE)
    else:
        print("\n[Step 1/5] Loading battery dataset...")
        raw_df = load_dataset(target_path)

    print(f"-> Loaded {len(raw_df)} total records from source.")

    # 2. Validation & Anomaly Flagging
    print("\n[Step 2/5] Validating data schema, checking duplicates & flagging anomalies...")
    validated_df, quality_report = validate_and_preprocess(raw_df)
    print(f"-> Valid records: {quality_report['plausible_records_count']}")
    print(f"-> Flagged anomalies: {quality_report['anomalous_records_count']}")
    print(f"-> Duplicates resolved: {quality_report['duplicate_records_removed']}")
    print(f"-> Batteries in fleet: {quality_report['batteries_analyzed']}")

    # 3. Big Data PySpark ETL & Spark SQL Transformations
    print("\n[Step 3/5] Starting Apache PySpark distributed processing & Spark SQL queries...")
    spark = get_spark_session(app_name="EVGuard_CLI_Pipeline")
    try:
        processed_df, spark_stats = run_spark_pipeline(validated_df, spark=spark)
        print(f"-> Spark Master: {spark_stats['spark_master']}")
        print(f"-> Partitions processed: {spark_stats['partitions']}")
        print(f"-> Execution time: {spark_stats['execution_time_seconds']}s")
        print(f"-> Saved cleaned Parquet to: {spark_stats['parquet_destination']}")
    finally:
        stop_spark_session(spark)

    # 4. SOH Degradation Analytics & RUL Estimation
    print("\n[Step 4/5] Computing capacity degradation kinetics & Remaining Useful Life (RUL)...")
    fleet_rankings = compute_fleet_degradation_rankings(processed_df)
    print("\nBattery Fleet Health Rankings (Sorted by Current SOH):")
    print("-" * 70)
    for _, row in fleet_rankings.iterrows():
        rul_text = f"{row['remaining_useful_life_cycles']} cycles" if row["remaining_useful_life_cycles"] is not None else "Reached EOL"
        print(
            f" Battery: {row['battery_id']:<12} | Current SOH: {row['current_soh_pct']:>6.2f}% | "
            f"Fade Rate: {row['fade_rate_pct_per_100_cycles']:>5.2f}%/100c | Est. RUL: {rul_text}"
        )
    print("-" * 70)

    # 5. K-Means Clustering & Model Artifacts
    print("\n[Step 5/5] Training K-Means clustering model on degradation profiles...")
    model, clustered_df, metrics = train_and_evaluate_clustering(processed_df)
    print(f"-> Silhouette Score: {metrics['silhouette_score']}")
    print(f"-> Davies-Bouldin Index: {metrics['davies_bouldin_index']}")
    print("-> Model artifact saved to: artifacts/kmeans_model.joblib")
    print("-> Metrics report saved to: artifacts/clustering_metrics.json")

    print("\n" + "=" * 70)
    print(" EVGuard Big Data Pipeline Completed Successfully! ")
    print(" Ready for Streamlit Dashboard visualization: python run.py --dashboard ")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
