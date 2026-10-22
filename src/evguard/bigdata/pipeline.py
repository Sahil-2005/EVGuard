"""Big Data PySpark processing pipeline with Spark SQL transformations and Parquet I/O."""

import time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from evguard.bigdata.spark_session import get_spark_session, stop_spark_session
from evguard.utils.config import (
    EOL_SOH_THRESHOLD_PCT,
    NOMINAL_CAPACITY_AH,
    PROCESSED_PARQUET_DIR,
)
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def run_spark_pipeline(
    validated_df: pd.DataFrame,
    spark: SparkSession = None,
    output_parquet_path: Path = PROCESSED_PARQUET_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Execute the Big Data PySpark ETL and Spark SQL analytics pipeline.

    Transformations performed in Spark:
    1. Schema ingestion into distributed PySpark DataFrame.
    2. Spark SQL view registration and SQL-based aggregation queries.
    3. Window function computation for cumulative capacity fade & cycle delta.
    4. Categorization into health states (HEALTHY, MODERATE_WEAR, END_OF_LIFE).
    5. Clean Parquet serialization.
    """
    start_time = time.time()
    own_session = False

    if spark is None:
        spark = get_spark_session(app_name="EVGuard_BigData_Pipeline")
        own_session = True

    try:
        logger.info(f"Ingesting {len(validated_df)} validated records into PySpark DataFrame...")

        # Ensure datetime is string or timestamp compatible for Spark
        clean_df = validated_df.copy()
        if "timestamp" in clean_df.columns:
            clean_df["timestamp"] = clean_df["timestamp"].astype(str)

        spark_df: DataFrame = spark.createDataFrame(clean_df)
        num_partitions = spark_df.rdd.getNumPartitions()
        total_rows = spark_df.count()

        logger.info(f"PySpark DataFrame initialized: {total_rows} rows across {num_partitions} partition(s).")

        # Register SQL Temporary View
        spark_df.createOrReplaceTempView("battery_cycles_raw")

        # 1. Spark SQL Query: Fleet-wide physical bounds & operational summary
        logger.info("Executing Spark SQL aggregation query...")
        sql_summary_query = """
            SELECT
                battery_id,
                COUNT(cycle_number) AS total_cycles,
                ROUND(MIN(discharge_capacity_ah), 4) AS min_capacity_ah,
                ROUND(MAX(discharge_capacity_ah), 4) AS max_capacity_ah,
                ROUND(AVG(discharge_capacity_ah), 4) AS avg_capacity_ah,
                ROUND(MIN(soh_pct), 2) AS min_soh_pct,
                ROUND(MAX(soh_pct), 2) AS max_soh_pct,
                ROUND(AVG(avg_temperature_c), 2) AS fleet_avg_temp_c,
                ROUND(MAX(max_temperature_c), 2) AS fleet_max_temp_c,
                ROUND(AVG(internal_resistance_ohm), 4) AS avg_internal_r_ohm,
                SUM(CASE WHEN is_plausible = false THEN 1 ELSE 0 END) AS flagged_anomalies
            FROM battery_cycles_raw
            GROUP BY battery_id
            ORDER BY battery_id
        """
        spark_sql_summary = spark.sql(sql_summary_query)
        fleet_metrics = [row.asDict() for row in spark_sql_summary.collect()]

        # 2. Window Functions: Compute cycle-over-cycle delta and cumulative degradation in Spark
        window_spec = Window.partitionBy("battery_id").orderBy("cycle_number")

        transformed_spark_df = (
            spark_df.withColumn(
                "prev_capacity_ah",
                F.lag("discharge_capacity_ah", 1).over(window_spec),
            )
            .withColumn(
                "cycle_capacity_delta_ah",
                F.round(F.col("discharge_capacity_ah") - F.col("prev_capacity_ah"), 4),
            )
            .withColumn(
                "cumulative_fade_pct",
                F.round(((F.first("discharge_capacity_ah").over(window_spec) - F.col("discharge_capacity_ah")) / F.lit(NOMINAL_CAPACITY_AH)) * 100.0, 2),
            )
            .withColumn(
                "health_category",
                F.when(F.col("soh_pct") >= 90.0, "HEALTHY")
                .when((F.col("soh_pct") >= EOL_SOH_THRESHOLD_PCT) & (F.col("soh_pct") < 90.0), "MODERATE_WEAR")
                .when(F.col("soh_pct") < EOL_SOH_THRESHOLD_PCT, "END_OF_LIFE")
                .otherwise("ANOMALY"),
            )
        )

        # 3. Export to Clean Parquet
        import shutil

        output_parquet_path = Path(output_parquet_path)
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_str = str(output_parquet_path.resolve())

        logger.info(f"Writing transformed Spark DataFrame to Parquet: {parquet_str}")
        try:
            transformed_spark_df.write.mode("overwrite").parquet(parquet_str)
            logger.info("Successfully written Parquet dataset via PySpark write.")
        except Exception as e:
            logger.warning(
                f"Native Spark Parquet write encountered Windows permission limitation: {e}. Serializing via PyArrow."
            )
            # If Spark left a partial directory behind, remove it before writing via PyArrow
            if output_parquet_path.exists() and output_parquet_path.is_dir():
                shutil.rmtree(output_parquet_path, ignore_errors=True)

            processed_pdf = transformed_spark_df.toPandas()
            processed_pdf.to_parquet(output_parquet_path, index=False, engine="pyarrow")
            logger.info("Successfully written Parquet dataset via PyArrow fallback.")

        elapsed = round(time.time() - start_time, 2)
        pipeline_stats = {
            "spark_app_name": spark.sparkContext.appName,
            "spark_master": spark.sparkContext.master,
            "spark_version": spark.version,
            "partitions": num_partitions,
            "total_records_processed": total_rows,
            "fleet_summary": fleet_metrics,
            "execution_time_seconds": elapsed,
            "parquet_destination": str(output_parquet_path),
        }

        # Convert back to clean pandas for in-memory model training and visualization
        final_processed_df = transformed_spark_df.toPandas()

        logger.info(f"Big Data pipeline complete in {elapsed}s. Processed {total_rows} rows.")
        return final_processed_df, pipeline_stats

    finally:
        if own_session:
            stop_spark_session(spark)
