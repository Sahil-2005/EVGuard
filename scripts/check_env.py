"""Environment & PySpark prerequisite check for EVGuard."""

import os
import sys
from pathlib import Path


def configure_java():
    """Ensure JAVA_HOME points to a compatible JDK (preferably JDK 17)."""
    # Check if current JAVA_HOME is set
    current_java = os.environ.get("JAVA_HOME", "")
    preferred_jdk_17 = Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")

    if preferred_jdk_17.exists():
        os.environ["JAVA_HOME"] = str(preferred_jdk_17)
        bin_dir = str(preferred_jdk_17 / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        print(f"[OK] Using preferred Java 17 at: {preferred_jdk_17}")
    elif current_java:
        print(f"[INFO] Using existing JAVA_HOME: {current_java}")
    else:
        print("[WARN] JAVA_HOME is not explicitly set; relying on system PATH.")


def main():
    print("=" * 60)
    print("EVGuard Environment Verification")
    print("=" * 60)

    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")

    configure_java()

    try:
        import pyspark

        print(f"PySpark Version   : {pyspark.__version__}")
    except ImportError as e:
        print(f"[FAIL] PySpark is not installed: {e}")
        return 1

    try:
        import numpy as np
        import pandas as pd
        import plotly
        import scipy
        import sklearn
        import streamlit

        print(f"Pandas Version    : {pd.__version__}")
        print(f"NumPy Version     : {np.__version__}")
        print(f"SciPy Version     : {scipy.__version__}")
        print(f"Scikit-learn      : {sklearn.__version__}")
        print(f"Streamlit Version : {streamlit.__version__}")
        print(f"Plotly Version    : {plotly.__version__}")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        return 1

    # Set Python executable for PySpark worker and driver
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    print("\nAttempting SparkSession initialization (local[2], 2GB driver memory)...")
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[2]")
            .appName("EVGuard_EnvCheck")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

        test_df = spark.createDataFrame([(1, "Battery_A", 2.0), (2, "Battery_B", 1.9)], ["id", "battery_id", "capacity"])
        row_count = test_df.count()
        test_df.createOrReplaceTempView("test_batt")
        sql_res = spark.sql("SELECT avg(capacity) as avg_cap FROM test_batt").collect()
        avg_cap = sql_res[0]["avg_cap"]

        print("[OK] Spark initialized successfully!")
        print(f"[OK] Test DataFrame row count: {row_count}")
        print(f"[OK] Spark SQL average capacity: {avg_cap:.2f}")

        spark.stop()
        print("[OK] SparkSession stopped cleanly.")
        print("\nAll EVGuard environment prerequisites PASSED successfully!")
        return 0

    except Exception as e:
        print(f"[FAIL] SparkSession initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
