"""PySpark session factory tailored for Windows 64-bit dual-core hardware."""

import os
import sys
from pathlib import Path
from typing import Optional
from pyspark.sql import SparkSession

from evguard.utils.config import (
    PREFERRED_JDK_17,
    SPARK_DRIVER_BIND_ADDRESS,
    SPARK_DRIVER_HOST,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
)
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


def setup_java_environment() -> None:
    """Ensure JAVA_HOME points to a compatible Java 17 JDK and is present in PATH."""
    if PREFERRED_JDK_17.exists():
        os.environ["JAVA_HOME"] = str(PREFERRED_JDK_17)
        bin_dir = str(PREFERRED_JDK_17 / "bin")
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
        logger.debug(f"Using Java 17 at {PREFERRED_JDK_17}")
    elif "JAVA_HOME" in os.environ:
        logger.debug(f"Using existing JAVA_HOME: {os.environ['JAVA_HOME']}")
    else:
        logger.warning("JAVA_HOME is not explicitly set; relying on system PATH.")

    # Configure Python worker executable explicitly for Windows compatibility
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_session(app_name: str = "EVGuard_Analytics") -> SparkSession:
    """Initialize and return a hardware-tuned local PySpark session.

    Optimized for:
    - Dual-core Intel Core i3 (local[2])
    - 2GB driver memory limit
    - Local IPv4 binding (127.0.0.1) for Windows loopback reliability
    - RawLocalFileSystem to prevent Windows winutils permission warnings
    """
    setup_java_environment()

    logger.info(
        f"Initializing SparkSession '{app_name}' with master={SPARK_MASTER}, "
        f"driver_memory={SPARK_DRIVER_MEMORY}, shuffle_partitions={SPARK_SHUFFLE_PARTITIONS}"
    )

    builder = (
        SparkSession.builder.master(SPARK_MASTER)
        .appName(app_name)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .config("spark.driver.bindAddress", SPARK_DRIVER_BIND_ADDRESS)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    )

    spark = builder.getOrCreate()
    # Set Spark log level to WARN to keep logs clean and readable
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession successfully acquired.")
    return spark


def stop_spark_session(spark: Optional[SparkSession]) -> None:
    """Safely shut down an active Spark session."""
    if spark is not None:
        try:
            logger.info("Stopping SparkSession...")
            spark.stop()
            logger.info("SparkSession stopped cleanly.")
        except Exception as e:
            logger.warning(f"Error while stopping SparkSession: {e}")
