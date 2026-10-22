"""Unit tests for PySpark session management and Spark SQL operations."""

import pytest

from evguard.bigdata.spark_session import get_spark_session, stop_spark_session


@pytest.fixture(scope="module")
def spark_session():
    """Shared SparkSession fixture for testing."""
    spark = get_spark_session(app_name="EVGuard_Pytest")
    yield spark
    stop_spark_session(spark)


class TestSparkProcessing:
    """Test suite for Big Data PySpark DataFrame operations and Spark SQL."""

    def test_spark_session_master_and_memory(self, spark_session):
        """Test Spark is configured with local[2] master."""
        assert "local[2]" in spark_session.sparkContext.master
        assert spark_session.version.startswith("3.")

    def test_spark_sql_aggregation(self, spark_session):
        """Test Spark SQL query execution on sample DataFrame."""
        data = [
            ("B01", 1, 2.02, 101.0),
            ("B01", 2, 1.98, 99.0),
            ("B02", 1, 1.95, 97.5),
            ("B02", 2, 1.91, 95.5),
        ]
        df = spark_session.createDataFrame(data, ["battery_id", "cycle_number", "capacity", "soh"])
        df.createOrReplaceTempView("test_cycles")

        res = spark_session.sql("""
            SELECT
                battery_id,
                COUNT(cycle_number) as total_c,
                AVG(capacity) as avg_cap,
                AVG(soh) as avg_soh
            FROM test_cycles
            GROUP BY battery_id
            ORDER BY battery_id
        """).collect()

        assert len(res) == 2
        assert res[0]["battery_id"] == "B01"
        assert res[0]["total_c"] == 2
        assert round(res[0]["avg_cap"], 2) == 2.00
        assert round(res[1]["avg_soh"], 2) == 96.50
