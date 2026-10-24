
import os
import sys

SPARK_HOME = os.environ.get("SPARK_HOME")

if not SPARK_HOME:
    raise EnvironmentError("SPARK_HOME is not configured.")

spark_python = os.path.join(SPARK_HOME, "python")
py4j_zip = os.path.join(
    SPARK_HOME, "python", "lib", "py4j-0.10.9.7-src.zip"
)

sys.path.insert(0, spark_python)
sys.path.insert(0, py4j_zip)

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("RealTimeFraudDetection")
    .master("local[*]")
    .getOrCreate()
)

print("Python version:", sys.version.split()[0])
print("Spark version:", spark.version)

spark.stop()
