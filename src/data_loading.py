
import os
import sys

spark_home = os.environ["SPARK_HOME"]

sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(
    0,
    os.path.join(
        spark_home, "python", "lib", "py4j-0.10.9.7-src.zip"
    )
)

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("FraudDetection")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load the dataset
df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# Display dataset information
print("Total transactions:", df.count())

print("Dataset columns:")
df.printSchema()

print("First 5 transactions:")
df.show(5, truncate=False)

spark.stop()
