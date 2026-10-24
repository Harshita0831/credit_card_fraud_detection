
import os
import sys

spark_home = os.environ["SPARK_HOME"]

sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(
    0,
    os.path.join(spark_home, "python", "lib", "py4j-0.10.9.7-src.zip")
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = (
    SparkSession.builder
    .appName("FraudDataAnalysis")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

# Count legitimate and fraudulent transactions
print("\nTransaction class distribution:")
df.groupBy("Class").agg(
    count("*").alias("Total")
).orderBy("Class").show()

# Calculate the percentage of each class
total = df.count()

print("Total transactions:", total)

df.groupBy("Class").count().withColumn(
    "Percentage",
    col("count") * 100 / total
).orderBy("Class").show()

# Check for missing values
print("Missing values per column:")
df.select([
    count(col(c)).alias(c) for c in df.columns
]).show(truncate=False)

spark.stop()
