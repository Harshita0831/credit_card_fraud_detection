
import os
import sys

spark_home = os.environ["SPARK_HOME"]

sys.path.insert(0, os.path.join(spark_home, "python"))
sys.path.insert(
    0,
    os.path.join(spark_home, "python", "lib", "py4j-0.10.9.7-src.zip")
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

spark = (
    SparkSession.builder
    .appName("SaveFraudDetectionModel")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load dataset
df = spark.read.csv(
    "data/creditcard.csv",
    header=True,
    inferSchema=True
)

feature_columns = [c for c in df.columns if c != "Class"]

df = df.select(
    *[col(c).cast("double").alias(c) for c in feature_columns],
    col("Class").cast("double").alias("label")
).dropna()

# Build a pipeline: feature assembly + Decision Tree
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

tree = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(stages=[assembler, tree])

# Train on all available data for the saved model
final_model = pipeline.fit(df)


import tempfile

model_path = os.path.join(
    tempfile.gettempdir(),
    "fraud_decision_tree_test"
)

print("Saving model to:", model_path)

try:
    final_model.write().overwrite().save(model_path)
    print("Model saved successfully at:", model_path)
    print("Features used:", len(feature_columns))
except Exception as e:
    print("MODEL SAVE FAILED")
    print(type(e).__name__)
    print(str(e)[:2000])
finally:
    spark.stop()
