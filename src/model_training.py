
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
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("FraudDetectionModel")
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

# Select features and label
feature_columns = [
    c for c in df.columns if c not in ["Class"]
]

df = df.select(
    *[col(c).cast("double").alias(c) for c in feature_columns],
    col("Class").cast("double").alias("label")
).dropna()

# Convert feature columns into a single vector
assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

data = assembler.transform(df).select("features", "label")

# Split data: 80% training, 20% testing
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print("Training rows:", train_data.count())
print("Testing rows:", test_data.count())

# Train Logistic Regression using Spark MLlib
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50
)

model = lr.fit(train_data)

# Generate predictions
predictions = model.transform(test_data)

print("\nConfusion matrix (actual rows, predicted columns):")
predictions.groupBy("label").pivot("prediction", [0.0, 1.0]) \
    .count().na.fill(0).orderBy("label").show()


# Evaluate the model with multiple metrics
for metric in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric
    )
    print(f"{metric}: {evaluator.evaluate(predictions):.4f}")

# Calculate fraud-specific precision and recall
tp = predictions.filter(
    (col("label") == 1.0) & (col("prediction") == 1.0)
).count()

fp = predictions.filter(
    (col("label") == 0.0) & (col("prediction") == 1.0)
).count()

fn = predictions.filter(
    (col("label") == 1.0) & (col("prediction") == 0.0)
).count()

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print(f"Fraud precision: {precision:.4f}")
print(f"Fraud recall:    {recall:.4f}")
print(f"Fraud F1-score:  {f1:.4f}")


# Train Decision Tree using Spark MLlib
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

# Calculate fraud-specific metrics
tp_dt = dt_predictions.filter(
    (col("label") == 1.0) & (col("prediction") == 1.0)
).count()

fp_dt = dt_predictions.filter(
    (col("label") == 0.0) & (col("prediction") == 1.0)
).count()

fn_dt = dt_predictions.filter(
    (col("label") == 1.0) & (col("prediction") == 0.0)
).count()

precision_dt = tp_dt / (tp_dt + fp_dt) if tp_dt + fp_dt else 0
recall_dt = tp_dt / (tp_dt + fn_dt) if tp_dt + fn_dt else 0
f1_dt = (
    2 * precision_dt * recall_dt / (precision_dt + recall_dt)
    if precision_dt + recall_dt else 0
)

print("\n--- Decision Tree Results ---")
print("True positives:", tp_dt)
print("False positives:", fp_dt)
print("False negatives:", fn_dt)
print(f"Fraud precision: {precision_dt:.4f}")
print(f"Fraud recall:    {recall_dt:.4f}")
print(f"Fraud F1-score:  {f1_dt:.4f}")

print("\nDecision Tree confusion matrix:")
dt_predictions.groupBy("label").pivot(
    "prediction", [0.0, 1.0]
).count().na.fill(0).orderBy("label").show()