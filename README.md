# credit_card_fraud_detection
# Credit Card Fraud Detection Using Apache Spark and Machine Learning

## Project Overview

This project focuses on detecting potentially fraudulent credit card transactions using Apache Spark and machine learning techniques. It analyses transaction data, trains classification models, and evaluates their performance using different evaluation metrics.

Apache Spark MLlib is used to implement Logistic Regression and Decision Tree classifiers. A separate scikit-learn Decision Tree model is trained and saved using Joblib to demonstrate transaction prediction.

**Project Type:** Academic Project  
**Course:** INT315 – Cluster Computing  
**Programming Language:** Python

## Objectives

- Analyse a credit card transaction dataset using Apache Spark.
- Examine the distribution of legitimate and fraudulent transactions.
- Identify missing values and understand class imbalance.
- Train Logistic Regression and Decision Tree models using Spark MLlib.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Train and save a separate scikit-learn Decision Tree model.
- Demonstrate prediction using a saved machine learning model.

## Dataset

This project uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset details:

- Total transactions: 284,807
- Total columns: 31
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Target column: `Class`
- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset is not included in this repository because of its file size.

### Dataset Setup

1. Download `creditcard.csv` from the Kaggle link above.
2. Create or open the `data` folder in the project directory.
3. Place `creditcard.csv` inside the `data` folder.
4. Run the Python scripts from the project directory, using the configured Python and Spark environment.

## Technologies Used

- Python 3.10
- Apache Spark 3.5.9
- PySpark
- Spark MLlib
- scikit-learn
- Pandas
- Joblib
- Java 8
- Visual Studio Code
- Git and GitHub

## Project Structure

```text
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv          # Download separately from Kaggle
├── src/
│   ├── data_loading.py
│   ├── data_analysis.py
│   ├── test_spark.py
│   ├── model_training.py
│   ├── train_and_save_model.py
│   ├── train_sklearn_model.py
│   └── predict_fraud.py
├── models/
│   └── fraud_model.joblib      # Saved scikit-learn model, if included
├── output/
├── screenshots/
├── .gitignore
└── README.md
```

*Note: The structure above describes the intended organisation. Some folders or files may be generated locally when the scripts are executed.*

## Methodology

1. **Data Loading:** Load the credit card transaction dataset.
2. **Data Analysis:** Examine the dataset dimensions, class distribution, and missing values.
3. **Feature Preparation:** Prepare the input features and target label for classification.
4. **Spark MLlib Training:** Train Logistic Regression and Decision Tree models.
5. **Model Evaluation:** Compare performance using confusion matrices, accuracy, precision, recall, and F1-score.
6. **Model Saving:** Train a separate scikit-learn Decision Tree and save it using Joblib.
7. **Prediction:** Load the saved scikit-learn model and demonstrate a prediction on a transaction record.

## Results

### Spark MLlib

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 99.91% | Approximately 99.92% |
| Fraud Precision | 87.50% | 87.18% |
| Fraud Recall | 61.17% | 66.02% |
| Fraud F1-score | 72.00% | 75.14% |

In the observed Spark MLlib experiment, the Decision Tree achieved higher fraud recall and F1-score than Logistic Regression.

### scikit-learn Decision Tree

The separate scikit-learn experiment achieved approximately 96.94% accuracy and 87.76% fraud recall. However, its fraud precision was low because it incorrectly classified many legitimate transactions as fraudulent.

The scikit-learn and Spark MLlib results use different data splits and configurations, so they should be interpreted separately.

## Limitations

- The implementation uses batch-based transaction data.
- Live transaction ingestion and Kafka integration are not implemented.
- Model performance depends on the dataset, training configuration, and evaluation method.
- High accuracy alone is not sufficient to establish effective fraud detection.

## Future Improvements

- Explore Kafka and Spark Streaming for live transaction ingestion.
- Improve class imbalance handling and classification thresholds.
- Reduce false positives while maintaining fraud detection recall.
- Test models using additional datasets and validation techniques.
- Develop a dashboard for visualising suspicious transactions.

## How
