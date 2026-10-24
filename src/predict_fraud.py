
import os
import joblib
import pandas as pd

# Locate the saved model and dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

# Load model and transaction data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Select one transaction from the dataset
transaction = df.drop(columns=["Class"]).iloc[[0]]

# Predict
prediction = model.predict(transaction)[0]
probabilities = model.predict_proba(transaction)[0]

print("Transaction features loaded successfully.")
print("Predicted class:", int(prediction))
print("Fraud probability:", round(float(probabilities[1]) * 100, 2), "%")

if prediction == 1:
    print("Result: Potentially fraudulent transaction")
else:
    print("Result: Likely legitimate transaction")
