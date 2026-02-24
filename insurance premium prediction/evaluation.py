import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("🔹 Loading model...")

# Load trained pipeline
pipeline = joblib.load("insurance_model.pkl")

print("🔹 Loading data...")

# Load dataset
df = pd.read_csv("insurance (2) (1).csv")

# Clean column names (safe practice)
df.columns = df.columns.str.strip()

X = df.drop("charges", axis=1)
y = df["charges"]

print("🔹 Train-test split (same as training)...")

# IMPORTANT: same split logic as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔹 Predicting on test set...")

# ✅ Predict ONLY on test data (no leakage)
y_pred = pipeline.predict(X_test)


# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 MODEL EVALUATION (Insurance – Final)")
print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))
# --------------------------------------------------
# 📊 Actual vs Predicted Table
# --------------------------------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Optional: error column (very useful)
results_df["Error"] = results_df["Actual"] - results_df["Predicted"]

print("\n🔍 Sample Predictions (Actual vs Predicted)")
print(results_df.head(10))
