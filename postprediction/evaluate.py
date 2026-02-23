import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load trained CatBoost model
model = joblib.load("catboost_model_correct.pkl")

# Load dataset
df = pd.read_csv("hospital_data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Use SAME features as training
X = df[[
    "Age",
    "Gender",
    "Condition",
    "Procedure",
    "Length_of_Stay"
]]
y = df["Cost"]

# Train-test split (same logic as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 MODEL EVALUATION (CatBoost – Correct Features)")
print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))
