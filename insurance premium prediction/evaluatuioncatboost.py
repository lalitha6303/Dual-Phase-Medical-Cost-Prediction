import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load model
model = joblib.load("insurance_catboost_model.pkl")

# Load data
df = pd.read_csv("insurance (2) (1).csv")

X = df.drop("charges", axis=1)
y = df["charges"]

# Predict
y_pred = model.predict(X)

# Evaluate
print("R² Score :", r2_score(y, y_pred))
print("MAE :", mean_absolute_error(y, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y, y_pred)))
