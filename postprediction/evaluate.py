import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "hospital_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "catboost_model_correct.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

print("🔹 Loading model and data...")
model = joblib.load(MODEL_PATH)
schema = joblib.load(SCHEMA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")

X = df[schema["feature_columns"]]
y = df["Cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔹 Predicting...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 MODEL EVALUATION (CatBoost – Final)")
print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))


print("\n🔹 Running Cross-Validation (CatBoost safe)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # IMPORTANT: recreate model each fold
    cv_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    cat_features = [X.columns.get_loc(col) for col in schema["categorical_cols"]]

    cv_model.fit(X_tr, y_tr, cat_features=cat_features)

    preds = cv_model.predict(X_val)
    score = r2_score(y_val, preds)
    cv_scores.append(score)

print("CV R² Mean:", round(np.mean(cv_scores), 4))
print("CV R² Std :", round(np.std(cv_scores), 4))

# --------------------------------------------------
# 📊 Actual vs Predicted (Hospital)
# --------------------------------------------------
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Error column (very useful for viva)
results_df["Error"] = results_df["Actual"] - results_df["Predicted"]

print("\n🔍 Hospital Sample Predictions (Actual vs Predicted)")
print(results_df.head(10))