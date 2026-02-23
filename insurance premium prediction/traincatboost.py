import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv("insurance (2) (1).csv")

X = df.drop("charges", axis=1)
y = df["charges"]

# Identify categorical feature indices
categorical_features = ["sex", "smoker", "region"]
cat_features_index = [X.columns.get_loc(col) for col in categorical_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_state=42,
    verbose=100
)

# Train
model.fit(
    X_train,
    y_train,
    cat_features=cat_features_index,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# Save model
joblib.dump(model, "insurance_catboost_model.pkl")

print("✅ CatBoost model trained and saved")
