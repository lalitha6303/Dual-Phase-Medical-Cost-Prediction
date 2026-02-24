import pandas as pd
import joblib
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "hospital_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "catboost_model_correct.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

print("🔹 Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Load schema
schema = joblib.load(SCHEMA_PATH)
feature_columns = schema["feature_columns"]
categorical_cols = schema["categorical_cols"]

X = df[feature_columns]
y = df["Cost"]

# CatBoost categorical indices
cat_features = [X.columns.get_loc(col) for col in categorical_cols]

print("🔹 Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔹 Training CatBoost...")
model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

joblib.dump(model, MODEL_PATH)
print("✅ catboost_model_correct.pkl saved")

# Overfitting check
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print("\n📊 Overfitting Check")
print("Train R²:", round(train_r2, 4))
print("Test  R²:", round(test_r2, 4))