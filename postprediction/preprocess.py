import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "hospital_data.csv")
SCHEMA_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")  # keep same name

print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

required_columns = [
    "Age",
    "Gender",
    "Condition",
    "Procedure",
    "Length_of_Stay",
    "Cost"
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

feature_columns = [
    "Age",
    "Gender",
    "Condition",
    "Procedure",
    "Length_of_Stay"
]

categorical_cols = ["Gender", "Condition", "Procedure"]

schema = {
    "feature_columns": feature_columns,
    "categorical_cols": categorical_cols
}

joblib.dump(schema, SCHEMA_PATH)

print("✅ preprocessor.pkl saved (schema only)")