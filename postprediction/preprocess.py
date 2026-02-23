import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
df = pd.read_csv("hospital_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Drop ID and target
X = df.drop(columns=["Patient_ID", "Cost"])

# Auto-detect column types
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Fit
preprocessor.fit(X)

# Save
joblib.dump(preprocessor, os.path.join(BASE_DIR, "preprocessor.pkl"))
print("✅ Preprocessing saved")
