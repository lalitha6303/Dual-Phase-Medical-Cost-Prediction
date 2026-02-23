import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("hospital_data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ✅ KEEP ONLY TRUE COST DRIVERS
X = df[[
    "Age",
    "Gender",
    "Condition",
    "Procedure",
    "Length_of_Stay"
]]

y = df["Cost"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
cat_features = [X.columns.get_loc(col) for col in categorical_cols]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train CatBoost
model = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

joblib.dump(model, "catboost_model_correct.pkl")
print("✅ Correct CatBoost model trained")
