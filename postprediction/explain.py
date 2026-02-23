import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from catboost import Pool

# Load correct model
model = joblib.load("catboost_model_correct.pkl")

# Load dataset
df = pd.read_csv("hospital_data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# SAME features used in training
X = df[[
    "Age",
    "Gender",
    "Condition",
    "Procedure",
    "Length_of_Stay"
]]

# Identify categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
cat_features = [X.columns.get_loc(col) for col in categorical_cols]

# Create Pool
pool = Pool(data=X, cat_features=cat_features)

# SHAP values
shap_values = model.get_feature_importance(
    pool, type="ShapValues"
)

# Remove base value column
shap_values = shap_values[:, :-1]

# Mean absolute SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Average Impact on Cost": mean_abs_shap
}).sort_values(by="Average Impact on Cost", ascending=False)

print("\n📊 TRUE COST DRIVER IMPACT (SHAP)")
print(shap_df)

# Plot
plt.figure(figsize=(8, 5))
plt.barh(
    shap_df["Feature"],
    shap_df["Average Impact on Cost"],
    color="#4C72B0"
)
plt.gca().invert_yaxis()
plt.title(
    "True Factors Influencing Hospital Cost (SHAP)",
    fontsize=14,
    weight="bold"
)
plt.xlabel("Average Absolute Impact on Cost")
plt.tight_layout()
plt.show()
