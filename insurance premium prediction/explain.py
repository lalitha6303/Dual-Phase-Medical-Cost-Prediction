import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# ===============================
# 1. LOAD SAVED PIPELINE & DATA
# ===============================

pipeline = joblib.load("insurance_model.pkl")

df = pd.read_csv("insurance (2) (1).csv")

# Apply SAME preprocessing as training
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

X = df[["age", "bmi", "children", "smoker"]]

# ===============================
# 2. EXTRACT PREPROCESSOR & MODEL
# ===============================

preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["model"]

# ===============================
# 3. GET ENCODED FEATURE NAMES
# ===============================

encoded_features = preprocessor.get_feature_names_out()
importances = model.feature_importances_

fi_df = pd.DataFrame({
    "encoded_feature": encoded_features,
    "importance": importances
})

# ===============================
# 4. MAP TO SIMPLE FEATURE NAMES
# ===============================

def simplify_feature(name):
    if "age" in name:
        return "Age"
    elif "bmi" in name:
        return "BMI"
    elif "children" in name:
        return "Children"
    elif "smoker" in name:
        return "Smoker"
    elif "region" in name:
        return "Region"
    elif "sex" in name:
        return "Sex"
    else:
        return "Other"

fi_df["feature"] = fi_df["encoded_feature"].apply(simplify_feature)

# Aggregate importance by original feature
final_fi = (
    fi_df.groupby("feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

print("\n📊 Simplified Feature Importance:")
print(final_fi)

# ===============================
# 5. BEAUTIFUL USER-FRIENDLY PLOT
# ===============================

plt.figure(figsize=(9, 5))

bars = plt.barh(
    final_fi["feature"],
    final_fi["importance"],
    color="#4C72B0"
)

plt.gca().invert_yaxis()
plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Feature", fontsize=11)
plt.title("Key Factors Influencing Insurance Cost", fontsize=14, weight="bold")

# Add values on bars
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center",
        fontsize=10
    )

plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# ===============================
# 6. SHAP EXPLAINABILITY (OPTIONAL)
# ===============================

X_processed = preprocessor.transform(X)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_processed)

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=encoded_features
)

# ===============================
# 7. SHAP LOCAL EXPLANATION
# ===============================

sample = X.iloc[[0]]
sample_processed = preprocessor.transform(sample)

shap_value_single = explainer.shap_values(sample_processed)

shap.force_plot(
    explainer.expected_value,
    shap_value_single,
    feature_names=encoded_features,
    matplotlib=True
)
