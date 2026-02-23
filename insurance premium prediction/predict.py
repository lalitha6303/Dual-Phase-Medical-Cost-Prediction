import pandas as pd
import joblib

# Load trained model (CatBoost or Gradient Boosting)
model = joblib.load("insurance_model.pkl")
# OR for GB:
# model = joblib.load("insurance_model.pkl")

# New unseen person
new_person = pd.DataFrame({
    "age": [45],
    "sex": ["male"],
    "bmi": [32.0],
    "children": [2],
    "smoker": ["no"],
    "region": ["southeast"]
})

predicted_cost = model.predict(new_person)

print("Predicted yearly insurance cost:", predicted_cost[0])
