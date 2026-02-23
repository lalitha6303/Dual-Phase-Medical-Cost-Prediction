import pandas as pd
import joblib

model = joblib.load("catboost_model.pkl")

new_patient = pd.DataFrame([{
    "Age": 12,
    "Gender": "male",
    "Condition": "Fractured Arm",
    "Procedure": "X-Ray and Splint",
    "Length_of_Stay": 7,
    "Readmission": "No",
    "Outcome": "Recovered",
    "Satisfaction": 5
}])

predicted_cost = model.predict(new_patient)

print("💰 Predicted Hospital Cost:", round(predicted_cost[0], 2))
