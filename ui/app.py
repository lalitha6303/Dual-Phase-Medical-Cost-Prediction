import streamlit as st
import pandas as pd
import joblib
import sqlite3
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dual-Phase Medical Cost Prediction with Coverage Eligibility Detection",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------
# DATABASE (users.db)
# --------------------------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# --------------------------------------------------
# AUTH FUNCTIONS
# --------------------------------------------------
def login_user(u, p):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
    return cur.fetchone()

def register_user(u, p):
    try:
        cur.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        return True
    except:
        return False

# --------------------------------------------------
# LOAD MODELS & DATA
# --------------------------------------------------
hospital_model = joblib.load("catboost_model_correct.pkl")
insurance_model = joblib.load("insurance_model.pkl")

hospital_df = pd.read_csv("hospital_data.csv")
insurance_df = pd.read_csv("insurance (2) (1).csv")

# --------------------------------------------------
# SESSION
# --------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# --------------------------------------------------
# LOGIN / REGISTER
# --------------------------------------------------
if st.session_state.user is None:
    st.title("Dual-Phase Medical Cost Prediction with Coverage Eligibility Detection")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        ru = st.text_input("New Username")
        rp = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(ru, rp):
                st.success("Registration successful. Please login.")
            else:
                st.error("User already exists")

    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.success(f"Logged in as {st.session_state.user}")
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------
st.title("Dual-Phase Medical Cost Prediction with Coverage Eligibility Detection")
tab1, tab2, tab3 = st.tabs([
    "🏥 Post-Treatment Costs",
    "🛡 Insurance Premium Prediction",
    "🏛 Healthcare Schemes Recommender"
])


# ==================================================
# 🏥 HOSPITAL POST-PREDICTION
# ==================================================
with tab1:
    st.subheader("Hospital Cost Prediction (Post-Diagnosis)")

    # --- Dynamic dropdown logic ---
    procedures = sorted(hospital_df["Procedure"].unique())
    procedure = st.selectbox("Select Procedure", procedures)

    filtered = hospital_df[hospital_df["Procedure"] == procedure]
    conditions = sorted(filtered["Condition"].unique())
    condition = st.selectbox("Select Condition", conditions)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        stay = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5)

    if st.button("Predict Hospital Cost"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Condition": condition,
            "Procedure": procedure,
            "Length_of_Stay": stay
        }])

        cost = hospital_model.predict(input_df)[0]
        st.success(f"Estimated Hospital Cost: ₹ {cost:,.2f}")

   

    st.markdown("### 🔍 Explainability")

    if st.button("Show Hospital Explainability"):
      st.info("This graph shows which factors influence hospital cost the most overall.")

      img = Image.open("hospital_shap.png")
      st.image(img, use_container_width=True)


# ==================================================
# 🛡 INSURANCE PREDICTION
# ==================================================
with tab2:
    st.subheader("Insurance Cost Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age ", 0, 120, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with col2:
        children = st.number_input("Children", 0, 10, 1)
        smoker = st.selectbox("Smoker", ["yes", "no"])
    with col3:
        gender = st.selectbox("Gender ", ["male", "female"])
        region = st.selectbox("Region", insurance_df["region"].unique())

    if st.button("Predict Insurance Cost"):
        df = pd.DataFrame([{
            "age": age,
            "sex": gender,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        cost = insurance_model.predict(df)[0]
        st.success(f"Estimated Annual Insurance Cost: ₹ {cost:,.2f}")

    st.markdown("### 🔍 Explainability")

    if st.button("Show Insurance Explainability"):
       st.info("This graph shows which factors influence insurance cost the most overall.")

       img2 = Image.open("insurance_importance.png")
       st.image(img2, use_container_width=True)


with tab3:
    st.subheader("🏛 Government Healthcare Schemes (India)")
    st.caption("Schemes that help reduce medical and hospitalization costs")

    schemes_df = pd.read_csv("schemes.csv")

    search = st.text_input(
        "🔍 Search by Scheme name or Treatment",
        placeholder="e.g. Cancer, Dialysis, Ayushman"
    )

    if search:
        search = search.lower()
        schemes_df = schemes_df[
            schemes_df.apply(
                lambda row:
                search in row["Scheme"].lower()
                or search in row["Covered_Treatments"].lower()
                or search in row["Eligibility"].lower(),
                axis=1
            )
        ]

    st.dataframe(
        schemes_df,
        use_container_width=True,
        height=450
    )

    st.markdown("### 📌 Scheme Details")
    selected = st.selectbox(
        "Select a scheme to view full details",
        ["Select"] + schemes_df["Scheme"].tolist()
    )

    if selected != "Select":
        row = schemes_df[schemes_df["Scheme"] == selected].iloc[0]
        st.write(f"**Scheme Name:** {row['Scheme']}")
        st.write(f"**Max Amount:** {row['Max_Amount']}")
        st.write(f"**Eligibility:** {row['Eligibility']}")
        st.write(f"**Covered Treatments:** {row['Covered_Treatments']}")
        st.info(row["Notes"])

