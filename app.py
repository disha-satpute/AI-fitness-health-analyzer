import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="AI Fitness Health Analyzer",
    layout="wide",
    page_icon="💪"
)

sns.set(style="whitegrid")

# Load models
MODEL_DIR = "models"

calorie_model = joblib.load(os.path.join(MODEL_DIR, "model_gradient_boosting.pkl"))
sleep_model = joblib.load(os.path.join(MODEL_DIR, "sleep_clf.pkl"))
diet_model = joblib.load(os.path.join(MODEL_DIR, "diet_clf.pkl"))
scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

# Utility functions
def bmr_mifflin(gender, weight, height, age):
    return 10*weight + 6.25*height - 5*age + (5 if gender == "male" else -161)

ACTIVITY_MAP = {
    "Sedentary": 1.2,
    "Light": 1.375,
    "Moderate": 1.55,
    "Active": 1.725,
    "Very Active": 1.9,
}

GOAL_MAP = {"Lose Weight": -1, "Maintain Weight": 0, "Gain Weight": 1}

SLEEP_LABELS = {0: "Poor", 1: "Average", 2: "Good"}
DIET_LABELS = {0: "Poor", 1: "Average", 2: "Good"}


# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Prediction", "📊 Dashboard", "ℹ️ About Project"]
)


# -----------------------------------------------------
# PAGE 1 — PREDICTION PAGE (FIRST PRIORITY)
# -----------------------------------------------------
if page == "🏠 Prediction":

    st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🤖 AI Fitness Health Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Enter your details to get calorie, sleep & diet predictions</h4>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📝 Enter Your Details", expanded=True):

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 15, 80, 22)
                gender = st.selectbox("Gender", ["male", "female"])
                weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
                height = st.number_input("Height (cm)", 120.0, 220.0, 165.0)

            with col2:
                activity = st.selectbox("Activity Level", list(ACTIVITY_MAP.keys()))
                goal = st.selectbox("Goal", list(GOAL_MAP.keys()))
                sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
                phone = st.number_input("Phone Use Before Sleep (minutes)", 0, 300, 30)

            col3, col4 = st.columns(2)

            with col3:
                meals = st.slider("Meals/day", 1, 6, 3)
                junk = st.slider("Junk Food/week", 0, 20, 2)

            with col4:
                water = st.slider("Water Intake (L/day)", 0.0, 5.0, 2.0)
                fruits = st.slider("Fruit/Veg Servings", 0, 10, 3)
                stress = st.slider("Stress Level (1–5)", 1, 5, 3)

            submit = st.form_submit_button("Predict")

    if submit:
        st.markdown("## 🎯 Prediction Results")

        # Feature Engineering
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        bmr = bmr_mifflin(gender, weight, height, age)

        data = pd.DataFrame([{
            "age": age,
            "gender_num": 1 if gender == "male" else 0,
            "weight_kg": weight,
            "height_cm": height,
            "bmi": bmi,
            "bmr": bmr,
            "activity_mult": ACTIVITY_MAP[activity],
            "goal_num": GOAL_MAP[goal],
            "sleep_hours": sleep_hours,
            "phone_before_sleep": phone,
            "phone_sleep_ratio": phone / (sleep_hours + 1e-6),
            "meals_per_day": meals,
            "junk_food_per_week": junk,
            "water_l_per_day": water,
            "fruit_veg_servings": fruits,
            "diet_score_v2": fruits + (1 if water > 1.5 else 0) - junk,
            "stress_score": stress
        }])

        predicted_cal = int(calorie_model.predict(data)[0])
        scaled = scaler_class.transform(data)

        sleep_out = SLEEP_LABELS[sleep_model.predict(scaled)[0]]
        diet_out = DIET_LABELS[diet_model.predict(scaled)[0]]

        colA, colB, colC = st.columns(3)
        colA.metric("🔥 Calories Needed", f"{predicted_cal} kcal/day")
        colB.metric("😴 Sleep Quality", sleep_out)
        colC.metric("🍎 Diet Quality", diet_out)

        st.markdown("---")


# -----------------------------------------------------
# PAGE 2 — DASHBOARD WITH SMALL CHARTS + TABS
# -----------------------------------------------------
elif page == "📊 Dashboard":

    st.title("📊 Health Data Dashboard")
    st.write("Compact visualizations to understand lifestyle patterns.")

    try:
        df = pd.read_csv("fitness_dataset_processed.csv")
    except:
        st.error("Dataset missing.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📌 Basics", "🔥 Calories", "🍎 Diet", "😴 Sleep"])

    # ---------------- TAB 1 — BASICS -----------------
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.histplot(df["bmi"], kde=True, bins=20, ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("Water Intake")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.histplot(df["water_l_per_day"], kde=True, color="skyblue", ax=ax)
            st.pyplot(fig, use_container_width=False)

    # ---------------- TAB 2 — CALORIES -----------------
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Activity vs Calories")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.boxplot(x="activity_mult", y="daily_calorie_need", data=df, ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("BMI vs Calories")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.scatterplot(x="bmi", y="daily_calorie_need", data=df, ax=ax)
            st.pyplot(fig, use_container_width=False)

    # ---------------- TAB 3 — DIET -----------------
    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Diet Score Distribution")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.histplot(df["diet_score_v2"], kde=True, color="orange", ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("Junk Food Intake")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.countplot(x="junk_food_per_week", data=df, ax=ax)
            st.pyplot(fig, use_container_width=False)

    # ---------------- TAB 4 — SLEEP -----------------
    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sleep Quality Pie")
            fig, ax = plt.subplots(figsize=(3,3))
            df["sleep_quality"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            st.pyplot(fig, use_container_width=False)

        with col2:
            st.subheader("Stress Score")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.histplot(df["stress_score"], kde=True, color="red", ax=ax)
            st.pyplot(fig, use_container_width=False)


# -----------------------------------------------------
# PAGE 3 — ABOUT
# -----------------------------------------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.write("""
    This project predicts:
    - Daily calorie requirement  
    - Sleep quality  
    - Diet quality  
    using Machine Learning models trained on a 10K+ dataset.
    
    It includes:
    - Regression (Calorie Prediction)
    - Classification (Sleep & Diet)
    - Compact Visualization Dashboard
    - Streamlit Cloud Deployment  
    """)
