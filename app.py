import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Fitness App", layout="centered")
sns.set(style="whitegrid")

# -----------------------------------------------------
# Load Models
# -----------------------------------------------------
MODEL_DIR = "models"

calorie_model = joblib.load(os.path.join(MODEL_DIR, "model_gradient_boosting.pkl"))
sleep_model = joblib.load(os.path.join(MODEL_DIR, "sleep_clf.pkl"))
diet_model = joblib.load(os.path.join(MODEL_DIR, "diet_clf.pkl"))
scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
def bmr_mifflin(gender, weight, height, age):
    if gender == "male":
        return 10*weight + 6.25*height - 5*age + 5
    return 10*weight + 6.25*height - 5*age - 161

ACTIVITY_MAP = {
    "Sedentary": 1.2,
    "Light": 1.375,
    "Moderate": 1.55,
    "Active": 1.725,
    "Very Active": 1.9,
}

GOAL_MAP = {
    "Lose Weight": -1,
    "Maintain Weight": 0,
    "Gain Weight": 1,
}

SLEEP_LABELS = {0: "Poor", 1: "Average", 2: "Good"}
DIET_LABELS = {0: "Poor", 1: "Average", 2: "Good"}

# -----------------------------------------------------
# App UI
# -----------------------------------------------------
st.title("🏋️ AI Fitness & Calorie Recommendation System")
st.write("Enter details to get calorie needs + diet & sleep predictions.")

with st.form("form"):
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
        phone_before_sleep = st.number_input("Phone Use Before Sleep (minutes)", 0, 300, 30)

    col3, col4 = st.columns(2)
    with col3:
        meals = st.slider("Meals per Day", 1, 6, 3)
        junk_food = st.slider("Junk Food per Week", 0, 20, 2)

    with col4:
        water = st.slider("Water Intake (Liters)", 0.0, 5.0, 2.0)
        fruits = st.slider("Fruit/Veg Servings", 0, 10, 3)
        stress = st.slider("Stress Level (1–5)", 1, 5, 3)

    submit = st.form_submit_button("Predict")

# -----------------------------------------------------
# Process Input
# -----------------------------------------------------
if submit:
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    bmr = bmr_mifflin(gender, weight, height, age)
    activity_mult = ACTIVITY_MAP[activity]
    goal_num = GOAL_MAP[goal]
    phone_sleep_ratio = phone_before_sleep / (sleep_hours + 1e-6)
    diet_score = fruits + (1 if water > 1.5 else 0) - junk_food

    X_input = pd.DataFrame([{
        "age": age,
        "gender_num": 1 if gender == "male" else 0,
        "weight_kg": weight,
        "height_cm": height,
        "bmi": bmi,
        "bmr": bmr,
        "activity_mult": activity_mult,
        "goal_num": goal_num,
        "sleep_hours": sleep_hours,
        "phone_before_sleep": phone_before_sleep,
        "phone_sleep_ratio": phone_sleep_ratio,
        "meals_per_day": meals,
        "junk_food_per_week": junk_food,
        "water_l_per_day": water,
        "fruit_veg_servings": fruits,
        "diet_score_v2": diet_score,
        "stress_score": stress
    }])

    # -----------------------------------------------------
    # Calorie Prediction (NO SCALING NEEDED)
    # -----------------------------------------------------
    predicted_calories = int(calorie_model.predict(X_input)[0])

    # -----------------------------------------------------
    # Classification (WITH SCALING)
    # -----------------------------------------------------
    X_scaled = scaler_class.transform(X_input)

    sleep_pred = SLEEP_LABELS[int(sleep_model.predict(X_scaled)[0])]
    diet_pred = DIET_LABELS[int(diet_model.predict(X_scaled)[0])]

    # -----------------------------------------------------
    # Output Section
    # -----------------------------------------------------
    st.subheader("🔢 Calorie Prediction")
    st.success(f"Estimated Calories Needed: **{predicted_calories} kcal/day**")

    st.subheader("🛌 Sleep Quality Prediction")
    st.info(f"Sleep Quality: **{sleep_pred}**")

    st.subheader("🍎 Diet Quality Prediction")
    st.info(f"Diet Quality: **{diet_pred}**")

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------
    st.subheader("💡 Personalized Suggestions")

    if goal == "Lose Weight":
        st.write("- Reduce daily calorie intake by 300–400 kcal.")
        st.write("- Add 30–40 mins walking or light cardio.")

    elif goal == "Gain Weight":
        st.write("- Increase daily calorie intake by 250–350 kcal.")
        st.write("- Increase protein and strength training.")

    else:
        st.write("- Maintain current diet and exercise routine.")

    if sleep_pred == "Poor":
        st.write("⚠ Reduce phone use before sleep. Try to sleep ≥ 7 hrs.")
    elif sleep_pred == "Average":
        st.write("🙂 Improve by reducing screen time & increasing sleep consistency.")
    else:
        st.write("✔ Great! Maintain your good sleep routine.")

    if diet_pred == "Poor":
        st.write("⚠ Reduce junk food. Add fruits, vegetables, and water.")
    elif diet_pred == "Average":
        st.write("🙂 Add 1–2 servings of fruits or vegetables daily.")
    else:
        st.write("✔ Diet looks good! Continue balanced eating.")
