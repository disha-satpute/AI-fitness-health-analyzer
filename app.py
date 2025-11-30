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

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
MODEL_DIR = "models"

calorie_model = joblib.load(os.path.join(MODEL_DIR, "model_gradient_boosting.pkl"))
sleep_model = joblib.load(os.path.join(MODEL_DIR, "sleep_clf.pkl"))
diet_model = joblib.load(os.path.join(MODEL_DIR, "diet_clf.pkl"))
scaler_class = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

# -----------------------------------------------------
# UTILITY FUNCTIONS
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
# MAIN TITLE
# -----------------------------------------------------
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>📊 AI Fitness & Lifestyle Health Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Explore your health insights & predictions powered by Machine Learning</h4>", unsafe_allow_html=True)
st.markdown("---")


# -----------------------------------------------------
# SECTION 1 — BEAUTIFUL DATA VISUALIZATION DASHBOARD
# -----------------------------------------------------
st.header("📈 Interactive Data Dashboard")
st.write("Explore insights from the health dataset by clicking the buttons below.")

# Load dataset
try:
    df = pd.read_csv("fitness_dataset_processed.csv")
except:
    df = None
    st.error("❌ Dataset not found: fitness_dataset_processed.csv")

if df is not None:

    # Layout grid
    col1, col2, col3 = st.columns(3)

    # 1 BMI Histogram
    if col1.button("📌 BMI Distribution"):
        st.subheader("📌 BMI Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df["bmi"], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    # 2 Sleep Hours
    if col2.button("😴 Sleep Hours Distribution"):
        st.subheader("😴 Sleep Hours Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df["sleep_hours"], kde=True, color='green', bins=30, ax=ax)
        st.pyplot(fig)

    # 3 Diet Score
    if col3.button("🥗 Diet Score Distribution"):
        st.subheader("🥗 Diet Score Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df["diet_score_v2"], kde=True, color='orange', bins=30, ax=ax)
        st.pyplot(fig)

    # Row 2
    col4, col5, col6 = st.columns(3)

    if col4.button("🔥 Activity vs Calories"):
        st.subheader("🔥 Activity Level vs Calorie Need")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.boxplot(x="activity_mult", y="daily_calorie_need", data=df, ax=ax)
        st.pyplot(fig)

    if col5.button("⚖️ BMI vs Calories"):
        st.subheader("⚖️ BMI vs Daily Calorie Need")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.scatterplot(x="bmi", y="daily_calorie_need",
                        hue="activity_mult", palette="coolwarm",
                        data=df, ax=ax)
        st.pyplot(fig)

    if col6.button("💧 Water Intake Chart"):
        st.subheader("💧 Water Intake Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df["water_l_per_day"], kde=True, color='skyblue', bins=20, ax=ax)
        st.pyplot(fig)

    # Row 3
    col7, col8, col9 = st.columns(3)

    if col7.button("🍔 Junk Food Consumption"):
        st.subheader("🍔 Junk Food Consumption per Week")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(x="junk_food_per_week", data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    if col8.button("⚡ Stress Score Distribution"):
        st.subheader("⚡ Stress Score Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df["stress_score"], kde=True, color='red', bins=20, ax=ax)
        st.pyplot(fig)

    if col9.button("🍽 Meals per Day Chart"):
        st.subheader("🍽 Meals per Day")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.countplot(x="meals_per_day", data=df, ax=ax)
        st.pyplot(fig)

    # Row 4
    col10, col11, col12 = st.columns(3)

    if col10.button("🛌 Sleep Quality Pie Chart"):
        st.subheader("🛌 Sleep Quality Breakdown")
        fig, ax = plt.subplots(figsize=(6,3))
        df["sleep_quality"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    if col11.button("🍎 Diet Quality Pie Chart"):
        st.subheader("🍎 Diet Quality Breakdown")
        fig, ax = plt.subplots(figsize=(6,3))
        df["diet_quality"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    if col12.button("🔥 Correlation Heatmap"):
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

st.markdown("---")


# -----------------------------------------------------
# SECTION 2 — MACHINE LEARNING PREDICTION
# -----------------------------------------------------
st.header("🤖 AI-Based Calorie, Sleep & Diet Prediction")

with st.expander("📝 Enter Your Details for Prediction", expanded=True):

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
            phone_before_sleep = st.number_input("Phone before Sleep (minutes)", 0, 300, 30)

        col3, col4 = st.columns(2)
        with col3:
            meals = st.slider("Meals per Day", 1, 6, 3)
            junk = st.slider("Junk Food per Week", 0, 20, 2)

        with col4:
            water = st.slider("Water Intake (Liters)", 0.0, 5.0, 2.0)
            fruits = st.slider("Fruit/Veg Servings", 0, 10, 3)
            stress = st.slider("Stress Level (1–5)", 1, 5, 3)

        submit = st.form_submit_button("Predict")

# -----------------------------------------------------
# MAKE PREDICTION
# -----------------------------------------------------
if submit:
    st.markdown("---")
    st.markdown("## 🎯 Your Personalized Health Predictions")
    
    # Feature engineering
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    bmr = bmr_mifflin(gender, weight, height, age)
    activity_mult = ACTIVITY_MAP[activity]
    goal_num = GOAL_MAP[goal]
    phone_sleep_ratio = phone_before_sleep / (sleep_hours + 1e-6)
    diet_score = fruits + (1 if water > 1.5 else 0) - junk

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
        "junk_food_per_week": junk,
        "water_l_per_day": water,
        "fruit_veg_servings": fruits,
        "diet_score_v2": diet_score,
        "stress_score": stress
    }])

    # Predictions
    predicted_cal = int(calorie_model.predict(X_input)[0])
    X_scaled = scaler_class.transform(X_input)

    sleep_pred = sleep_model.predict(X_scaled)[0]
    diet_pred = diet_model.predict(X_scaled)[0]

    sleep_label = SLEEP_LABELS[sleep_pred]
    diet_label = DIET_LABELS[diet_pred]

    # -------------------------------------------
    # 🎛 Display Results
    # -------------------------------------------
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("🔥 Daily Calorie Requirement", f"{predicted_cal} kcal/day")

    with colB:
        st.metric("😴 Sleep Quality", f"{sleep_label}")

    with colC:
        st.metric("🍎 Diet Quality", f"{diet_label}")

    st.markdown("---")
    st.subheader("💡 Personalized Recommendations for You")

    # -------------------------------------------
    # 🧠 Sleep Recommendations
    # -------------------------------------------
    st.markdown("### 😴 Sleep Recommendations")

    if sleep_label == "Poor":
        st.error("Your sleep quality is **Poor** ⚠️")
        st.write("""
        - Reduce phone usage before bedtime  
        - Maintain a consistent sleep schedule  
        - Avoid caffeine after 5 PM  
        - Try meditation or calm music before sleep  
        """)
    elif sleep_label == "Average":
        st.warning("Your sleep quality is **Average** 🙂")
        st.write("""
        - Improve lighting and reduce screen time  
        - Sleep 7–8 hours consistently  
        - Maintain a relaxing pre-sleep routine  
        """)
    else:
        st.success("Your sleep quality is **Good** ✔")
        st.write("""
        - Great job! Maintain your sleep habits  
        - Continue consistent bed & wake-up timings  
        """)

    # -------------------------------------------
    # 🥗 Diet Recommendations
    # -------------------------------------------
    st.markdown("### 🍎 Diet Recommendations")

    if diet_label == "Poor":
        st.error("Your diet quality is **Poor** ⚠️")
        st.write("""
        - Reduce processed & junk food  
        - Add 2–3 servings of fruits/vegetables  
        - Increase water intake  
        - Focus on balanced meals  
        """)
    elif diet_label == "Average":
        st.warning("Your diet quality is **Average** 🙂")
        st.write("""
        - Increase natural fiber & fruits  
        - Keep limiting junk food  
        - Drink 2–3L water daily  
        """)
    else:
        st.success("Your diet quality is **Good** ✔")
        st.write("""
        - Keep up your healthy food habits  
        - Maintain balanced meals with proteins  
        """)

    # -------------------------------------------
    # 🧘 Goal-Based Recommendations
    # -------------------------------------------
    st.markdown("### 🎯 Based on Your Fitness Goal")

    if goal == "Lose Weight":
        st.info("""
        **Goal: Weight Loss**  
        - Reduce daily calories by 300–400  
        - Add light cardio (walking/cycling)  
        - Increase protein, reduce sugar  
        """)
    elif goal == "Gain Weight":
        st.info("""
        **Goal: Weight Gain**  
        - Increase calories by 250–350  
        - Add resistance training  
        - Increase protein intake  
        """)
    else:
        st.info("""
        **Goal: Maintain Weight**  
        - Maintain balanced calorie intake  
        - Stay consistent with current routine  
        """)
