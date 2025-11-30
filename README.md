# 💪 AI Fitness & Health Analyzer  
An AI-powered web application that predicts **daily calorie needs**, classifies **sleep quality**, evaluates **diet quality**, and visualizes lifestyle insights through an interactive dashboard.

Built using **Machine Learning (Regression + Classification)** and deployed using **Streamlit Cloud**.

---

## 🚀 Features

### 🔥 Prediction Module
- **Daily Calorie Requirement** using Gradient Boosting Regression  
- **Sleep Quality Classification** (Poor / Average / Good)  
- **Diet Quality Classification** (Poor / Average / Good)  
- Clean UI with metric cards and personalized lifestyle recommendations  

### 📊 Dashboard & Analytics
- Compact, interactive charts inside **Tabs**
- Lifestyle insights:
  - BMI distribution  
  - Sleep patterns  
  - Diet score trends  
  - Activity vs calorie relationship  
  - Water intake  
  - Junk food consumption  
  - Stress levels  
- Professionally designed compact visualizations  

### 🧠 ML Models Used
- `GradientBoostingRegressor`
- `XGBoostClassifier`
- `RandomForestClassifier`
- `MLPClassifier`
- Preprocessing using StandardScaler  
- All models stored in `/models` as `.pkl` files  

### 🖥 Deployment
- Live on **Streamlit Cloud**
- End-to-end automated pipeline

---

## ⚙️ How It Works

### 1️⃣ User Inputs
The user provides:
- Age, Gender, Weight, Height  
- Sleep hours, phone usage  
- Water intake, junk food frequency  
- Meals/day, fruits & veggies  
- Activity level  
- Stress score  

### 2️⃣ Feature Engineering  
The system computes:
- BMI  
- BMR (Mifflin Equation)  
- Diet Score  
- Phone/Sleep Ratio  
- Activity Multiplier  

### 3️⃣ Predictions  
ML models output:
- 🔥 **Calories Needed**
- 😴 **Sleep Quality**
- 🍎 **Diet Quality**

### 4️⃣ Personalized Suggestions  
Based on predictions, the system generates:
- Sleep improvement tips  
- Diet optimization guidance  
- Goal-based calorie recommendations (Lose / Gain / Maintain weight)

---

## 🛠 Installation

### ▶ Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
