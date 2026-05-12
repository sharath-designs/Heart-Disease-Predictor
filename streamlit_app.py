import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load files
# -------------------------------
model = joblib.load('KNN.heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

# -------------------------------
# Reset Button
# -------------------------------
# if st.button("🔄 Reset"):
#     st.experimental_rerun()

# -------------------------------
# Inputs
# -------------------------------
Age = st.slider("Age", 1, 100, 25)
RestingBP = st.slider("Resting Blood Pressure", 80, 200, 120)
Cholesterol = st.slider("Cholesterol", 100, 400, 200)
FastingBS = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
MaxHR = st.slider("Max Heart Rate", 60, 220, 150)
Oldpeak = st.slider("Oldpeak", 0.0, 5.0, 1.0)

Sex = st.selectbox("Sex", ["M", "F"])
ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------
# Input Summary
# -------------------------------
st.subheader("📋 Input Summary")

input_dict = {
    "Age": Age,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "MaxHR": MaxHR,
    "Oldpeak": Oldpeak,
    "Sex": Sex,
    "ChestPainType": ChestPainType,
    "RestingECG": RestingECG,
    "ExerciseAngina": ExerciseAngina,
    "ST_Slope": ST_Slope
}

st.write(pd.DataFrame([input_dict]))

# -------------------------------
# Encoding
# -------------------------------
Sex_M = 1 if Sex == "M" else 0

ChestPainType_ATA = 1 if ChestPainType == "ATA" else 0
ChestPainType_NAP = 1 if ChestPainType == "NAP" else 0
ChestPainType_TA  = 1 if ChestPainType == "TA" else 0

RestingECG_Normal = 1 if RestingECG == "Normal" else 0
RestingECG_ST = 1 if RestingECG == "ST" else 0

ExerciseAngina_Y = 1 if ExerciseAngina == "Y" else 0

ST_Slope_Flat = 1 if ST_Slope == "Flat" else 0
ST_Slope_Up = 1 if ST_Slope == "Up" else 0

# -------------------------------
# Arrange Input
# -------------------------------
input_data = [
    Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
    Sex_M,
    ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
    RestingECG_Normal, RestingECG_ST,
    ExerciseAngina_Y,
    ST_Slope_Flat, ST_Slope_Up
]

input_array = np.array([input_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("🧠 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"📊 Confidence: {round(np.max(probability)*100, 2)}%")

    # -------------------------------
    # Basic Data Insights
    # -------------------------------
    st.subheader("📈 Basic Insights")

    if Age > 50:
        st.warning("Age above 50 may increase risk")

    if Cholesterol > 240:
        st.warning("High cholesterol detected")

    if RestingBP > 140:
        st.warning("High blood pressure detected")

    if MaxHR < 100:
        st.warning("Low max heart rate")

    # -------------------------------
    # Feature Importance (Explanation)
    # -------------------------------
    st.subheader("🔎 Model Insight")

    st.info(
        "KNN model does not provide direct feature importance. "
        "Prediction is based on similarity to nearest patients in dataset."
    )