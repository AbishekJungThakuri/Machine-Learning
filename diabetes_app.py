import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Prediction App")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female","Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have any heart disease?", ["No", "Yes"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=500.0, value=100.0)
smoking = st.selectbox("Smoking History", [
    "No Info", "current", "ever", "former", "never", "not current"
])

# Mapping and one-hot encoding
gender_map = {"Other":2,"Male": 1, "Female": 0}
smoking_categories = [
    "No Info", "current", "ever", "former", "never", "not current"
]
# Create one-hot encoding manually
smoking_encoded = [1 if smoking == cat else 0 for cat in smoking_categories]

# Create feature input as DataFrame
input_data = pd.DataFrame([[
    gender_map[gender],
    age,
    1 if hypertension == "Yes" else 0,
    1 if heart_disease == "Yes" else 0,
    bmi,
    hba1c,
    glucose,
] + smoking_encoded], columns=[
    "gender", "age", "hypertension", "heart_disease", "bmi",
    "HbA1c_level", "blood_glucose_level",
    "smoking_history_No Info", "smoking_history_current",
    "smoking_history_ever", "smoking_history_former",
    "smoking_history_never", "smoking_history_not current"
])

# Predict
if st.button("Predict Diabetes"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts: **Diabetes detected** (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The model predicts: **No Diabetes** (Probability: {prob:.2f})")
