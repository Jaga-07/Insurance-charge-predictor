import streamlit as st
import numpy as np
import joblib

# Load the trained models
model = joblib.load('best_gradient_boosting_model.pkl')
model1 = joblib.load('linear_regression_model.joblib')

# Title
st.title("💸 Insurance Charge Prediction App 💸")

# Subtitle
st.write("Provide the following details to estimate your insurance charge.")

# User Inputs
claim_amount = st.number_input("🧾 Claim Amount (₹)", min_value=0.0, value=10000.0, format="%.2f")
past_consultations = st.number_input("📋 Number of Past Consultations", min_value=0, value=2)
hospital_expenditure = st.number_input("🏥 Hospital Expenditure (₹)", min_value=0.0, value=5000.0, format="%.2f")
annual_salary = st.number_input("💼 Annual Salary (₹)", min_value=0.0, value=600000.0, format="%.2f")
children = st.number_input("👶 Number of Children", min_value=0, value=1)
smoker = st.selectbox("🚬 Are you a smoker?", ["No", "Yes"])

# Encode smoker
smoker_encoded = 1 if smoker == "Yes" else 0

# Predict Button
if st.button("Predict Insurance Charges 💰"):
    input_data = np.array([[claim_amount, past_consultations, hospital_expenditure,
                            annual_salary, children, smoker_encoded]])

    # Predictions
    prediction = model.predict(input_data)[0]
    prediction1 = model1.predict(input_data)[0]

    # Display Results in Tabs
    tab1, tab2 = st.tabs(["Gradient Boosting Model", "Linear Regression Model"])

    with tab1:
        st.subheader("💸 Gradient Boosting Model Prediction")
        st.success(f"Estimated Insurance Charge: ₹{prediction:,.2f}")

    with tab2:
        st.subheader("💵 Linear Regression Model Prediction")
        st.info(f"Estimated Insurance Charge: ₹{prediction1:,.2f}")
