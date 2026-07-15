"""
FlowSentinel Mini-Project: Customer Churn Prediction — Streamlit App
Run locally with:  streamlit run app.py
(Install streamlit first:  pip install streamlit)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/churn_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    return model, encoders, feature_columns

model, encoders, feature_columns = load_artifacts()

st.title("📉 Customer Churn Prediction")
st.write("Base deployment demo — Random Forest model trained on customer usage data.")

st.divider()
st.subheader("Enter customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))

st.divider()

if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Apply the same label encoders used during training
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — probability: {proba:.1%}")
    else:
        st.success(f"✅ Likely to Stay — churn probability: {proba:.1%}")

    st.progress(float(proba))

st.divider()

