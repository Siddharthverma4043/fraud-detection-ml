import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🔍 Fraud Detection ML App")

st.write("Enter transaction details below:")

# Input fields
step = st.number_input("Step (Time)", min_value=1)
amount = st.number_input("Amount")
oldbalanceOrg = st.number_input("Old Balance Origin")
newbalanceOrig = st.number_input("New Balance Origin")
oldbalanceDest = st.number_input("Old Balance Dest")
newbalanceDest = st.number_input("New Balance Dest")
isFlaggedFraud = st.selectbox("Is Flagged Fraud", [0, 1])
type_CASH_OUT = st.selectbox("Is CASH_OUT Type?", [0, 1])
type_DEBIT = st.selectbox("Is DEBIT Type?", [0, 1])
type_PAYMENT = st.selectbox("Is PAYMENT Type?", [0, 1])
type_TRANSFER = st.selectbox("Is TRANSFER Type?", [0, 1])

# Button
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': isFlaggedFraud,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER
    }])
    
    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]
    
    if pred == 1:
        st.error(f"⚠️ FRAUD DETECTED with {prob:.2f} confidence.")
    else:
        st.success(f"✅ Legit Transaction (Confidence: {prob:.2f})")
