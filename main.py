import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
# Load Model & Preprocessing
model = tf.keras.models.load_model("churn_model.h5")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
metadata = joblib.load("metadata.pkl")

categorical_cols = metadata["categorical_cols"]
numerical_cols = metadata["numerical_cols"]
category_sizes = metadata["category_sizes"]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will churn using a trained Deep Learning model.")
# User Input Form

st.header("Enter Customer Details")

user_input = {}

# Collect categorical inputs
for col in categorical_cols:
    options = list(label_encoders[col].classes_)
    user_input[col] = st.selectbox(f"{col}", options)

# Collect numerical inputs
for col in numerical_cols:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Prepare model input

def prepare_input_streamlit(user_input):
    inputs = []

    # categorical (reshape)
    for col in categorical_cols:
        val = label_encoders[col].transform([user_input[col]])[0]
        inputs.append(np.array([[val]]))

    # numerical
    numeric_values = np.array([[user_input[col] for col in numerical_cols]], dtype=float)
    numeric_values = scaler.transform(numeric_values)
    inputs.append(numeric_values)

    return inputs

# Predict Button
if st.button("Predict Churn"):
    model_input = prepare_input_streamlit(user_input)
    proba = model.predict(model_input).flatten()[0]
    pred = int(proba > 0.5)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{proba*100:.2f}%")

    with col2:
        st.metric("Final Prediction", "YES (Churn)" if pred==1 else "NO (Not Churn)")

    if pred == 1:
        st.warning("This customer is likely to churn. Consider sending retention offers.")
    else:
        st.success("This customer is unlikely to churn.")

st.write("---")
st.caption("Customer Churn Prediction App • Powered by Streamlit + Deep Learning")
