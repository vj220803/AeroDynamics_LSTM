import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.keras")  # saved LSTM model
    scaler = joblib.load("scaler.pkl")      # same scaler used in training
    return model, scaler

model, scaler = load_assets()

# --- Streamlit UI ---
st.set_page_config(page_title="Aircraft Maintenance Prediction", page_icon="✈️", layout="wide")
st.title("✈️ Aircraft Maintenance Prediction (LSTM)")
st.write("Predict whether an aircraft engine will fail within the next 30 cycles.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload Engine Sensor Data (CSV with 50 cycles × 25 features)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Check dimensions
    if df.shape == (50, 25):
        # Scale input
        data = scaler.transform(df.values)
        sequence = np.array(data).reshape(1, 50, 25)

        # Predict
        prob = model.predict(sequence)[0][0]
        pred = "Failure soon" if prob >= 0.5 else "Healthy"

        # Show results
        st.subheader("🔎 Prediction Result")
        st.metric("Failure Probability", f"{prob:.2%}")
        if pred == "Failure soon":
            st.error("⚠️ Engine at risk of failure soon! Schedule maintenance.")
        else:
            st.success("✅ Engine is healthy. No failure expected in next 30 cycles.")
    else:
        st.error("CSV must contain exactly 50 rows (cycles) × 25 columns (features).")
