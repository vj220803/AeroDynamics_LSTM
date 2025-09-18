# app.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os

# --- Try to import TF safely ---
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None
    tf_import_error = e
else:
    tf_import_error = None

# --- Streamlit page setup ---
st.set_page_config(page_title="Aircraft Maintenance Prediction", page_icon="✈️", layout="wide")
st.title("✈️ Aircraft Maintenance Prediction (LSTM)")
st.write("Predict whether an aircraft engine will fail within the next 30 cycles.")

# --- Expected filenames ---
MODEL_NAME = "lstm_model.keras"
SCALER_NAME = "scaler.pkl"
SEQCOLS_NAME = "sequence_cols.json"

# --- Debug: show files in container ---
repo_root = Path.cwd()
st.sidebar.markdown("### 📂 Files in repo root")
for f in os.listdir(repo_root):
    st.sidebar.write("-", f)

# --- Helper: search for file ---
def find_file(filename: str):
    """Look for filename in repo root and models/ folder."""
    root_path = repo_root / filename
    models_path = repo_root / "models" / filename
    if root_path.exists():
        return root_path
    elif models_path.exists():
        return models_path
    return None

# --- Helper: load model + scaler ---
@st.cache_resource
def load_assets():
    if tf_import_error is not None:
        return None, None, f"❌ TensorFlow import failed: {tf_import_error}"

    model_path = find_file(MODEL_NAME)
    scaler_path = find_file(SCALER_NAME)

    if model_path is None:
        return None, None, f"❌ Model file `{MODEL_NAME}` not found (searched repo root and models/)"
    if scaler_path is None:
        return None, None, f"❌ Scaler file `{SCALER_NAME}` not found (searched repo root and models/)"

    try:
        model = load_model(str(model_path))
    except Exception as e:
        return None, None, f"❌ Failed to load model: {e}"

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        return None, None, f"❌ Failed to load scaler: {e}"

    return model, scaler, None

# --- Prediction function ---
def safe_predict(model, scaler, df: pd.DataFrame):
    try:
        data = scaler.transform(df.values)
    except Exception as e:
        raise RuntimeError(f"Scaler.transform failed: {e}")

    seq = np.array(data).reshape((1, data.shape[0], data.shape[1]))
    pred = model.predict(seq)

    prob = None
    if hasattr(pred, "__len__"):
        p = np.array(pred)
        if p.size == 1:
            prob = float(p.flatten()[0])
        elif p.shape[-1] == 2:
            prob = float(p[0, 1])  # binary classification
        else:
            prob = float(p.flatten()[0])
    else:
        prob = float(pred)

    return max(0.0, min(1.0, prob))

# --- Load model & scaler ---
model, scaler, load_error = load_assets()

if load_error:
    st.warning(load_error)
    st.info("👉 Upload model & scaler below if not bundled in repo.")
else:
    st.success("✅ Model & scaler loaded successfully!")

# --- Upload fallback ---
uploaded_model = st.file_uploader("Upload model file (.keras or .h5)", type=["keras", "h5"])
uploaded_scaler = st.file_uploader("Upload scaler (.pkl)", type=["pkl"])

if uploaded_model and uploaded_scaler:
    with open("uploaded_model.keras", "wb") as f:
        f.write(uploaded_model.getbuffer())
    with open("uploaded_scaler.pkl", "wb") as f:
        f.write(uploaded_scaler.getbuffer())
    model, scaler, load_error = load_assets()
    if not load_error:
        st.success("✅ Model & scaler loaded from uploads.")

# --- CSV upload ---
st.header("Upload Engine Sensor Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    rows, cols = df.shape
    if rows < 50 or cols < 25:
        st.error("❌ CSV must have at least 50 rows × 25 columns.")
        st.stop()

    df = df.iloc[:50, :25]

    if model is None or scaler is None:
        st.error("❌ Model and/or scaler not loaded.")
        st.stop()

    try:
        df_numeric = df.astype(float)
        prob = safe_predict(model, scaler, df_numeric)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    label = "Failure soon" if prob >= 0.5 else "Healthy"
    st.metric("Failure Probability", f"{prob:.2%}")
    if label == "Failure soon":
        st.error("⚠️ Engine at risk of failure soon!")
    else:
        st.success("✅ Engine is healthy.")
