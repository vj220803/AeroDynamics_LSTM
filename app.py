# app.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os
import io

# try/except import for TF to give a nicer error if missing
try:
    from tensorflow.keras.models import load_model
except Exception as e:
    load_model = None
    tf_import_error = e
else:
    tf_import_error = None

st.set_page_config(page_title="Aircraft Maintenance Prediction", page_icon="✈️", layout="wide")
st.title("✈️ Aircraft Maintenance Prediction (LSTM)")
st.write("Predict whether an aircraft engine will fail within the next 30 cycles.")

MODEL_FILENAME = "lstm_model.keras"
SCALER_FILENAME = "scaler.pkl"

# --- Helper functions ---
@st.cache_resource
def load_assets_from_paths(model_path: Path, scaler_path: Path):
    """Load model and scaler from filesystem paths. Return (model, scaler, error_message)."""
    if tf_import_error is not None:
        return None, None, f"TensorFlow import failed: {tf_import_error}"

    if not model_path.exists():
        return None, None, f"Model file not found at: {model_path}"
    if not scaler_path.exists():
        return None, None, f"Scaler file not found at: {scaler_path}"

    try:
        model = load_model(str(model_path))
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    try:
        scaler = joblib.load(str(scaler_path))
    except Exception as e:
        return None, None, f"Failed to load scaler: {e}"

    return model, scaler, None

def safe_predict(model, scaler, df: pd.DataFrame):
    """Scale and reshape df, run model.predict and return probability (float 0..1)."""
    try:
        data = scaler.transform(df.values)
    except Exception as e:
        raise RuntimeError(f"Scaler.transform failed: {e}")

    # Ensure shape is (1, timesteps, features)
    seq = np.array(data)
    if seq.ndim == 1:
        # single row -> make (1, timesteps, features) is ambiguous, raise
        raise ValueError("Input after scaling is 1D. Expected 2D (rows x cols).")
    seq = seq.reshape((1, seq.shape[0], seq.shape[1]))
    try:
        pred = model.predict(seq)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # handle different shapes e.g., (1,1), (1,), (1,2)
    prob = None
    if hasattr(pred, "__len__"):
        p = np.array(pred)
        if p.size == 1:
            prob = float(p.flatten()[0])
        else:
            # if model outputs probability for classes, try to pick class 1 prob
            if p.shape[-1] == 2:
                prob = float(p[0, 1])
            else:
                prob = float(p.flatten()[0])
    else:
        prob = float(pred)

    # clamp 0..1
    prob = max(0.0, min(1.0, prob))
    return prob

# --- UI: show where to place model/scaler ---
st.sidebar.header("Model & Scaler")
st.sidebar.write(
    "The app will look for these files in the same folder as this script:\n\n"
    f"- `{MODEL_FILENAME}` (Keras .keras/.h5 saved model)\n"
    f"- `{SCALER_FILENAME}` (joblib .pkl scaler used during training)\n\n"
    "If they are not present, upload them below."
)

# If files exist in repo folder, try to load them
repo_root = Path.cwd()
model_path = repo_root / MODEL_FILENAME
scaler_path = repo_root / SCALER_FILENAME

model, scaler, load_error = load_assets_from_paths(model_path, scaler_path)

# If missing, allow user to upload model & scaler
if load_error:
    st.warning(load_error)
    st.info("You can upload model and scaler files here (or place them in the same folder as this script).")

    uploaded_model = st.file_uploader("Upload model file (.keras or .h5)", type=["keras", "h5"], key="model_upload")
    uploaded_scaler = st.file_uploader("Upload scaler (.pkl)", type=["pkl"], key="scaler_upload")

    if uploaded_model is not None and uploaded_scaler is not None:
        try:
            # save to temporary BytesIO and load
            tmp_model_path = repo_root / "uploaded_model.keras"
            with open(tmp_model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())

            tmp_scaler_path = repo_root / "uploaded_scaler.pkl"
            with open(tmp_scaler_path, "wb") as f:
                f.write(uploaded_scaler.getbuffer())

            model, scaler, load_error = load_assets_from_paths(tmp_model_path, tmp_scaler_path)
            if load_error:
                st.error(load_error)
            else:
                st.success("Model and scaler loaded from uploads.")
        except Exception as e:
            st.error(f"Failed to save/load uploaded files: {e}")
    else:
        st.info("Upload both model and scaler to proceed. (You can still upload CSV to preview.)")
else:
    st.success("Model and scaler loaded from local files.")

# --- CSV upload and prediction ---
st.header("Upload Engine Sensor Data (CSV)")
st.write("CSV expected: 50 rows (timesteps/cycles) × 25 columns (features).")

uploaded_file = st.file_uploader("Upload Engine Sensor Data (CSV)", type=["csv"], key="csv_upload")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head())

        rows, cols = df.shape
        ok_shape = (rows == 50 and cols == 25)
        if not ok_shape:
            st.warning(f"Uploaded CSV shape is {rows} rows × {cols} columns.")
            # Attempt to auto-adjust if larger than needed
            if rows >= 50 and cols >= 25:
                st.info("Taking first 50 rows and first 25 columns automatically.")
                df_adj = df.iloc[:50, :25].copy()
                df = df_adj
            else:
                st.error("CSV must have at least 50 rows and 25 columns. Please upload correct file.")
                st.stop()

        # ensure model & scaler available
        if model is None or scaler is None:
            st.error("Model and/or scaler not loaded. Upload them in the sidebar or place them in the app folder.")
            st.stop()

        # Convert any non-numeric to numeric if possible
        try:
            df_numeric = df.astype(float)
        except Exception as e:
            st.warning(f"Converting data to numeric where possible: {e}")
            df_numeric = df.apply(pd.to_numeric, errors="coerce")
            if df_numeric.isna().any().any():
                st.error("Some values could not be converted to numeric. Please clean your CSV.")
                st.stop()

        # Prediction
        try:
            prob = safe_predict(model, scaler, df_numeric)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        label = "Failure soon" if prob >= 0.5 else "Healthy"
        st.subheader("🔎 Prediction Result")
        st.metric("Failure Probability", f"{prob:.2%}")
        if label == "Failure soon":
            st.error("⚠️ Engine at risk of failure soon! Schedule maintenance.")
        else:
            st.success("✅ Engine is healthy. No failure expected in next 30 cycles.")

# --- Footer / debug info ---
st.markdown("---")
st.markdown("**Notes & troubleshooting**")
st.markdown(
    "- Make sure your model was trained to accept input shape `(50, 25)` (timesteps=50, features=25).\n"
    "- If the app shows `TensorFlow import failed`, install TensorFlow in the environment where Streamlit runs.\n"
    "- For large models, consider running the app on Colab or a cloud host (Streamlit Cloud / Hugging Face Spaces / Render)."
)
