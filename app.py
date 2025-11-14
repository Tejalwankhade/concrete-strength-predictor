"""
streamlit_concrete_strength_app.py

Streamlit app for predicting concrete compressive strength (csMPa).

Usage:
    1. Place your model pickle at /mnt/data/Concrete_Strength_Model.pkl (or change MODEL_PATH).
    2. Or upload a model via the UI (supports pickle files containing sklearn pipeline or estimator).
    3. Run: streamlit run streamlit_concrete_strength_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
from typing import Optional

# --------- Config ----------
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
MODEL_PATH = "/mnt/data/Concrete_Strength_Model.pkl"
FEATURES = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]

# --------- Helpers ----------
@st.cache(allow_output_mutation=True)
def load_model_from_path(path: str):
    """Load a pickle model from disk. Returns None if not found or fails."""
    if not os.path.exists(path):
        return None
    try:
        with open(Concrete_Strength_Model.pkl, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

@st.cache(allow_output_mutation=True)
def load_model_from_bytes(b: bytes):
    """Load a model from uploaded bytes (pickle)."""
    try:
        model = pickle.loads(b)
        return model
    except Exception:
        return None

def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """Run predict using model; if model has predict_proba fallback not used here."""
    # If model is a pipeline/estimator conforming to sklearn API, .predict will work
    return np.asarray(model.predict(X))

def validate_input_df(df: pd.DataFrame) -> Optional[str]:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"
    return None

# --------- Load default model (if present) ----------
default_model = load_model_from_path(MODEL_PATH)

# --------- App UI ----------
st.title("🧱 Concrete Compressive Strength Predictor")
st.write(
    "Predict the compressive strength (csMPa) of concrete from composition and age. "
    "You can use the default model path or upload your own model (.pkl)."
)

st.markdown("### Model")
model_status_col, model_upload_col = st.columns([2, 3])

with model_status_col:
    if default_model is not None:
        st.success(f"Default model loaded from: `{MODEL_PATH}`")
        st.write(f"Model type: `{type(default_model).__name__}`")
    else:
        st.warning(f"No model found at `{MODEL_PATH}`. Upload a model below or place it there.")

with model_upload_col:
    uploaded_model_file = st.file_uploader(
        "Upload model (.pkl) — optional (will override default model for this session)", type=["pkl", "pickle"]
    )

# Use uploaded model if provided, else default
model = default_model
if uploaded_model_file is not None:
    bytes_data = uploaded_model_file.read()
    uploaded_model = load_model_from_bytes(bytes_data)
    if uploaded_model is None:
        st.error("Uploaded file couldn't be loaded as a pickle model.")
    else:
        model = uploaded_model
        st.success("Uploaded model loaded and will be used for predictions.")
        st.write(f"Uploaded model type: `{type(model).__name__}`")

st.write("---")

# Single-prediction inputs
st.subheader("Single prediction")
col1, col2 = st.columns(2)
with col1:
    cement = st.number_input("Cement (kg/m³)", value=300.0, min_value=0.0, step=1.0, format="%.2f")
    slag = st.number_input("Blast Furnace Slag (kg/m³)", value=0.0, min_value=0.0, step=1.0, format="%.2f")
    flyash = st.number_input("Fly Ash (kg/m³)", value=0.0, min_value=0.0, step=1.0, format="%.2f")
    water = st.number_input("Water (kg/m³)", value=180.0, min_value=0.0, step=1.0, format="%.2f")

with col2:
    superplasticizer = st.number_input(
        "Superplasticizer (kg/m³)", value=0.0, min_value=0.0, step=0.01, format="%.3f"
    )
    coarseaggregate = st.number_input(
        "Coarse Aggregate (kg/m³)", value=1040.0, min_value=0.0, step=1.0, format="%.2f"
    )
    fineaggregate = st.number_input(
        "Fine Aggregate (kg/m³)", value=676.0, min_value=0.0, step=1.0, format="%.2f"
    )
    age = st.number_input("Age (days)", value=28, min_value=1, step=1)

input_dict = {
    "cement": cement,
    "slag": slag,
    "flyash": flyash,
    "water": water,
    "superplasticizer": superplasticizer,
    "coarseaggregate": coarseaggregate,
    "fineaggregate": fineaggregate,
    "age": age,
}

st.markdown("**Input summary**")
st.json(input_dict)

predict_col, info_col = st.columns([2, 1])
with predict_col:
    if st.button("Predict csMPa"):
        if model is None:
            st.error("No model available. Upload a model or place it at the default path.")
        else:
            try:
                X = pd.DataFrame([input_dict], columns=FEATURES)
                preds = predict_with_model(model, X)
                pred_val = float(preds[0])
                st.success(f"Predicted compressive strength: **{pred_val:.3f} MPa**")
                st.info("Treat this as a model estimate. Validate on real-world data before using in production.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

with info_col:
    st.write("Model info")
    if model is None:
        st.warning("Model not loaded.")
    else:
        try:
            st.write(f"Model class: `{type(model).__name__}`")
        except Exception:
            pass

st.write("---")

# Batch predictions via CSV
st.subheader("Batch predictions (CSV)")
st.write("Upload a CSV with columns: " + ", ".join(FEATURES))
csv_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch_csv")

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None

    if df is not None:
        validation_error = validate_input_df(df)
        if validation_error:
            st.error(validation_error)
            st.write("Sample columns found in uploaded file: ", df.columns.tolist()[:20])
        else:
            if model is None:
                st.error("No model available for predictions.")
            else:
                try:
                    preds = predict_with_model(model, df[FEATURES])
                    df_out = df.copy()
                    df_out["pred_csMPa"] = preds
                    st.success("Batch predictions complete — preview below.")
                    st.dataframe(df_out.head(50))
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions CSV",
                        data=csv_bytes,
                        file_name="concrete_predictions_with_csMPa.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error during batch prediction: {e}")

st.write("---")
st.caption("App created by Tejal")
