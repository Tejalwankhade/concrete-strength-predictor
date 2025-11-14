import streamlit as st
import numpy as np
import pandas as pd
import os

# Try robust model loading (joblib first, then pickle)
def load_model(path="Concrete_Strength_Model.pkl"):
    try:
        import joblib
        model = joblib.load(path)
        return model
    except Exception:
        pass
    try:
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{path}': {e}")

st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

st.title("Concrete Strength Predictor (csMPa)")
st.write(
    "Enter the input features (in SI/appropriate units used during training) "
    "and click **Predict**. Make sure `Concrete_Strength_Model.pkl` is in the same folder."
)

MODEL_PATH = "Concrete_Strength_Model.pkl"

# Load model (with friendly message)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
else:
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.info("Place your Concrete_Strength_Model.pkl in the same directory as this script.")
    st.stop()

# Input defaults (you can change these to dataset means if you want)
col1, col2 = st.columns(2)
with col1:
    cement = st.number_input("cement (kg/m³)", min_value=0.0, value=300.0, step=1.0, format="%.3f")
    slag = st.number_input("slag (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.3f")
    flyash = st.number_input("flyash (kg/m³)", min_value=0.0, value=0.0, step=1.0, format="%.3f")
    water = st.number_input("water (kg/m³)", min_value=0.0, value=180.0, step=0.1, format="%.3f")
with col2:
    superplasticizer = st.number_input("superplasticizer (kg/m³)", min_value=0.0, value=0.0, step=0.01, format="%.3f")
    coarseaggregate = st.number_input("coarseaggregate (kg/m³)", min_value=0.0, value=1000.0, step=1.0, format="%.3f")
    fineaggregate = st.number_input("fineaggregate (kg/m³)", min_value=0.0, value=800.0, step=1.0, format="%.3f")
    age = st.number_input("age (days)", min_value=1, value=28, step=1, format="%d")

st.markdown("---")
st.write("**Optional:** Load example presets")
preset = st.selectbox("Example preset", ["None", "Typical mix (300,0,0,180,0,1000,800,28)"])
if preset != "None":
    if preset == "Typical mix (300,0,0,180,0,1000,800,28)":
        cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age = (
            300.0, 0.0, 0.0, 180.0, 0.0, 1000.0, 800.0, 28
        )
        st.success("Preset loaded into inputs (you can still modify values).")

# Prepare input for model
features = np.array(
    [[cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]],
    dtype=float,
)

st.write("")
if st.button("Predict csMPa"):
    try:
        # If model expects a DataFrame with column names, try to create one
        try:
            # many models accept numpy array; but some pipelines expect DataFrame with specific columns
            if hasattr(model, "predict"):
                # Try array prediction first
                pred = model.predict(features)
            else:
                # fallback: if model is a function
                pred = model(features)
        except Exception:
            # Try DataFrame with expected column names
            cols = ["cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "age"]
            df_input = pd.DataFrame(features, columns=cols)
            pred = model.predict(df_input)

        # Ensure pred is array-like
        if isinstance(pred, (list, tuple, np.ndarray, pd.Series)):
            predicted_value = float(np.asarray(pred).ravel()[0])
        else:
            predicted_value = float(pred)

        st.success(f"Predicted concrete strength: **{predicted_value:.3f} csMPa**")
        st.write("✅ Note: This app assumes your model was trained on raw feature values in the same order.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("If your model was trained using a preprocessing pipeline (scaler, encoder), include the pipeline in the saved model (recommended).")

st.markdown("---")
st.write("Model info:")
try:
    st.write(type(model))
    # Try showing pipeline steps if it's a sklearn Pipeline
    if hasattr(model, "named_steps"):
        st.write("Pipeline steps:")
        st.write(list(model.named_steps.keys()))
except Exception:
    pass

st.caption("App created by Tejal")
