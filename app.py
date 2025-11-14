"""
Robust Streamlit app for predicting concrete compressive strength (csMPa).

This version has a more robust model-loading routine that tries:
 - joblib.load(file_path)
 - pickle.loads(bytes)
 - dill.loads(bytes)
 - cloudpickle.loads(bytes)

It also attempts to extract the estimator if the pickle contains a wrapper (dict/object).
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import traceback

# Try to import optional libraries
try:
    import joblib
except Exception:
    joblib = None

try:
    import dill
except Exception:
    dill = None

try:
    import cloudpickle
except Exception:
    cloudpickle = None

import pickle
from typing import Any, Optional

# --------- Config ----------
st.set_page_config(page_title="Concrete Strength Predictor (robust loader)", layout="centered")
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


# --------- Robust loading helpers ----------
def try_joblib_load(path: str):
    if joblib is None:
        raise RuntimeError("joblib not available in environment")
    return joblib.load(path)


def try_pickle_load_bytes(b: bytes):
    return pickle.loads(b)


def try_dill_load_bytes(b: bytes):
    if dill is None:
        raise RuntimeError("dill not available in environment")
    return dill.loads(b)


def try_cloudpickle_load_bytes(b: bytes):
    if cloudpickle is None:
        raise RuntimeError("cloudpickle not available in environment")
    return cloudpickle.loads(b)


def extract_estimator(obj: Any):
    """
    If obj is a wrapper like a dict or sklearn GridSearchCV, try to extract a usable estimator.
    Returns estimator or None.
    """
    # common attributes / keys that might contain the estimator
    candidates = []

    if obj is None:
        return None

    # If it's already has predict -> done
    if hasattr(obj, "predict"):
        return obj

    # If it's a dict-like, check keys
    try:
        if isinstance(obj, dict):
            candidates += ["model", "estimator", "best_estimator_", "pipeline", "clf"]
            for k in candidates:
                if k in obj and hasattr(obj[k], "predict"):
                    return obj[k]
    except Exception:
        pass

    # If object has attribute 'best_estimator_' (GridSearchCV)
    for attr in ("best_estimator_", "estimator_", "model", "pipeline"):
        try:
            candidate = getattr(obj, attr, None)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
        except Exception:
            continue

    # Some wrappers store the sklearn object under .clf or .estimator
    for attr in ("clf", "estimator"):
        try:
            candidate = getattr(obj, attr, None)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
        except Exception:
            continue

    return None


def robust_load_from_path(path: str):
    """
    Try loading model from the filesystem using joblib then pickle.
    Returns (model, errors_list)
    """
    errors = []
    # 1) try joblib.load (file-based)
    if joblib is not None:
        try:
            m = try_joblib_load(path)
            est = extract_estimator(m)
            return (est or m), errors
        except Exception as e:
            errors.append(f"joblib.load failed: {repr(e)}\n{traceback.format_exc()}")

    # 2) fallback to pickle.load (file-based)
    try:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            m = try_pickle_load_bytes(raw)
            est = extract_estimator(m)
            return (est or m), errors
        except Exception as e:
            errors.append(f"pickle.loads (from file bytes) failed: {repr(e)}\n{traceback.format_exc()}")
            # try dill/cloudpickle too
            if dill is not None:
                try:
                    m = try_dill_load_bytes(raw)
                    est = extract_estimator(m)
                    return (est or m), errors
                except Exception as ex:
                    errors.append(f"dill.loads (from file bytes) failed: {repr(ex)}\n{traceback.format_exc()}")
            if cloudpickle is not None:
                try:
                    m = try_cloudpickle_load_bytes(raw)
                    est = extract_estimator(m)
                    return (est or m), errors
                except Exception as ex:
                    errors.append(f"cloudpickle.loads (from file bytes) failed: {repr(ex)}\n{traceback.format_exc()}")
    except Exception as e:
        errors.append(f"Failed to read file bytes: {repr(e)}\n{traceback.format_exc()}")

    return None, errors


def robust_load_from_bytes(b: bytes):
    """
    Try multiple in-memory loaders. Returns (model_or_none, errors_list)
    """
    errors = []
    # 1) plain pickle
    try:
        m = try_pickle_load_bytes(b)
        est = extract_estimator(m)
        return (est or m), errors
    except Exception as e:
        errors.append(f"pickle.loads failed: {repr(e)}\n{traceback.format_exc()}")

    # 2) dill
    if dill is not None:
        try:
            m = try_dill_load_bytes(b)
            est = extract_estimator(m)
            return (est or m), errors
        except Exception as e:
            errors.append(f"dill.loads failed: {repr(e)}\n{traceback.format_exc()}")

    # 3) cloudpickle
    if cloudpickle is not None:
        try:
            m = try_cloudpickle_load_bytes(b)
            est = extract_estimator(m)
            return (est or m), errors
        except Exception as e:
            errors.append(f"cloudpickle.loads failed: {repr(e)}\n{traceback.format_exc()}")

    # 4) write to a temp file and try joblib.load if available
    if joblib is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                tmp.write(b)
                tmp.flush()
                tmp_path = tmp.name
            try:
                m = joblib.load(tmp_path)
                est = extract_estimator(m)
                # cleanup
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return (est or m), errors
            except Exception as e:
                errors.append(f"joblib.load (from temp file) failed: {repr(e)}\n{traceback.format_exc()}")
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            errors.append(f"failed to write temp file for joblib attempt: {repr(e)}\n{traceback.format_exc()}")

    return None, errors


# --------- Prediction helpers ----------
def predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise RuntimeError("Loaded object does not have a .predict method. It's not a usable estimator.")


def validate_input_df(df: pd.DataFrame) -> Optional[str]:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"
    return None


# --------- Load default model (if present) ----------
default_model, default_load_errors = None, []
if os.path.exists(MODEL_PATH):
    default_model, default_load_errors = robust_load_from_path(MODEL_PATH)


# --------- App UI ----------
st.title("🧱 Concrete Compressive Strength Predictor — Robust Loader")
st.write(
    "This app attempts several ways to load a model pickle (joblib/pickle/dill/cloudpickle). "
    "If your model was created with a specific serializer, make sure that library is installed in the environment."
)

st.markdown("### Model")
col_status, col_upload = st.columns([2, 3])

with col_status:
    if default_model is not None:
        st.success(f"Default model loaded from `{MODEL_PATH}`")
        st.write(f"Model type: `{type(default_model).__name__}`")
    else:
        st.warning(f"No usable model found at `{MODEL_PATH}`.")
        if default_load_errors:
            with st.expander("Why default model couldn't be loaded (details)"):
                for e in default_load_errors:
                    st.text(e.splitlines()[0])
                st.write("Open expander to see full traces.")
            with st.expander("Full loader trace for default model"):
                for e in default_load_errors:
                    st.code(e)

with col_upload:
    uploaded_model_file = st.file_uploader(
        "Upload model (.pkl/.joblib) — optional (will override default model for this session)", type=["pkl", "pickle", "joblib"]
    )

model = default_model
upload_errors = []
if uploaded_model_file is not None:
    # read bytes
    bytes_data = uploaded_model_file.read()
    model, upload_errors = robust_load_from_bytes(bytes_data)
    if model is None:
        st.error("Uploaded file couldn't be loaded as a model. See details below.")
        with st.expander("Upload load attempts and errors"):
            for err in upload_errors:
                st.code(err)
        st.info(
            "Common fixes: (1) Make sure the .pkl was created with standard joblib/pickle/dill/cloudpickle. "
            "If you used a custom environment or custom classes, recreate the pipeline with a supported serializer "
            "or include the custom class definitions in the environment."
        )
    else:
        st.success("Uploaded model loaded successfully.")
        st.write(f"Model type: `{type(model).__name__}`")


st.write("---")

# Single prediction UI
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
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.exception(e)

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
                    st.exception(e)

st.write("---")
st.caption("App created by Tejal")
