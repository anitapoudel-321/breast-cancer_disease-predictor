import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================================================
# Load model and scaler (must match training exactly)
# =========================================================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🩺 Breast Cancer Diagnosis Predictor (SVM)")
st.write(
    "Enter the cell nuclei measurements below. "
    "The model predicts whether the tumor is **Benign** or **Malignant** "
    "based on an SVM trained on the Wisconsin Breast Cancer dataset."
)

# =========================================================
# Feature list — MUST match training order exactly
# (mean -> se -> worst, same order as the original CSV
# after dropping 'id' and 'Unnamed: 32')
# =========================================================
feature_groups = {
    "Mean": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    ],
    "Standard Error": [
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",
    ],
    "Worst": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ],
}

all_features = feature_groups["Mean"] + feature_groups["Standard Error"] + feature_groups["Worst"]

# Reasonable default values (rough dataset means) so the form isn't full of zeros
default_values = {
    "radius_mean": 14.13, "texture_mean": 19.29, "perimeter_mean": 91.97, "area_mean": 654.89,
    "smoothness_mean": 0.096, "compactness_mean": 0.104, "concavity_mean": 0.089,
    "concave points_mean": 0.048, "symmetry_mean": 0.181, "fractal_dimension_mean": 0.063,
    "radius_se": 0.405, "texture_se": 1.217, "perimeter_se": 2.866, "area_se": 40.34,
    "smoothness_se": 0.007, "compactness_se": 0.025, "concavity_se": 0.032,
    "concave points_se": 0.012, "symmetry_se": 0.021, "fractal_dimension_se": 0.004,
    "radius_worst": 16.27, "texture_worst": 25.68, "perimeter_worst": 107.26, "area_worst": 880.58,
    "smoothness_worst": 0.132, "compactness_worst": 0.254, "concavity_worst": 0.272,
    "concave points_worst": 0.115, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.084,
}

inputs = {}

tab1, tab2, tab3 = st.tabs(["Mean Values", "Standard Error", "Worst Values"])

with tab1:
    cols = st.columns(2)
    for i, feat in enumerate(feature_groups["Mean"]):
        inputs[feat] = cols[i % 2].number_input(
            feat, value=default_values[feat], format="%.5f", key=feat
        )

with tab2:
    cols = st.columns(2)
    for i, feat in enumerate(feature_groups["Standard Error"]):
        inputs[feat] = cols[i % 2].number_input(
            feat, value=default_values[feat], format="%.5f", key=feat
        )

with tab3:
    cols = st.columns(2)
    for i, feat in enumerate(feature_groups["Worst"]):
        inputs[feat] = cols[i % 2].number_input(
            feat, value=default_values[feat], format="%.5f", key=feat
        )

st.divider()

# =========================================================
# Predict
# =========================================================
if st.button("Predict", type="primary"):
    X = pd.DataFrame([[inputs[f] for f in all_features]], columns=all_features)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    decision_score = model.decision_function(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = np.max(proba) * 100
    st.success(f"Confidence: {confidence:.2f}%")

    label = "Malignant" if prediction == 1 else "Benign"
    color = "red" if prediction == 1 else "green"

    st.markdown(f"## Prediction: :{color}[{label}]")
    st.caption(
        f"SVM decision score: {decision_score:.3f} "
        "(further from 0 = more confident; this model was not trained with "
        "probability calibration, so no percentage confidence is available)"
    )

    if prediction == 1:
        st.warning(
            "⚠️ This is a demo tool using a small research dataset — "
            "not a medical diagnosis. Please consult a healthcare professional."
        )
    else:
        st.info(
            "This is a demo tool using a small research dataset — "
            "not a medical diagnosis. Please consult a healthcare professional."
        )