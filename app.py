

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="MediPredict Pro", page_icon="🧑‍⚕️", layout="wide")

st.title("🧑‍⚕️ MediPredict Pro")
st.markdown("### Multi-Disease Risk Prediction with Explainable AI")

base_path = './'

@st.cache_resource
def load_model(disease):
    model = joblib.load(f'models/{disease}_model.pkl')
    scaler = joblib.load(f'models/{disease}_scaler.pkl')
    return model, scaler

# Sidebar always visible
disease = st.sidebar.selectbox(
    "Choose Disease",
    ("Diabetes", "Heart Disease", "Liver Cirrhosis Stage", "Parkinson's Disease")
)

# ---------------- Diabetes ----------------
if disease == "Diabetes":
    st.header("Diabetes Risk Prediction")
    model, scaler = load_model("diabetes")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3, key="preg")
        glucose = st.slider("Glucose", 0, 200, 120, key="glu")
        bp = st.slider("Blood Pressure", 0, 122, 70, key="bp")
        skin = st.slider("Skin Thickness", 0, 99, 20, key="skin")
    with col2:
        insulin = st.slider("Insulin", 0, 846, 80, key="ins")
        bmi = st.slider("BMI", 0.0, 67.1, 32.0, step=0.1, key="bmi")
        dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, step=0.01, key="dpf")
        age = st.slider("Age", 21, 81, 29, key="age")

    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled = scaler.transform(features)

    if st.button("Predict"):
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        if pred == 1:
            st.error(f"High Risk of Diabetes ({prob:.1f}% probability)")
        else:
            st.success(f"Low Risk ({100-prob:.1f}% probability)")

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        st.subheader("Why this prediction?")
        shap.plots.waterfall(shap_values[0])

# ---------------- Heart Disease ----------------
elif disease == "Heart Disease":
    st.header("Heart Disease Risk Prediction")
    model, scaler = load_model("heart")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 50, key="age_h")
        sex = st.selectbox("Sex", ("Male", "Female"), key="sex_h")
        sex = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"), key="cp_h")
        cp = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)
        trestbps = st.slider("Resting BP", 90, 200, 120, key="trestbps")
    with col2:
        chol = st.slider("Cholesterol", 100, 600, 200, key="chol")
        thalach = st.slider("Max Heart Rate", 70, 220, 150, key="thalach")
        exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"), key="exang_h")
        exang = 1 if exang == "Yes" else 0
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1, key="oldpeak")

    features = np.array([[age, sex, cp, trestbps, chol, thalach, exang, oldpeak]])
    scaled = scaler.transform(features)

    if st.button("Predict"):
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        if pred == 1:
            st.error(f"High Risk of Heart Disease ({prob:.1f}% probability)")
        else:
            st.success(f"Low Risk ({100-prob:.1f}% probability)")

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        st.subheader("Why this prediction?")
        shap.plots.waterfall(shap_values[0])

# ---------------- Liver Cirrhosis ----------------
elif disease == "Liver Cirrhosis Stage":
    st.header("Liver Cirrhosis Stage Prediction")
    model, scaler = load_model("cirrhosis")

    bilirubin = st.slider("Bilirubin", 0.0, 10.0, 1.2, step=0.1)
    albumin = st.slider("Albumin", 1.0, 6.0, 3.5, step=0.1)
    protime = st.slider("Prothrombin Time", 10, 20, 12)
    ascites = st.selectbox("Ascites", ("No", "Yes"))
    ascites = 1 if ascites == "Yes" else 0

    features = np.array([[bilirubin, albumin, protime, ascites, age, edema, stage]])
    scaled = scaler.transform(features)

    if st.button("Predict"):
        pred = model.predict(scaled)[0]
        st.success(f"Predicted Cirrhosis Stage: {pred}")

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        st.subheader("Why this prediction?")
        shap.plots.waterfall(shap_values[0])

# ---------------- Parkinson's ----------------
elif disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Risk Prediction")
    model, scaler = load_model("parkinsons")

    mdvp_fo = st.slider("MDVP:Fo(Hz)", 80.0, 300.0, 150.0, step=0.1)
    jitter = st.slider("Jitter(%)", 0.0, 1.0, 0.005, step=0.001)
    shimmer = st.slider("Shimmer", 0.0, 1.0, 0.02, step=0.001)
    nhr = st.slider("NHR", 0.0, 1.0, 0.02, step=0.001)

    features = np.array([[mdvp_fo, jitter, shimmer, nhr, spread1, spread2, PPE]])
    scaled = scaler.transform(features)

    if st.button("Predict"):
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100
        if pred == 1:
            st.error(f"High Risk of Parkinson's ({prob:.1f}% probability)")
        else:
            st.success(f"Low Risk ({100-prob:.1f}% probability)")

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        st.subheader("Why this prediction?")
        shap.plots.waterfall(shap_values[0])

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built by you | XGBoost + SHAP")
