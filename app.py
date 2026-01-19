
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt  # Needed for SHAP plots

st.set_page_config(page_title="MediPredict Pro", page_icon="🧑‍⚕️", layout="wide")

st.title("🧑‍⚕️ MediPredict Pro")
st.markdown("### Multi-Disease Risk Prediction with Explainable AI")

@st.cache_resource
def load_model(disease):
    model = joblib.load(f"models/{disease}_model.pkl")
    scaler = joblib.load(f"models/{disease}_scaler.pkl")
    return model, scaler

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
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose", 0, 200, 120)
        bp = st.slider("Blood Pressure", 0, 122, 70)
        skin = st.slider("Skin Thickness", 0, 99, 20)
    with col2:
        insulin = st.slider("Insulin", 0, 846, 80)
        bmi = st.slider("BMI", 0.0, 67.1, 32.0, step=0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5, step=0.01)
        age = st.slider("Age", 21, 81, 29)

    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    st.write("Input shape:", features.shape)
    st.write("Scaler expects:", getattr(scaler, "n_features_in_", "unknown"))
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
        shap.plots.waterfall(shap_values[0], show=True)

# ---------------- Heart Disease ----------------
elif disease == "Heart Disease":
    st.header("Heart Disease Risk Prediction")
    model, scaler = load_model("heart")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ("Male", "Female"))
        sex = 1 if sex == "Male" else 0
        cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"))
        cp = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)
        trestbps = st.slider("Resting BP", 90, 200, 120)
    with col2:
        chol = st.slider("Cholesterol", 100, 600, 200)
        thalach = st.slider("Max Heart Rate", 70, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
        exang = 1 if exang == "Yes" else 0
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)

    # Keep features consistent with your training pipeline (8 common features)
    features = np.array([[age, sex, cp, trestbps, chol, thalach, exang, oldpeak]])
    st.write("Input shape:", features.shape)
    st.write("Scaler expects:", getattr(scaler, "n_features_in_", "unknown")) 
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
        shap.plots.waterfall(shap_values[0], show=True)

# ---------------- Liver Cirrhosis ---------------

  
elif disease == "Liver Cirrhosis Stage":
    st.header("Liver Cirrhosis Stage Prediction")
    model, scaler = load_model("cirrhosis")

    bilirubin = st.slider("Bilirubin", 0.0, 10.0, 1.2, step=0.1)
    albumin = st.slider("Albumin", 1.0, 6.0, 3.5, step=0.1)
    protime = st.slider("Prothrombin Time", 10, 20, 12)
    ascites = st.selectbox("Ascites", ("No", "Yes"))
    ascites = 1 if ascites == "Yes" else 0
    age = st.slider("Age", 20, 90, 50)
    edema = st.selectbox("Edema", ("No", "Yes"))
    edema = 1 if edema == "Yes" else 0
    stage = st.slider("Stage", 1, 4, 2)

    features = np.array([[bilirubin, albumin, protime, ascites, age, edema, stage]])
    st.write("Input shape:", features.shape)
    st.write("Scaler expects:", getattr(scaler, "n_features_in_", "unknown"))
    scaled = scaler.transform(features)

    if st.button("Predict"):
        pred = model.predict(scaled)[0]
        st.success(f"Predicted Cirrhosis Stage: {pred}")

        explainer = shap.Explainer(model)
        shap_values = explainer(scaled)
        st.subheader("Why this prediction?")
        shap.plots.waterfall(shap_values[0], show=True)


# ---------------- Parkinson's ----------------
elif disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Risk Prediction")
    model, scaler = load_model("parkinsons")

    mdvp_fo = st.slider("MDVP:Fo(Hz)", 80.0, 300.0, 150.0, step=0.1)
    mdvp_fhi = st.slider("MDVP:Fhi(Hz)", 100.0, 400.0, 200.0, step=0.1)
    mdvp_flo = st.slider("MDVP:Flo(Hz)", 60.0, 200.0, 100.0, step=0.1)
    jitter = st.slider("Jitter(%)", 0.0, 1.0, 0.11, step=0.001)
    shimmer = st.slider("Shimmer", 0.0, 1.0, 0.06, step=0.001)
    nhr = st.slider("NHR", 0.0, 1.0, 0.06, step=0.001)
    spread1 = st.slider("Spread1", -7.0, -0.1, -4.0, step=0.1)
    PPE = st.slider("PPE", 0.0, 1.0, 0.3, step=0.01)

    features = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, jitter, shimmer, nhr, spread1, PPE]])
    st.write("Input shape:", features.shape)
    st.write("Scaler expects:", getattr(scaler, "n_features_in_", "unknown"))
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
        shap.plots.waterfall(shap_values[0], show=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Built by you | XGBoost + SHAP")


