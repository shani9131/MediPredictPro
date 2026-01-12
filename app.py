
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="MediPredict Pro", page_icon="🧑‍⚕️", layout="wide")  # Wide layout add kiya

st.title("🧑‍⚕️ MediPredict Pro")
st.markdown("### Multi-Disease Risk Prediction with Explainable AI")

# Sidebar ko force kar rahe hain
st.sidebar.title("Disease Selector")  # Sidebar title add kiya visibility ke liye
disease = st.sidebar.selectbox("Choose Disease", ("Diabetes", "Heart Disease", "Liver Cirrhosis Stage", "Parkinson's Disease"))

# Baaki code same rakh (load_model, if conditions etc.)
# (tu apna pura app code yahan paste kar dena, sirf upar wale 2 lines change kar)

st.markdown("---")
st.caption("Built by you | XGBoost + SHAP")
