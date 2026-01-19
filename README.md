# MediPredict Pro

Multi-disease risk prediction web app using XGBoost models with SHAP for explainability.

**Diseases Covered**
- Diabetes Prediction (binary classification)
- Heart Disease Prediction (binary)
- Liver Cirrhosis Stage Prediction (multiclass: stages 1-4)
- Parkinson's Disease Prediction (binary)

**Key Features**
- Interactive Streamlit web interface
- User inputs via sliders/dropdowns
- Risk probability scores
- SHAP waterfall plots to explain predictions (which features influenced the result)

**Technologies**
- Python, XGBoost, SHAP (explainable AI)
- Streamlit (web app)
- Pandas, NumPy, Scikit-learn, SMOTE for preprocessing & imbalance
- Trained on public Kaggle/UCI datasets

**Live Demo**
https://medipredictpro-erpv bldhaskyysev3bqr ud.streamlit.app

**How to Run Locally**
1. Clone repo: `git clone https://github.com/shani9131/MediPredictPro.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

**Screenshots**
(Add 3-4 screenshots yahan – sidebar, inputs form, prediction result, SHAP plot)

**Note**
Models trained in Colab and saved in `models/` folder. Datasets in `Project_datasets/` (download from Kaggle/UCI if needed).
