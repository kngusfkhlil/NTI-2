import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime

# === Page configuration ===
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

# === Load feature names and scaler ===
@st.cache_data
def load_artifacts():
    try:
        feature_columns = joblib.load("model_features.pkl")
        scaler = joblib.load("scaler.pkl")
        return feature_columns, scaler
    except Exception as e:
        st.error(f"❌ Failed to load feature/scaler: {e}")
        st.stop()

feature_columns, scaler = load_artifacts()

# === Load all models dynamically from models folder ===
@st.cache_data
def load_models():
    model_files = glob.glob("models/*_model.pkl")
    models = {}
    for path in model_files:
        model_name = os.path.basename(path).replace("_model.pkl", "").replace("_", " ").title()
        try:
            models[model_name] = joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Couldn't load {model_name}: {e}")
    if not models:
        st.error("❌ No models loaded. Please run `train_models.py` first.")
        st.stop()
    return models

models = load_models()

# === Prediction function ===
def predict_heart_disease(input_dict, model, feature_columns, scaler):
    input_df = pd.DataFrame([input_dict])
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    
    # Align features with training set
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    
    # Debug feature alignment
    st.write("🔍 Input Feature Names:", input_df.columns.tolist())
    st.write("🔍 Expected Feature Names:", feature_columns)
    
    # Scale
    scaled_input = scaler.transform(input_df)
    st.write("🔍 Scaled Input Values:", scaled_input[0])
    
    # Predict
    pred = model.predict(scaled_input)[0]
    return pred

# === Page: Prediction ===
def prediction_page():
    st.title("💓 Heart Disease Risk Predictor")
    st.markdown("Fill in the details to check heart disease risk.")

    # Model selection
    selected_model_name = st.selectbox("🔽 Choose Model", list(models.keys()))
    selected_model = models[selected_model_name]
    st.write(f"🔍 Loaded Model: {selected_model_name}")

    # Input form
    with st.form("input_form"):
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        submitted = st.form_submit_button("🔍 Predict")

    # === Handle form submission ===
    if submitted:
        input_dict = {
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'sex': 1 if sex == "Male" else 0,
            'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
            'fbs': 1 if fbs == "Yes" else 0,
            'restecg': ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
            'exang': 1 if exang == "Yes" else 0,
            'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
            'ca': int(ca),
            'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
        }

        try:
            pred = predict_heart_disease(input_dict, selected_model, feature_columns, scaler)
            st.subheader(f"Prediction using {selected_model_name}")
            st.write("🔍 Raw Prediction:", pred)
            if pred == 1:
                st.error("⚠️ High risk of heart disease.")
            else:
                st.success("✅ Low risk of heart disease.")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

    # === Test with high-risk profile ===
    if st.button("Test High-Risk Profile"):
        input_dict = {
            'age': 65,
            'trestbps': 160,
            'chol': 300,
            'thalach': 120,
            'oldpeak': 2.5,
            'sex': 1,
            'cp': 3,  # Asymptomatic
            'fbs': 1,
            'restecg': 1,  # ST-T Abnormality
            'exang': 1,
            'slope': 2,  # Downsloping
            'ca': 3,
            'thal': 2  # Reversible Defect
        }

        try:
            pred = predict_heart_disease(input_dict, selected_model, feature_columns, scaler)
            st.subheader(f"High-Risk Test Prediction using {selected_model_name}")
            st.write("🔍 Raw Prediction:", pred)
            if pred == 1:
                st.error("⚠️ High risk of heart disease.")
            else:
                st.success("✅ Low risk of heart disease.")
        except Exception as e:
            st.error(f"❌ High-risk test error: {e}")

# === Page: Dashboard ===
def dashboard_page():
    st.title("📊 Heart Disease Dashboard")
    st.markdown(f"Current Date and Time: {datetime(2025, 7, 31, 11, 21).strftime('%I:%M %p EEST, %B %d, %Y')} (Thursday)")
    st.markdown("Visualizations generated by the training script:")

    visualization_files = glob.glob("plots/*.png")
    if visualization_files:
        for viz_file in visualization_files:
            st.image(viz_file, caption=os.path.basename(viz_file), use_container_width=True)
    else:
        st.warning("⚠️ No visualizations found. Run `train_models.py` to generate plots.")

    if st.button("Go to Prediction"):
        st.switch_page("app.py")

# === Sidebar navigation ===
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Prediction"])

if page == "Dashboard":
    dashboard_page()
elif page == "Prediction":
    prediction_page()