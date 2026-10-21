import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# Same fixed category lists as the notebook, so the codes match exactly
sex_enc = LabelEncoder().fit(["Male", "Female"])
cp_enc = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
exang_enc = LabelEncoder().fit(["Yes", "No"])

model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below. A Support Vector Classifier trained on the "
         "UCI Heart Disease (Cleveland) dataset predicts whether heart disease is likely.")
st.markdown("*Model accuracy on held-out test data: **70%** (303-row UCI Cleveland dataset).*")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    cp = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"], index=3)
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])

if st.button("Predict", type="primary", use_container_width=True):
    x = pd.DataFrame({
        "Age": [age],
        "Sex": sex_enc.transform([sex]),
        "ChestPain": cp_enc.transform([cp]),
        "ExerciseAngina": exang_enc.transform([exang]),
    })
    st.divider()
    if model.predict(x)[0] == 1:
        st.error("### Heart disease likely")
    else:
        st.success("### No heart disease likely")
    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(x, hide_index=True)

st.caption("Built for educational purposes from the 303-row UCI Heart Disease (Cleveland) dataset, "
              "using only 4 input attributes (Age, Sex, Chest Pain Type, Exercise-Induced Angina). "
              "Not medical advice.")
