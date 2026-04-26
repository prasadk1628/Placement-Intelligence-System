import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Placement Prediction System")

# ===== INPUTS =====

ssc_p = st.number_input("10th Percentage", 0.0, 100.0)
hsc_p = st.number_input("12th Percentage", 0.0, 100.0)
degree_p = st.number_input("Degree Percentage", 0.0, 100.0)
etest_p = st.number_input("Employability Test %", 0.0, 100.0)
mba_p = st.number_input("MBA Percentage", 0.0, 100.0)

gender = st.selectbox("Gender", ["Female", "Male"])
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])
degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Sci&Tech", "Others"])
workex = st.selectbox("Work Experience", ["No", "Yes"])
specialisation = st.selectbox("Specialisation", ["Mkt&Fin", "Mkt&HR"])

# ===== ENCODING =====

gender_M = 1 if gender == "Male" else 0

ssc_b_Others = 1 if ssc_b == "Others" else 0
hsc_b_Others = 1 if hsc_b == "Others" else 0

hsc_s_Commerce = 1 if hsc_s == "Commerce" else 0
hsc_s_Science = 1 if hsc_s == "Science" else 0

degree_t_Others = 1 if degree_t == "Others" else 0
degree_t_SciTech = 1 if degree_t == "Sci&Tech" else 0

workex_Yes = 1 if workex == "Yes" else 0
specialisation_MktHR = 1 if specialisation == "Mkt&HR" else 0

# ===== FEATURE ARRAY (ORDER MUST MATCH TRAINING) =====

features = np.array([[ 
    ssc_p, hsc_p, degree_p, etest_p, mba_p,
    gender_M,
    ssc_b_Others, hsc_b_Others,
    hsc_s_Commerce, hsc_s_Science,
    degree_t_Others, degree_t_SciTech,
    workex_Yes,
    specialisation_MktHR
]])

# ===== PREDICTION =====

if st.button("Predict"):

    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader(f"Placement Chance: {prob*100:.2f}%")

    if pred == 1:
        st.success("Likely to be Placed")
    else:
        st.error("Low Chance of Placement")

    # ===== FEEDBACK SYSTEM =====

    feedback = []

    if degree_p < 65:
        feedback.append("Improve your degree percentage.")

    if workex_Yes == 0:
        feedback.append("Gain internship or work experience.")

    if specialisation_MktHR == 1:
        feedback.append("Strengthen finance-related skills.")

    if ssc_p < 70:
        feedback.append("Improve academic consistency.")

    if not feedback:
        st.info("Profile looks strong. Focus on interview preparation.")
    else:
        st.info(" ".join(feedback))