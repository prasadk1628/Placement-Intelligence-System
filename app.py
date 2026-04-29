import streamlit as st
import numpy as np
import joblib

# ===== PAGE SETUP =====
st.set_page_config(page_title="Placement Intelligence", layout="centered")

st.title("🎯 Placement Intelligence System")
st.markdown("### Know your placement chances & how to improve")

# ===== LOAD MODEL =====
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== INPUT SECTION =====
st.subheader("📌 Enter Your Details")

# Use columns for better UI
col1, col2 = st.columns(2)

with col1:
    ssc_p = st.number_input("10th %", 0.0, 100.0)
    hsc_p = st.number_input("12th %", 0.0, 100.0)
    degree_p = st.number_input("Degree %", 0.0, 100.0)

with col2:
    etest_p = st.number_input("Employability Test %", 0.0, 100.0)
    mba_p = st.number_input("MBA %", 0.0, 100.0)

# Categorical inputs
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

# ===== FEATURE ARRAY =====
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
if st.button("🚀 Predict Placement Chances"):

    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.divider()

    # ===== RESULT SECTION =====
    st.subheader("📊 Your Result")

    st.progress(float(prob))
    st.metric("Placement Probability", f"{prob*100:.1f}%")

    if prob > 0.75:
        st.success("🔥 Strong chances of placement")
    elif prob > 0.5:
        st.warning("⚠️ Moderate chances — improvement needed")
    else:
        st.error("❌ Low chances — take action")

    # ===== WHY THIS RESULT =====
    st.subheader("🧠 Why this result?")

    reasons = []
    if workex_Yes:
        reasons.append("Work experience improves your chances")
    if specialisation_MktHR:
        reasons.append("HR specialization has lower placement rate than Finance")
    if degree_p < 65:
        reasons.append("Low degree percentage affects placement chances")
    if ssc_p < 70:
        reasons.append("Academic consistency is important")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• Your profile is well-balanced")

    # ===== FEEDBACK SECTION =====
    st.subheader("💡 How to Improve")

    feedback = []

    if degree_p < 65:
        feedback.append("Improve your degree percentage")

    if workex_Yes == 0:
        feedback.append("Gain internship or work experience")

    if specialisation_MktHR == 1:
        feedback.append("Strengthen finance-related skills")

    if ssc_p < 70:
        feedback.append("Maintain consistent academic performance")

    if not feedback:
        st.success("Your profile looks strong. Focus on interview preparation.")
    else:
        for f in feedback:
            st.write("•", f)

# ===== FOOTER =====
st.divider()
st.markdown("""
### About  
This system analyzes student academic and professional profiles  
to predict placement probability and provide actionable insights.  

Built by Prasad 🚀
""")
