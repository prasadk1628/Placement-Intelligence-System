import streamlit as st
import numpy as np
import joblib

# ===== PAGE SETUP =====
st.set_page_config(page_title="Placement Intelligence", layout="centered")

st.title("🎯 Placement Readiness Checker")
st.markdown("### Estimate your placement chances and get simple improvement tips")

# ===== LOAD MODEL =====
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== INFO BLOCK =====
st.info("""
This tool uses past student data to estimate placement chances.
It considers academics, skills (aptitude), work experience, and career focus.
""")

# ===== INPUT SECTION =====
st.subheader("📌 Tell us about you")

col1, col2 = st.columns(2)

with col1:
    ssc_p = st.number_input(
        "10th Grade (%)",
        0.0, 100.0,
        help="Your final marks in 10th standard"
    )
    hsc_p = st.number_input(
        "12th Grade (%)",
        0.0, 100.0,
        help="Your final marks in 12th / Intermediate"
    )
    degree_p = st.number_input(
        "College Percentage (%)",
        0.0, 100.0,
        help="Your current or final degree percentage"
    )

with col2:
    etest_p = st.number_input(
        "Aptitude Level (0-100)",
        0.0, 100.0,
        help="Your ability in reasoning, aptitude, and problem-solving (estimate if unsure)"
    )
    mba_p = st.number_input(
        "Postgraduate Percentage (%)",
        0.0, 100.0,
        help="Leave 0 if not applicable"
    )

# ===== SIMPLE LANGUAGE INPUTS =====
st.subheader("📊 Additional Details")

gender = st.selectbox("Gender", ["Female", "Male"])

ssc_b = st.selectbox(
    "10th Board Type",
    ["Central", "Others"],
    help="CBSE/ICSE = Central, State boards = Others"
)

hsc_b = st.selectbox(
    "12th Board Type",
    ["Central", "Others"]
)

hsc_s = st.selectbox(
    "12th Stream",
    ["Commerce", "Science", "Arts"]
)

degree_t = st.selectbox(
    "Degree Background",
    ["Business/Commerce", "Science/Technology", "Other"]
)

workex = st.selectbox(
    "Do you have internship or work experience?",
    ["No", "Yes"]
)

specialisation = st.selectbox(
    "Career Focus",
    ["Finance / Business", "Human Resources"]
)

# ===== ENCODING =====
gender_M = 1 if gender == "Male" else 0
ssc_b_Others = 1 if ssc_b == "Others" else 0
hsc_b_Others = 1 if hsc_b == "Others" else 0
hsc_s_Commerce = 1 if hsc_s == "Commerce" else 0
hsc_s_Science = 1 if hsc_s == "Science" else 0

degree_t_Others = 1 if degree_t == "Other" else 0
degree_t_SciTech = 1 if degree_t == "Science/Technology" else 0

workex_Yes = 1 if workex == "Yes" else 0
specialisation_MktHR = 1 if specialisation == "Human Resources" else 0

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
if st.button("🚀 Check My Placement Chances"):

    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]

    display_prob = min(prob, 0.95)
    risk = 1 - prob

    st.divider()

    # ===== RESULT =====
    st.subheader("📊 Your Placement Estimate")

    st.progress(float(display_prob))

    colA, colB = st.columns(2)
    with colA:
        st.metric("Chance of Getting Placed", f"{display_prob*100:.1f}%")
    with colB:
        st.metric("Risk Level", f"{risk*100:.1f}%")

    # ===== INTERPRETATION =====
    if prob > 0.85:
        st.success("🔥 You have strong chances of getting placed")
    elif prob > 0.65:
        st.warning("⚠️ You have moderate chances — some improvements needed")
    else:
        st.error("❌ Your chances are low — focus on improvement")

    st.caption("This is an estimate based on past student data, not a guarantee.")

    # ===== WHY RESULT =====
    st.subheader("🧠 Why this result?")

    reasons = []

    if workex_Yes:
        reasons.append("Having work experience improves your chances")
    if specialisation_MktHR:
        reasons.append("HR roles have fewer opportunities compared to finance roles")
    if degree_p < 65:
        reasons.append("Your college percentage is below average")
    if ssc_p < 70:
        reasons.append("Academic consistency plays an important role")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• Your profile looks balanced")

    # ===== IMPROVEMENT =====
    st.subheader("💡 How you can improve")

    feedback = []

    if degree_p < 65:
        feedback.append("Try to improve your college performance")

    if workex_Yes == 0:
        feedback.append("Gain internship or project experience")

    if specialisation_MktHR == 1:
        feedback.append("Build finance or technical skills for more opportunities")

    if ssc_p < 70:
        feedback.append("Maintain consistent academic performance")

    if not feedback:
        st.success("You're on the right track. Focus on interview preparation.")
    else:
        for f in feedback:
            st.write("•", f)

# ===== FOOTER =====
st.divider()
st.markdown("""
### About  
This tool helps students understand their placement readiness  
and gives simple guidance to improve their chances.

Built by Prasad 🚀
""")