"""
Placement Intelligence v2.0 — Career Readiness Estimator
Fixed layout: centered, constrained width, clean column grid
"""

import streamlit as st
import numpy as np
import joblib
import time
import time

# ── PAGE CONFIG — "centered" layout, not wide ──────────────────────────────
st.set_page_config(
    page_title="Placement Intelligence",
    page_icon="🎯",
    layout="centered",           # ← KEY FIX: centered = no stretching
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/main.css")

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("placement_model.pkl"), joblib.load("scaler.pkl"), False
    except FileNotFoundError:
        return None, None, True

model, scaler, demo_mode = load_model()

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Placement Intelligence · v2.0</div>
  <div class="hero-title">Know Your<br><span class="grad">Career Odds.</span></div>
  <div class="hero-sub">Enter your academic profile and get an ML-powered placement
  readiness estimate with actionable insights to strengthen your chances.</div>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown(
        '<div class="demo-tag">⚡ Demo mode — add <code>placement_model.pkl</code> + <code>scaler.pkl</code> for real predictions</div>',
        unsafe_allow_html=True,
    )

# ── SECTION 01: ACADEMICS ─────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">01 — Academic Profile</div>', unsafe_allow_html=True)

# Row 1: three equal columns
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    ssc_p    = st.number_input("10th Grade %",  0.0, 100.0, 75.0, 0.5,
                                help="10th standard final percentage")
with col2:
    hsc_p    = st.number_input("12th Grade %",  0.0, 100.0, 72.0, 0.5,
                                help="12th / Intermediate percentage")
with col3:
    degree_p = st.number_input("Degree %",      0.0, 100.0, 68.0, 0.5,
                                help="Undergraduate degree percentage")

# Row 2: PG narrow + aptitude slider wide
col4, col5 = st.columns([1, 2], gap="medium")
with col4:
    mba_p    = st.number_input("PG / MBA %",    0.0, 100.0,  0.0, 0.5,
                                help="Leave 0 if not applicable")
with col5:
    etest_p  = st.slider("Aptitude / E-Test Score",
                          0.0, 100.0, 65.0, 0.5,
                          help="Reasoning, aptitude, verbal — estimate if unsure")

# ── SECTION 02: PROFILE ───────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">02 — Profile Details</div>', unsafe_allow_html=True)

# Row 1: Gender | 10th Board | 12th Board
r1c1, r1c2, r1c3 = st.columns(3, gap="medium")
with r1c1:
    gender   = st.selectbox("Gender",     ["Female", "Male"])
with r1c2:
    ssc_b    = st.selectbox("10th Board", ["Central (CBSE/ICSE)", "State Board"])
with r1c3:
    hsc_b    = st.selectbox("12th Board", ["Central (CBSE/ICSE)", "State Board"])

# Row 2: 12th Stream | Degree Background | Work Experience
r2c1, r2c2, r2c3 = st.columns(3, gap="medium")
with r2c1:
    hsc_s    = st.selectbox("12th Stream",          ["Commerce", "Science", "Arts"])
with r2c2:
    degree_t = st.selectbox("Degree Background",
                             ["Business / Commerce", "Science / Technology", "Other"])
with r2c3:
    workex   = st.selectbox("Work Experience",      ["No", "Yes"],
                             help="Internship, part-time, or full-time")

# Row 3: Specialisation — full width
specialisation = st.selectbox(
    "Specialisation / Career Focus",
    ["Finance & Business Analytics", "Human Resources & Marketing"],
    help="Finance roles typically have higher placement rates",
)

st.markdown("<br>", unsafe_allow_html=True)

# ── ENCODE ───────────────────────────────────────────────────────────────────
gender_M          = 1 if gender == "Male" else 0
ssc_b_Others      = 1 if ssc_b == "State Board" else 0
hsc_b_Others      = 1 if hsc_b == "State Board" else 0
hsc_s_Commerce    = 1 if hsc_s == "Commerce" else 0
hsc_s_Science     = 1 if hsc_s == "Science" else 0
degree_t_Others   = 1 if degree_t == "Other" else 0
degree_t_SciTech  = 1 if "Science" in degree_t else 0
workex_Yes        = 1 if workex == "Yes" else 0
specialisation_HR = 1 if "Human Resources" in specialisation else 0

features = np.array([[
features = np.array([[
    ssc_p, hsc_p, degree_p, etest_p, mba_p,
    gender_M, ssc_b_Others, hsc_b_Others,
    gender_M, ssc_b_Others, hsc_b_Others,
    hsc_s_Commerce, hsc_s_Science,
    degree_t_Others, degree_t_SciTech,
    workex_Yes, specialisation_HR,
    workex_Yes, specialisation_HR,
]])

# ── CTA ──────────────────────────────────────────────────────────────────────
run = st.button("Analyse My Placement Readiness →")

# ── RESULTS ──────────────────────────────────────────────────────────────────
if run:
    if demo_mode:
        raw = (
            ssc_p    * 0.12 +
            hsc_p    * 0.12 +
            degree_p * 0.20 +
            etest_p  * 0.18 +
            (mba_p if mba_p > 0 else 50) * 0.08 +
            workex_Yes * 12 +
            (5 if not specialisation_HR else -3) +
            (3 if gender_M else 2)
        )
        prob = min(max(raw / 100, 0.05), 0.95)
    else:
        scaled = scaler.transform(features)
        prob   = min(float(model.predict_proba(scaled)[0][1]), 0.95)

    risk      = 1 - prob
    pct       = prob * 100
    cls       = "hi" if pct >= 75 else ("mi" if pct >= 50 else "lo")
    verdict   = "Strong Profile ↑" if pct >= 75 else ("Moderate Profile —" if pct >= 50 else "Needs Work ↓")
    vsub      = (
        "You're well-positioned for campus placements." if pct >= 75 else
        ("A few targeted improvements can make a big difference." if pct >= 50 else
         "Focus on the action plan below to improve significantly.")
    )

    st.divider()
    st.markdown('<div class="sec-lbl">03 — Your Results</div>', unsafe_allow_html=True)

    with st.spinner("Analysing your profile…"):
        time.sleep(0.45)

    # Score card
    st.markdown(f"""
    <div class="result-card">
      <div class="rc-label">Placement Probability</div>
      <div class="rc-score {cls}">{pct:.1f}<span style="font-size:34px;font-weight:400">%</span></div>
      <div class="rc-verdict">{verdict}</div>
      <div class="rc-sub">{vsub}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs — 4 equal columns
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Placement Chance",  f"{pct:.1f}%")
    k2.metric("Risk Level",        f"{risk*100:.1f}%")
    k3.metric("Aptitude Score",    f"{etest_p:.0f}/100")
    k4.metric("Degree Score",      f"{degree_p:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(prob))
    tier = "Strong" if pct >= 75 else ("Moderate" if pct >= 50 else "Low")
    st.caption(f"Overall readiness: **{pct:.1f}%** — {tier}")

    st.divider()

    # ── Strength breakdown ──
    st.markdown('<div class="sec-lbl">04 — Profile Strength</div>', unsafe_allow_html=True)

    def sbar(label, value, mx=100):
        pf = value / mx * 100
        c  = "#22C97A" if pf >= 75 else ("#F5A623" if pf >= 50 else "#F0454A")
        return (
            f'<div class="s-row">'
            f'<div class="s-lbl">{label}</div>'
            f'<div class="s-trk"><div class="s-fill" style="width:{pf:.1f}%;background:{c}"></div></div>'
            f'<div class="s-val">{value:.0f}</div></div>'
        )

    st.markdown(
        sbar("10th Grade", ssc_p) +
        sbar("12th Grade", hsc_p) +
        sbar("Degree %",   degree_p) +
        sbar("Aptitude",   etest_p) +
        (sbar("PG / MBA",  mba_p) if mba_p > 0 else ""),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Why this score ──
    st.markdown('<div class="sec-lbl">05 — Why This Score?</div>', unsafe_allow_html=True)

    insights = []
    insights = []
    if workex_Yes:
        insights.append(("✅", "Work Experience",
                          "Having internship experience is one of the strongest positive signals in placement data."))
    else:
        insights.append(("📋", "No Work Experience",
                          "Candidates without internship experience score lower — highest-impact area to fix."))
    if degree_p >= 75:
        insights.append(("🎓", "Strong Degree Score",
                          f"{degree_p:.0f}% is above average. Recruiters weight degree performance heavily."))
    elif degree_p >= 65:
        insights.append(("📘", "Average Degree Score",
                          f"{degree_p:.0f}% is adequate. Improving to 70%+ noticeably shifts your odds."))
    else:
        insights.append(("⚠️", "Below-Average Degree",
                          f"{degree_p:.0f}% is below the preferred threshold (65%+)."))
    if etest_p >= 70:
        insights.append(("🧠", "Strong Aptitude",
                          f"Score of {etest_p:.0f} is competitive — boosts screening and interview rounds."))
    else:
        insights.append(("📐", "Aptitude Gap",
                          f"Score of {etest_p:.0f} leaves room. Most companies screen at 65+."))
    if specialisation_HR:
        insights.append(("📉", "HR Specialisation",
                          "HR roles have a smaller hiring pool vs Finance/Analytics. Consider broadening."))
    else:
        insights.append(("📈", "Finance / Analytics Track",
                          "Finance and analytics roles see higher demand — a positive signal."))

    grid = '<div class="ig">'
    for ic, ti, tx in insights:
        grid += (f'<div class="ig-tile"><div class="ig-icon">{ic}</div>'
                 f'<div class="ig-txt"><strong>{ti}</strong>{tx}</div></div>')
    grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)

    st.divider()

    # ── Action plan ──
    st.markdown('<div class="sec-lbl">06 — Personalised Action Plan</div>', unsafe_allow_html=True)

    actions = []
    if not workex_Yes:
        actions.append(("#22C97A", "High Impact", "Get Internship Experience",
                         "Apply on Internshala, LinkedIn, or AngelList. Even a 2-month unpaid role is a strong signal."))
    if degree_p < 65:
        actions.append(("#22C97A", "High Impact", "Improve Academic Performance",
                         "Focus on upcoming semesters. A 5% improvement in degree score meaningfully shifts screening pass rates."))
    if etest_p < 70:
        actions.append(("#F5A623", "Medium Impact", "Sharpen Aptitude Skills",
                         "Practice on IndiaBix or PrepInsta. 30 min/day for 8 weeks can push you past the 70 threshold."))
    if specialisation_HR:
        actions.append(("#F5A623", "Medium Impact", "Broaden Your Skill Set",
                         "Add Excel, Power BI or basic Python. Opens Finance and Operations roles alongside HR tracks."))
    if ssc_p < 70 or hsc_p < 70:
        actions.append(("#4F8EF7", "Context", "Frame Academic Trajectory",
                         "Older scores are fixed — highlight upward progression on your CV if degree marks improved."))
    if not actions:
        actions.append(("#4F8EF7", "Maintain", "Keep Up the Strong Work",
                         "Focus on interview prep — mock GDs, case studies, HR rounds. Your profile is competitive."))
        actions.append(("#9B6CF5", "Growth", "Add Certifications",
                         "A CFA Level 1, Google Analytics, or AWS cert can differentiate you in final-round evaluations."))

    for i, (tc, tag, title, text) in enumerate(actions, 1):
        st.markdown(
            f'<div class="ai">'
            f'  <div class="ai-num" style="color:{tc};border:1px solid {tc}44;background:{tc}11">{i:02d}</div>'
            f'  <div class="ai-body"><strong>{title}'
            f'    <span class="ai-tag" style="color:{tc};border:1px solid {tc}33;background:{tc}0D">{tag}</span>'
            f'  </strong>{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("⚡ Estimate based on historical placement patterns. Not a guarantee of any outcome.")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <b>PLACEMENT INTELLIGENCE &middot</b> <b>ML-POWERED &middot</b><b> BUILT BY <b>VARA PRASAD K 🚀</b>
</div>
""", unsafe_allow_html=True)