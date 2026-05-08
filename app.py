"""
Placement Intelligence — Career Readiness Estimator
Premium Streamlit App · ML-Powered

Run:  streamlit run app.py
Deps: pip install streamlit numpy joblib
"""

import streamlit as st
import numpy as np
import joblib
import time

# ─────────────────────────────────────────────────────────────────────────────
# 0 · PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1 · CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #06080D;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(79,142,247,0.08) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.018) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.018) 40px);
}
.main .block-container {
    font-family: 'DM Sans', sans-serif;
    max-width: 980px;
    padding: 0 2rem 4rem;
}
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero { padding: 52px 0 36px; }
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 400; color: #4F8EF7;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 24px; height: 1px; background: #4F8EF7;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 5vw, 58px); font-weight: 800;
    color: #F0F2F8; line-height: 1.05;
    letter-spacing: -0.03em; margin-bottom: 14px;
}
.hero-title span {
    background: linear-gradient(135deg, #4F8EF7 0%, #9B6CF5 50%, #4F8EF7 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 4s linear infinite;
}
@keyframes shine { to { background-position: 200% center; } }
.hero-sub {
    font-size: 16px; font-weight: 300; color: #7A8299;
    max-width: 520px; line-height: 1.7;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500; color: #4F8EF7;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #1E2433, transparent);
}

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: #111420 !important; border: 1px solid #1E2433 !important;
    border-radius: 8px !important; color: #E8EAF0 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 14px !important;
}
div[data-baseweb="select"] span { color: #E8EAF0 !important; }
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important; color: #8A90A2 !important;
}
div[data-testid="stSlider"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: #8A90A2 !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2B4FD9 0%, #4F8EF7 100%) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-family: 'Syne', sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; padding: 14px 36px !important;
    box-shadow: 0 4px 24px rgba(79,142,247,0.3) !important;
    width: 100% !important; transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(79,142,247,0.45) !important;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: #0C0F18 !important; border: 1px solid #1A1F2E !important;
    border-radius: 12px !important; padding: 18px 20px !important;
}
div[data-testid="stMetricValue"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 26px !important; font-weight: 500 !important; color: #E8EAF0 !important;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 10px !important; color: #505568 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Progress */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #2B4FD9, #4F8EF7, #9B6CF5) !important;
    border-radius: 4px !important;
}
div[data-testid="stProgress"] > div > div {
    background: #1A1F2E !important; border-radius: 4px !important; height: 8px !important;
}

/* Divider */
hr { border-color: #1A1F2E !important; margin: 28px 0 !important; }

/* Result hero */
.result-hero {
    background: linear-gradient(135deg, #0D1526 0%, #0C0F18 100%);
    border: 1px solid #1A2847; border-radius: 16px;
    padding: 36px 36px 28px; text-align: center;
    position: relative; overflow: hidden;
}
.result-hero::before {
    content: ''; position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(79,142,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.score-ring-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase;
    color: #505568; margin-bottom: 8px;
}
.score-number {
    font-family: 'Syne', sans-serif; font-size: 80px; font-weight: 800;
    line-height: 1; letter-spacing: -0.04em; margin-bottom: 6px;
}
.score-number.high  { color: #22C97A; text-shadow: 0 0 40px rgba(34,201,122,0.3); }
.score-number.mid   { color: #F5A623; text-shadow: 0 0 40px rgba(245,166,35,0.3); }
.score-number.low   { color: #F0454A; text-shadow: 0 0 40px rgba(240,69,74,0.3); }
.score-verdict { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: #E8EAF0; margin-bottom: 6px; }
.score-sub { font-size: 13px; color: #505568; }

/* Insight grid */
.insight-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.insight-tile {
    background: #0C0F18; border: 1px solid #1A1F2E;
    border-radius: 10px; padding: 14px 16px;
    display: flex; align-items: flex-start; gap: 12px;
}
.insight-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
.insight-text { font-size: 13px; color: #9EA3B0; line-height: 1.55; }
.insight-text strong { color: #E8EAF0; font-weight: 500; display: block; margin-bottom: 2px; }

/* Strength bars */
.strength-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.strength-label { font-size: 12px; color: #7A8299; width: 130px; flex-shrink: 0; }
.strength-track { flex: 1; height: 6px; background: #1A1F2E; border-radius: 3px; overflow: hidden; }
.strength-fill { height: 100%; border-radius: 3px; }
.strength-val { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #505568; width: 36px; text-align: right; }

/* Action items */
.action-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 14px 16px; background: #0C0F18;
    border: 1px solid #1A1F2E; border-radius: 10px; margin-bottom: 8px;
}
.action-step {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
    border-radius: 6px; padding: 4px 8px; flex-shrink: 0;
    margin-top: 1px; min-width: 32px; text-align: center;
}
.action-body { font-size: 13px; color: #9EA3B0; line-height: 1.55; }
.action-body strong { color: #E8EAF0; font-weight: 500; display: block; margin-bottom: 2px; font-size: 14px; }

/* Info tag */
.info-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,142,247,0.08); border: 1px solid rgba(79,142,247,0.2);
    border-radius: 6px; padding: 6px 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #4F8EF7;
    margin-bottom: 28px;
}

/* Footer */
.footer {
    text-align: center; padding: 32px 0 12px;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #2E3348; letter-spacing: 0.08em;
}
.footer span { color: #4F8EF7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2 · LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        mdl = joblib.load("placement_model.pkl")
        scl = joblib.load("scaler.pkl")
        return mdl, scl, False
    except FileNotFoundError:
        return None, None, True


model, scaler, demo_mode = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# 3 · HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Placement Intelligence · v2.0</div>
  <div class="hero-title">Know Your<br><span>Career Odds.</span></div>
  <div class="hero-sub">
    Enter your academic profile and get an ML-powered placement readiness estimate
    with actionable insights to strengthen your chances.
  </div>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.markdown(
        '<div class="info-tag">⚡ Demo mode — place placement_model.pkl + scaler.pkl alongside app.py for real predictions</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4 · ACADEMICS INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Academic Profile</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    ssc_p    = st.number_input("10th Grade %",  0.0, 100.0, 75.0, 0.1, help="10th standard final percentage")
with c2:
    hsc_p    = st.number_input("12th Grade %",  0.0, 100.0, 72.0, 0.1, help="12th / Intermediate percentage")
with c3:
    degree_p = st.number_input("Degree %",      0.0, 100.0, 68.0, 0.1, help="Undergraduate degree percentage")

c4, c5 = st.columns([1, 2])
with c4:
    mba_p    = st.number_input("PG / MBA %",    0.0, 100.0,  0.0, 0.1, help="Leave 0 if not applicable")
with c5:
    etest_p  = st.slider("Aptitude / E-Test Score  (0 → 100)",
                          0.0, 100.0, 65.0, 0.5,
                          help="Reasoning, aptitude, verbal — estimate if unsure")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 · PROFILE DETAILS INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Profile Details</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    gender   = st.selectbox("Gender", ["Female", "Male"])
    ssc_b    = st.selectbox("10th Board", ["Central (CBSE/ICSE)", "State Board"])

with col_b:
    hsc_b    = st.selectbox("12th Board", ["Central (CBSE/ICSE)", "State Board"])
    hsc_s    = st.selectbox("12th Stream", ["Commerce", "Science", "Arts"])

with col_c:
    degree_t = st.selectbox("Degree Background",
                             ["Business / Commerce", "Science / Technology", "Other"])
    workex   = st.selectbox("Internship / Work Experience", ["No", "Yes"])

specialisation = st.selectbox(
    "Specialisation / Career Focus",
    ["Finance & Business Analytics", "Human Resources & Marketing"],
    help="Finance roles typically show higher placement rates in MBA programs",
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6 · ENCODING
# ─────────────────────────────────────────────────────────────────────────────
gender_M          = 1 if gender == "Male" else 0
ssc_b_Others      = 1 if ssc_b == "State Board" else 0
hsc_b_Others      = 1 if hsc_b == "State Board" else 0
hsc_s_Commerce    = 1 if hsc_s == "Commerce" else 0
hsc_s_Science     = 1 if hsc_s == "Science" else 0
degree_t_Others   = 1 if degree_t == "Other" else 0
degree_t_SciTech  = 1 if degree_t == "Science / Technology" else 0
workex_Yes        = 1 if workex == "Yes" else 0
specialisation_HR = 1 if "Human Resources" in specialisation else 0

features = np.array([[
    ssc_p, hsc_p, degree_p, etest_p, mba_p,
    gender_M, ssc_b_Others, hsc_b_Others,
    hsc_s_Commerce, hsc_s_Science,
    degree_t_Others, degree_t_SciTech,
    workex_Yes, specialisation_HR,
]])

# ─────────────────────────────────────────────────────────────────────────────
# 7 · CTA BUTTON
# ─────────────────────────────────────────────────────────────────────────────
predict_clicked = st.button("Analyse My Placement Readiness →")

# ─────────────────────────────────────────────────────────────────────────────
# 8 · RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:

    # Probability
    if demo_mode:
        raw = (
            ssc_p * 0.12 + hsc_p * 0.12 + degree_p * 0.20 +
            etest_p * 0.18 + (mba_p if mba_p > 0 else 50) * 0.08 +
            workex_Yes * 12 + (5 if not specialisation_HR else -3) +
            (3 if gender_M else 2)
        )
        prob = min(max(raw / 100, 0.05), 0.95)
    else:
        scaled = scaler.transform(features)
        prob   = min(float(model.predict_proba(scaled)[0][1]), 0.95)

    risk      = 1 - prob
    pct       = prob * 100
    score_cls = "high" if pct >= 75 else ("mid" if pct >= 50 else "low")
    verdict   = ("Strong Profile ↑" if pct >= 75
                 else ("Moderate Profile —" if pct >= 50 else "Needs Work ↓"))
    vsub      = ("You're well-positioned for campus placements." if pct >= 75
                 else ("A few targeted improvements can make a big difference." if pct >= 50
                       else "Focus on the action plan below to improve significantly."))

    st.divider()
    st.markdown('<div class="section-label">03 — Your Results</div>', unsafe_allow_html=True)

    with st.spinner("Analysing your profile…"):
        time.sleep(0.5)

    # Big score card
    st.markdown(f"""
    <div class="result-hero">
      <div class="score-ring-label">Placement Probability</div>
      <div class="score-number {score_cls}">{pct:.1f}<span style="font-size:36px;font-weight:400">%</span></div>
      <div class="score-verdict">{verdict}</div>
      <div class="score-sub">{vsub}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Placement Chance",  f"{pct:.1f}%")
    k2.metric("Risk Level",        f"{risk*100:.1f}%")
    k3.metric("Aptitude Score",    f"{etest_p:.0f}/100")
    k4.metric("Degree Score",      f"{degree_p:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(float(prob))
    tier_label = "Low" if pct < 50 else ("Moderate" if pct < 75 else "Strong")
    st.caption(f"Score: {pct:.1f}% · {tier_label} placement readiness")

    st.divider()

    # ── Profile Strength ──
    st.markdown('<div class="section-label">04 — Profile Strength Breakdown</div>',
                unsafe_allow_html=True)

    def sbar(label, value, max_val=100):
        pf = (value / max_val) * 100
        c  = "#22C97A" if pf >= 75 else ("#F5A623" if pf >= 50 else "#F0454A")
        return (
            f'<div class="strength-row">'
            f'  <div class="strength-label">{label}</div>'
            f'  <div class="strength-track">'
            f'    <div class="strength-fill" style="width:{pf:.1f}%;background:{c}"></div>'
            f'  </div>'
            f'  <div class="strength-val">{value:.0f}</div>'
            f'</div>'
        )

    bars = (sbar("10th Grade", ssc_p) +
            sbar("12th Grade", hsc_p) +
            sbar("Degree %",   degree_p) +
            sbar("Aptitude",   etest_p) +
            (sbar("PG / MBA",  mba_p) if mba_p > 0 else ""))
    st.markdown(bars, unsafe_allow_html=True)

    st.divider()

    # ── Why this score ──
    st.markdown('<div class="section-label">05 — Why This Score?</div>', unsafe_allow_html=True)

    insights = []
    if workex_Yes:
        insights.append(("✅", "Work Experience",
                          "Having internship/work experience is one of the strongest positive signals in placement data."))
    else:
        insights.append(("📋", "No Work Experience",
                          "Candidates without internship experience tend to score lower — high-impact area to address."))
    if degree_p >= 75:
        insights.append(("🎓", "Strong Degree Score",
                          f"{degree_p:.0f}% is above average — recruiters consistently weigh degree performance heavily."))
    elif degree_p >= 65:
        insights.append(("📘", "Average Degree Score",
                          f"{degree_p:.0f}% is adequate. Improving to 70%+ would noticeably shift your odds."))
    else:
        insights.append(("⚠️", "Below-Average Degree",
                          f"{degree_p:.0f}% is below the typical threshold recruiters prefer (65%+)."))
    if etest_p >= 70:
        insights.append(("🧠", "Strong Aptitude",
                          f"Your score of {etest_p:.0f} is competitive — boosts both screening and interview rounds."))
    else:
        insights.append(("📐", "Aptitude Gap",
                          f"A score of {etest_p:.0f} leaves room. Most companies screen at 65+ aptitude."))
    if specialisation_HR:
        insights.append(("📉", "HR Specialisation",
                          "HR roles have a smaller hiring pool vs Finance/Analytics. Consider broadening skills."))
    else:
        insights.append(("📈", "Finance/Analytics Track",
                          "Finance and analytics roles see higher placement demand — a positive signal."))

    grid_html = '<div class="insight-grid">'
    for icon, title, text in insights:
        grid_html += (
            f'<div class="insight-tile">'
            f'  <div class="insight-icon">{icon}</div>'
            f'  <div class="insight-text"><strong>{title}</strong>{text}</div>'
            f'</div>'
        )
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    st.divider()

    # ── Action Plan ──
    st.markdown('<div class="section-label">06 — Personalised Action Plan</div>',
                unsafe_allow_html=True)

    actions = []
    if not workex_Yes:
        actions.append(("High Impact", "#22C97A",
                         "Get Internship Experience",
                         "Apply to 2–3 month internships on Internshala, LinkedIn, or AngelList. "
                         "Even unpaid roles count — the signal to recruiters is significant."))
    if degree_p < 65:
        actions.append(("High Impact", "#22C97A",
                         "Improve Academic Performance",
                         "Focus on upcoming semesters. A 5% improvement in degree score "
                         "meaningfully increases screening pass rates."))
    if etest_p < 70:
        actions.append(("Medium Impact", "#F5A623",
                         "Sharpen Aptitude Skills",
                         "Practice on IndiaBix, PrepInsta, or Testbook. 30 min/day for 8 weeks "
                         "can push your score past the 70 threshold."))
    if specialisation_HR:
        actions.append(("Medium Impact", "#F5A623",
                         "Broaden Skill Set",
                         "Supplement HR with Excel, Power BI, or basic Python. "
                         "Opens Finance/Operations roles alongside HR tracks."))
    if ssc_p < 70 or hsc_p < 70:
        actions.append(("Context", "#4F8EF7",
                         "Academic Consistency Matters",
                         "While older scores are fixed, highlight upward trajectory "
                         "on your CV if degree marks improved."))
    if not actions:
        actions.append(("Maintain", "#4F8EF7",
                         "You're on the Right Track",
                         "Focus on interview prep — mock GDs, case studies, HR round practice. "
                         "Your profile is competitive; execution is what's left."))
        actions.append(("Growth", "#9B6CF5",
                         "Certifications Add Edge",
                         "A CFA Level 1 or Google/AWS data certification can differentiate "
                         "you in final-round evaluations."))

    for i, (tag, tag_color, title, text) in enumerate(actions, 1):
        st.markdown(
            f'<div class="action-item">'
            f'  <div class="action-step" style="color:{tag_color};'
            f'       background:rgba(0,0,0,0.4);border:1px solid {tag_color}44">{i:02d}</div>'
            f'  <div class="action-body">'
            f'    <strong>{title}'
            f'      <span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
            f'             color:{tag_color};background:rgba(0,0,0,0.4);'
            f'             border:1px solid {tag_color}33;border-radius:4px;'
            f'             padding:1px 6px;margin-left:6px;font-weight:400;'
            f'             vertical-align:middle">{tag}</span>'
            f'    </strong>'
            f'    {text}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("⚡ Estimate based on historical placement data patterns. Not a guarantee of any outcome.")

# ─────────────────────────────────────────────────────────────────────────────
# 9 · FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  PLACEMENT INTELLIGENCE &middot; ML-POWERED CAREER TOOL &middot; BUILT BY <span>PRASAD 🚀</span>
</div>
""", unsafe_allow_html=True)
