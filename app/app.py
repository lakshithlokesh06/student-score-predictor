import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model and scaler
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced"
]

# Page config
st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d6efd);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .metric-card h1 { font-size: 2.5rem; margin: 0; }
    .metric-card p { font-size: 1rem; margin: 0; opacity: 0.8; }
    .result-box {
        background: linear-gradient(135deg, #064e3b, #10b981);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .result-box h1 { font-size: 3.5rem; margin: 0; }
    .result-box p { font-size: 1.2rem; margin: 0; opacity: 0.9; }
    .stButton > button {
        background: linear-gradient(135deg, #0d6efd, #6610f2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6610f2, #0d6efd);
    }
    .tip-box {
        background: #1c1f2e;
        border-left: 4px solid #0d6efd;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        color: #ccc;
        font-size: 0.9rem;
    }
    .landing-container {
        text-align: center;
        padding: 80px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state to track page
if "page" not in st.session_state:
    st.session_state.page = "home"

# ─── HOME PAGE ───
if st.session_state.page == "home":
    st.markdown("""
    <div class='landing-container'>
        <h1 style='font-size: 3.5rem; color: white;'>🎓 Student Score Predictor</h1>
        <p style='font-size: 1.3rem; color: #aaa; margin-top: 10px;'>
            Predict your academic performance using Machine Learning & Explainable AI
        </p>
        <br><br>
        <p style='color: #777; font-size: 1rem;'>
            📊 ML-Powered &nbsp;|&nbsp; 🔍 Explainable AI &nbsp;|&nbsp; 🎯 Instant Results
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Check Score Predictor"):
            st.session_state.page = "predictor"
            st.rerun()

# ─── PREDICTOR PAGE ───
elif st.session_state.page == "predictor":

    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h1 style='text-align:center; color:white;'>🎓 Student Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa; font-size:1.1rem;'>Predict academic performance using Machine Learning & Explainable AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Student Details")
        hours = st.slider("📚 Daily Study Hours", 1, 9, 4)
        prev_scores = st.slider("📊 Previous Scores", 40, 100, 70)
        extra = st.selectbox("🏃 Extracurricular Activities", ["Yes", "No"])
        sleep = st.slider("😴 Sleep Hours", 4, 9, 7)
        papers = st.slider("📝 Question Papers Practiced", 0, 9, 4)

        extra_val = 1 if extra == "Yes" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Performance")

        st.markdown("""
        <div class='tip-box'>
        💡 <b>Tip:</b> Adjust the sliders to see how each factor impacts the predicted score!
        </div>
        """, unsafe_allow_html=True)

    with col2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <p>Study Hours</p>
                <h1>{hours}h</h1>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <p>Previous Score</p>
                <h1>{prev_scores}</h1>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <p>Sleep Hours</p>
                <h1>{sleep}h</h1>
            </div>""", unsafe_allow_html=True)

        if predict_btn:
            input_raw = np.array([[hours, prev_scores, extra_val, sleep, papers]])
            input_scaled = scaler.transform(input_raw)
            prediction = model.predict(input_scaled)[0]
            prediction = np.clip(prediction, 0, 100)

            if prediction >= 80:
                grade = "A 🌟"
                msg = "Outstanding Performance!"
            elif prediction >= 60:
                grade = "B 👍"
                msg = "Good Performance!"
            elif prediction >= 40:
                grade = "C 📈"
                msg = "Average - Keep Pushing!"
            else:
                grade = "D 💪"
                msg = "Needs Improvement!"

            st.markdown(f"""
            <div class='result-box'>
                <p>Predicted Performance Index</p>
                <h1>{prediction:.1f} / 100</h1>
                <p>Grade: {grade} &nbsp;|&nbsp; {msg}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔍 Why this prediction?")
            coefficients = model.coef_
            input_contribution = input_scaled[0] * coefficients

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1c1f2e')
            ax.set_facecolor('#1c1f2e')

            colors = ['#10b981' if v > 0 else '#ef4444' for v in input_contribution]
            bars = ax.barh(feature_names, input_contribution, color=colors)
            ax.set_xlabel("Impact on Score", color='white')
            ax.set_title("Feature Contributions to Prediction", color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for bar, val in zip(bars, input_contribution):
                ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}', va='center', color='white', fontsize=10)

            st.pyplot(fig)

            st.markdown("### 📝 Explanation in Simple Words")
            explanation_items = sorted(
                zip(feature_names, input_contribution),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for fname, fval in explanation_items:
                if fval > 0:
                    st.markdown(f"✅ **{fname}** boosted your score by **+{fval:.1f}** points")
                else:
                    st.markdown(f"❌ **{fname}** reduced your score by **{fval:.1f}** points")

        else:
            st.markdown("""
            <div style='text-align:center; padding: 80px 0; color: #555;'>
                <h2>👈 Fill in the details</h2>
                <p>and click Predict to see your results!</p>
            </div>
            """, unsafe_allow_html=True)