import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

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

feature_tip_bank = {
    "Hours Studied": {
        "focus": "Study consistency",
        "tip": "Increase focused study time with a simple daily plan. Break sessions into short blocks, revise one concept at a time, and protect distraction-free hours."
    },
    "Previous Scores": {
        "focus": "Core concept recovery",
        "tip": "Your earlier scores suggest some concepts may still be weak. Revisit past mistakes, identify repeated topics, and rebuild those fundamentals before moving ahead."
    },
    "Extracurricular Activities": {
        "focus": "Time balance",
        "tip": "Try balancing activities with study time more intentionally. Keep extracurriculars, but anchor them around your study schedule so academic work stays consistent."
    },
    "Sleep Hours": {
        "focus": "Sleep routine",
        "tip": "A better sleep schedule can improve concentration and memory. Aim for a steady bedtime and enough rest before heavy study sessions or tests."
    },
    "Sample Question Papers Practiced": {
        "focus": "Exam practice",
        "tip": "Practice more sample papers under timed conditions. This builds speed, reduces exam stress, and reveals the exact topics where you lose marks most often."
    }
}

grade_recommendation_intro = {
    "A": "You are performing strongly. These suggestions will help you protect your momentum and sharpen the few areas pulling your score down.",
    "B": "You already have a solid base. These targeted improvements can help you convert a good result into an excellent one.",
    "C": "You are within reach of a much better score. Focus on the weakest drivers first and improve them one step at a time.",
    "D": "This score needs immediate attention, but it is still fixable. Start with the biggest negative factors below and work on them consistently this week."
}

grade_recommendation_action = {
    "A": "Maintain your habits while making small upgrades here for even stronger performance.",
    "B": "A focused push in this area can create a noticeable jump in your overall score.",
    "C": "Improving this area first should give you one of the fastest gains.",
    "D": "Treat this as a priority area and act on it right away for the next study cycle."
}

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
    .tip-box {
        background: #1c1f2e;
        border-left: 4px solid #0d6efd;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        color: #ccc;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .recommendation-card {
        background: linear-gradient(135deg, #161b26, #1f2937);
        border: 1px solid #2d3748;
        border-left: 4px solid #0d6efd;
        border-radius: 16px;
        padding: 18px 20px;
        margin: 14px 0;
        color: #e5e7eb;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }
    .recommendation-card h4 {
        margin: 0 0 8px 0;
        color: white;
        font-size: 1.05rem;
    }
    .recommendation-card p {
        margin: 0;
        color: #cbd5e1;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .recommendation-badge {
        display: inline-block;
        margin-bottom: 10px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(13, 110, 253, 0.18);
        color: #93c5fd;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# ─── HOME PAGE ───
if st.session_state.page == "home":
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
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

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("<h1 style='text-align:center; color:white;'>🎓 Student Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa;'>Predict academic performance using Machine Learning & Explainable AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Student Details")
        hours = st.slider("📚 Daily Study Hours", 1, 9, 1)
        prev_scores = st.slider("📊 Previous Scores", 40, 100, 40)
        extra = st.selectbox("🏃 Extracurricular Activities", ["Select...", "Yes", "No"])
        sleep = st.slider("😴 Sleep Hours", 4, 9, 4)
        papers = st.slider("📝 Question Papers Practiced", 0, 9, 0)

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
            if extra == "Select...":
                st.warning("⚠️ Please select Extracurricular Activities before predicting!")
            else:
                input_raw = np.array([[hours, prev_scores, extra_val, sleep, papers]])
                input_scaled = scaler.transform(input_raw)
                prediction = model.predict(input_scaled)[0]
                prediction = np.clip(prediction, 0, 100)

                # Grade logic
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

                # Save prediction history
                history_file = "prediction_history.csv"
                if not os.path.exists(history_file):
                    with open(history_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Timestamp", "Hours Studied", "Previous Scores",
                                        "Extracurricular", "Sleep Hours", "Papers Practiced",
                                        "Predicted Score", "Grade"])
                with open(history_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    hours, prev_scores, extra, sleep, papers,
                                    round(prediction, 1), grade.split()[0]])

                # Result box
                st.markdown(f"""
                <div class='result-box'>
                    <p>Predicted Performance Index</p>
                    <h1>{prediction:.1f} / 100</h1>
                    <p>Grade: {grade} &nbsp;|&nbsp; {msg}</p>
                </div>
                """, unsafe_allow_html=True)

                # Feature chart
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

                # Text explanation
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

                recommendation_container = st.container()
                with recommendation_container:
                    st.markdown("### 🎯 Personalized Recommendations")
                    grade_key = grade.split()[0]
                    negative_features = sorted(
                        [(fname, fval) for fname, fval in zip(feature_names, input_contribution) if fval < 0],
                        key=lambda x: x[1]
                    )
                    has_negative_contribution = len(negative_features) > 0

                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <div class='recommendation-badge'>Grade {grade_key} Action Plan</div>
                        <h4>What to focus on next</h4>
                        <p>{grade_recommendation_intro[grade_key]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if has_negative_contribution:
                        for fname, fval in negative_features[:3]:
                            tip_content = feature_tip_bank[fname]
                            st.markdown(f"""
                            <div class='recommendation-card'>
                                <div class='recommendation-badge'>{tip_content["focus"]}</div>
                                <h4>{fname} is lowering your predicted score</h4>
                                <p>
                                    This factor reduced your score by <b>{abs(fval):.1f}</b> points.
                                    {tip_content["tip"]} {grade_recommendation_action[grade_key]}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='recommendation-card'>
                            <div class='recommendation-badge'>Strong overall balance</div>
                            <h4>No major negative feature impact detected</h4>
                            <p>
                                Your current inputs are not showing any strong negative pull on the prediction.
                                Keep your routine stable, continue practicing consistently, and review performance weekly to stay on track.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Show history
                with st.expander("View Prediction History"):
                    if os.path.exists(history_file):
                        history_df = __import__('pandas').read_csv(history_file)
                        st.dataframe(history_df, use_container_width=True)
                    
                    else:
                        st.info("No prediction history available yet.")
        else:
            st.markdown("""
            <div style='text-align:center; padding: 80px 0; color: #555;'>
                <h2>👈 Fill in the details</h2>
                <p>and click Predict to see your results!</p>
            </div>
            """, unsafe_allow_html=True)
