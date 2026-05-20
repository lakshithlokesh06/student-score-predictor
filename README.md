# 🎓 Student Score Predictor

An end-to-end Machine Learning web app that predicts student academic performance based on study habits and lifestyle factors — with Explainable AI and personalized recommendations to explain and improve results.

🔗 **Live Demo:** https://academic-score-ai.streamlit.app

---

## 📌 Project Overview

This project was built as part of a personal data science learning journey before starting an MSc in Data Science. It covers the complete Machine Learning lifecycle — from data generation and exploratory data analysis to model training, explainability, deployment, and recommendation generation.

Unlike traditional prediction apps, this project not only predicts student performance but also explains why a prediction occurred and provides personalized suggestions for improvement.

---

## 🖼️ App Preview

Add screenshots here:

### Landing Page
(Add screenshot)

### Prediction Dashboard
(Add screenshot)

### SHAP Feature Contribution
(Add screenshot)

### Recommendation Engine
(Add screenshot)

---

## 🖥️ App Features

- 🎯 Predicts student Performance Index (0–100)
- 📊 Assigns Grade (A / B / C / D)
- 🔍 SHAP-based Explainable AI
- 📝 Plain-English explanation of prediction
- 🎛️ Interactive sliders for real-time input
- 🧠 Personalized Recommendation Engine
- 📈 Suggests improvement strategies based on weak factors
- 📜 Stores prediction history
- 🌙 Dark-themed interactive UI

---

## 🏗️ System Architecture

```text
User Input
     ↓
Data Preprocessing
     ↓
Trained ML Model
     ↓
Score Prediction
     ↓
SHAP Explainability
     ↓
Feature Contribution Analysis
     ↓
Recommendation Engine
     ↓
Streamlit Dashboard
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 📊 Dataset

Synthetic dataset generated using realistic academic patterns and relationships.

Features used:

| Feature | Description |
|---|---|
| Hours Studied | Daily study hours (1–9) |
| Previous Scores | Previous exam score (40–100) |
| Extracurricular Activities | Yes / No |
| Sleep Hours | Hours slept per day (4–9) |
| Sample Question Papers Practiced | Practice count (0–9) |
| Performance Index | Target variable (0–100) |

Dataset values were created with realistic dependencies among academic and lifestyle factors.

---

## 🤖 Models Trained

| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression | 4.91 | 0.88 ✅ |
| Gradient Boosting | 5.00 | 0.87 |
| XGBoost | 5.35 | 0.86 |
| Random Forest | 5.44 | 0.85 |

Linear Regression was selected as the final model because it achieved the highest R² score while maintaining simpler interpretability and explainability.

---

## 🔍 Explainability

Each prediction includes:

- SHAP Feature Contribution Chart
- Plain-English explanation
- Positive and negative contribution analysis

Example:

> "Hours Studied increased your predicted score by +15 points"

---

## 🧠 Recommendation Engine

The recommendation engine uses feature contribution values to identify weak areas and generate personalized suggestions.

Examples:

- Increase study hours
- Improve sleep schedule
- Practice more sample papers
- Maintain consistency in strong areas

Recommendations dynamically change based on:

- Feature contribution values
- Grade level
- Student performance patterns

If no negative factors are found, the app displays a positive feedback card indicating strong overall balance.

---

## 📜 Prediction History

The app stores previous prediction results so users can track inputs and compare trends over time.

History includes:

- Timestamp
- Hours Studied
- Previous Scores
- Sleep Hours
- Practice Papers
- Extracurricular Activity
- Predicted Score

---

## 📁 Project Structure

```text
student-score-predictor/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── app/
│   └── app.py
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── plots/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Run Locally

```bash
# Clone repo
git clone https://github.com/lakshithlokesh06/student-score-predictor.git

# Enter folder
cd student-score-predictor

# Create environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/app.py
```

---

## 🧠 What I Learned

- Built complete ML workflow from dataset creation to deployment
- Performed EDA and feature analysis
- Compared multiple Machine Learning models
- Implemented XGBoost model evaluation
- Applied Explainable AI using SHAP
- Built interactive Streamlit applications
- Integrated recommendation systems
- Used GitHub workflow and deployment pipelines

---

## 🚀 Future Improvements

- Add login and authentication
- Store predictions in database
- Connect real-world student datasets
- Add score improvement simulator
- Export prediction reports
- Add advanced dashboard analytics

---
