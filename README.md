# 🎓 Student Score Predictor

An end-to-end Machine Learning web app that predicts student academic performance based on study habits and lifestyle factors — with Explainable AI to show *why* a score was predicted.

🔗 **Live Demo:** [academic-score-ai.streamlit.app](https://academic-score-ai.streamlit.app)

---

## 📌 Project Overview

This project was built as part of a personal data science learning journey before starting an MSc in Data Science. It covers the full ML pipeline — from data generation and EDA to model training, explainability, and deployment.

---

## 🖥️ App Features

- 🎯 Predicts a student's Performance Index (0–100)
- 📊 Assigns a Grade (A / B / C / D) with feedback
- 🔍 Shows **why** the prediction was made using feature contribution chart
- 📝 Plain English explanation of each feature's impact
- 🎛️ Interactive sliders to adjust student details in real time

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

---

## 📊 Dataset

Synthetically generated dataset with realistic relationships between:

| Feature | Description |
|---|---|
| Hours Studied | Daily study hours (1–9) |
| Previous Scores | Score in last exam (40–100) |
| Extracurricular Activities | Yes / No |
| Sleep Hours | Hours of sleep per night (4–9) |
| Sample Question Papers Practiced | Number of papers practiced (0–9) |
| Performance Index | Target variable (0–100) |

---

## 🤖 Models Trained

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 4.91 | 0.88 ✅ |
| Gradient Boosting | 5.00 | 0.87 |
| Random Forest | 5.44 | 0.85 |

**Linear Regression** was selected as the best model with R² of 0.88.

---

## 🔍 Explainability

Each prediction comes with:
- **Feature Contribution Chart** — shows how much each feature pushed the score up or down
- **Plain English Explanation** — e.g. "Hours Studied boosted your score by +15 points"

---

## 📁 Project Structure
````
student-score-predictor/
│
├── data/
│   └── raw/                  # Dataset
│
├── notebooks/
│   └── 01_eda.ipynb          # EDA + Model Training
│
├── app/
│   └── app.py                # Streamlit app
│
├── models/
│   └── model.pkl             # Trained model
│   └── scaler.pkl            # Feature scaler
│
├── plots/                    # Saved visualizations
├── requirements.txt
└── README.md
````

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/lakshithlokesh06/student-score-predictor.git
cd student-score-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 🧠 What I Learned

- Full ML pipeline from data → model → deployment
- Feature engineering and EDA techniques
- Model comparison and selection
- Explainable AI using SHAP
- Building and deploying Streamlit apps

---
