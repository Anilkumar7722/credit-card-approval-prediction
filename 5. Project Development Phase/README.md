<p align="center">
  <img src="assets/banner/banner.png" alt="CreditIQ Banner" width="100%">
</p>

# CreditIQ — Credit Card Approval Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-D9B36C" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-2FD4B8" alt="Status">
  <img src="https://img.shields.io/badge/Deploy-Render%20%7C%20Docker%20%7C%20Railway-5B7FF0" alt="Deploy">
</p>

<p align="center">
  An end-to-end machine learning web application that predicts whether a credit card
  application should be <strong>approved</strong> or <strong>rejected</strong> — trained model,
  Flask API, and a premium fintech-grade UI, all in one repository.
</p>

---

## Overview

CreditIQ takes 17 applicant fields (income, credit score, employment, existing debt, etc.),
runs them through a full cleaning → feature-engineering → scaling pipeline, and scores them
with the best of 9–12 compared machine learning models. It returns an approval decision,
a calibrated probability, a risk band, and a plain-language recommendation — through a web
form, a JSON API, or batch CSV upload.

**This project intentionally trades a few "everything and the kitchen sink" bonus
features (login/auth, chatbot, voice input, PDF export, multi-language UI) for a
core pipeline and UI that are fully built, tested, and working end-to-end.** See
[Future Improvements](#future-improvements) for what's next.

## Architecture

```
Applicant data (form / JSON / CSV)
        │
        ▼
 ┌─────────────────┐     clean_data() + engineer_features()
 │  Preprocessor    │ ──  label encoders + StandardScaler
 └─────────────────┘     (utils/preprocessing.py, fit once in train.py)
        │
        ▼
 ┌─────────────────┐
 │  Trained model   │  best of: LogisticRegression, DecisionTree, RandomForest,
 │  (model.pkl)     │  ExtraTrees, GradientBoosting, AdaBoost, XGBoost, LightGBM,
 └─────────────────┘  CatBoost, SVM, NaiveBayes, KNN — selected by test ROC-AUC
        │
        ▼
 ┌─────────────────┐
 │  predict.py      │  probability → risk score → risk level → recommendation
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │  Flask (app.py)  │  web form · JSON API · batch CSV · dashboard · health check
 └─────────────────┘
```

## Features

- **Multi-model comparison** — up to 12 classifiers, 5-fold stratified cross-validation, GridSearchCV tuning on the winner
- **SMOTE class balancing** (falls back to a manual oversampler if `imbalanced-learn` isn't installed)
- **Feature engineering** — debt-to-income, loan-to-income, disposable income, savings ratio, credit utilization risk, employment stability
- **Explainable predictions** — approval probability, confidence, risk score/level, and a human-readable recommendation
- **JSON API** (`POST /api/predict`) alongside the web UI
- **Batch scoring** — upload a CSV, download every row scored
- **Live model dashboard** — leaderboard, confusion matrix, ROC-AUC comparison, feature importance, class balance, session prediction history
- **Premium UI** — glassmorphism, dark/light theme, animated credit-card hero, Chart.js visualizations, full mobile responsiveness
- **Deployment-ready** — Dockerfile, Render blueprint, Procfile, GitHub Actions CI

## Screenshots

> Run the app locally and drop your own screenshots into `screenshots/` — suggested shots:
> `screenshots/home.png`, `screenshots/predict.png`, `screenshots/result.png`, `screenshots/dashboard.png`.

## Installation

```bash
git clone https://github.com/<your-username>/CreditCardApprovalPrediction.git
cd CreditCardApprovalPrediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 1. Generate the dataset (synthetic — see "About the dataset" below)
python dataset/generate_dataset.py

# 2. Train the model (writes model/model.pkl, scaler.pkl, encoder.pkl, metrics.json)
python train.py

# 3. Run the app
python app.py
# → http://127.0.0.1:5000
```

## Usage

### Web UI
Visit `/predict`, fill in the applicant form, and submit for an instant decision with
a probability gauge, risk meter, and recommendation. Use the CSV uploader on the same
page for batch scoring.

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 34, "Gender": "Male", "Income": 5200, "Education": "Bachelors",
    "MaritalStatus": "Married", "EmploymentStatus": "Employed", "CreditScore": 715,
    "LoanAmount": 18000, "Debt": 4200, "PreviousDefault": 0, "ResidenceType": "Owned",
    "Dependents": 1, "BankBalance": 22000, "ExistingLoans": 1, "MonthlyExpenses": 2100,
    "YearsEmployed": 6.5, "CreditHistoryYears": 9
  }'
```

```json
{
  "success": true,
  "result": {
    "decision": "Approved",
    "approved": true,
    "approval_probability": 92.4,
    "rejection_probability": 7.6,
    "confidence": 92.4,
    "risk_score": 7.6,
    "risk_level": "Low",
    "recommendation": "Strong approval profile. Eligible for premium credit limits."
  }
}
```

### Batch CSV
`POST /batch-predict` with a `multipart/form-data` field named `csv_file` containing
all 17 required columns (see `predict.REQUIRED_FIELDS`). Returns the same CSV with
`Prediction`, `ApprovalProbability`, and `RiskLevel` columns appended.

## Folder structure

```
CreditCardApprovalPrediction/
├── app.py                  # Flask application (routes, API, batch upload)
├── train.py                # Full model training + comparison + tuning pipeline
├── predict.py               # Inference wrapper (singleton predictor)
├── requirements.txt
├── Dockerfile
├── render.yaml
├── Procfile
├── .gitignore
├── LICENSE
├── model/                  # model.pkl, scaler.pkl, encoder.pkl, metrics.json
├── static/
│   ├── css/style.css       # Design token system + component styles
│   ├── js/script.js        # Theme toggle, nav, reveal animations, counters
│   └── images/
├── templates/
│   ├── base.html, index.html, predict.html, result.html
│   ├── dashboard.html, about.html, 404.html, 500.html
├── notebook/
│   ├── EDA.ipynb           # Exploratory analysis, correlations, distributions
│   └── Training.ipynb      # Cell-by-cell walkthrough of train.py's logic
├── dataset/
│   └── generate_dataset.py # Synthetic dataset generator
├── utils/
│   └── preprocessing.py    # Cleaning, feature engineering, Preprocessor class
├── screenshots/
└── .github/workflows/ci.yml
```

## About the dataset

This environment had no internet access to download a dataset from Kaggle/UCI directly,
so `dataset/generate_dataset.py` **generates a synthetic dataset (~12,000 rows)** that
mirrors the schema, value ranges, and underlying approval logic (income, credit score,
DTI, employment stability, previous defaults, etc.) of real-world credit approval
datasets such as the UCI *Credit Approval* dataset and the Kaggle
*rikdifos/credit-card-approval-prediction* dataset.

To swap in a real dataset:
1. Download the real CSV(s) and place them in `dataset/`.
2. Map their columns to the schema documented in `utils/preprocessing.py` (`NUMERIC_COLUMNS`,
   `CATEGORICAL_COLUMNS`, `BINARY_COLUMNS`, `TARGET_COLUMN`).
3. Run `python train.py` — everything downstream (features, model, app) works unchanged.

## Model performance

Metrics are written to `model/metrics.json` after every training run and rendered live
on the `/dashboard` page. On the synthetic dataset, the best model (selected by test
ROC-AUC after tuning) typically reaches **~89–94% ROC-AUC** and **~80–94% accuracy**,
depending on which optional boosting libraries (XGBoost/LightGBM/CatBoost) are installed
at training time.

## Deployment

**Docker**
```bash
docker build -t creditiq .
docker run -p 5000:5000 creditiq
```

**Render** — connect the repo; `render.yaml` defines the build/start commands and health check.

**Railway / any Procfile-based platform** — uses `Procfile` (`gunicorn ... app:app`) automatically.

## Future improvements

Deliberately scoped out of this pass to keep the core pipeline solid:
- User accounts + persistent (SQLite/Postgres) prediction history
- PDF report export (current report download is a plain-text summary)
- A conversational assistant / chatbot layer over the prediction API
- Voice input and multi-language UI
- A/B testing of model versions in production

## License

MIT — see [LICENSE](LICENSE).

## Author

Built by **Anilkumar** as a portfolio / learning project. Not affiliated with any real
credit bureau or lender — see the [About](templates/about.html) page in-app for full
limitations.

# 👨‍💻 Project Team

| Name | Role |
|------|------|
| **Gorantla Anilkumar** | Team Lead • Machine Learning • GitHub Management • Deployment |
| **Karapakula Nandiswar** | Team Member |
| **Varun Kumar Reddy Chintha** | Team Member |
| **Sumanth Boddu** | Team Member |
| **Dhanush Tirumanyam** | Team Member |


# 🏫 Academic Information

**College**

Sri Venkateswara College of Engineering

**Department**

Computer Science & Engineering (AI & ML)

**Year**

IV Year

---

# 🎓 Internship Details

**Organization**

SmartBridge

**Internship**

Artificial Intelligence & Machine Learning Internship

**Project**

Credit Card Approval Prediction System

---

## ⭐ If you like this project, don't forget to star the repository!