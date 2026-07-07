"""
generate_dataset.py
--------------------
Generates a realistic, synthetic Credit Card Approval dataset.

NOTE ON DATA SOURCE:
Public datasets such as the UCI "Credit Approval" dataset and the Kaggle
"Credit Card Approval Prediction" dataset (by rikdifos) were used as the
structural reference for the feature set and value ranges below. Because
this environment has no internet access to download those files directly,
this script generates a synthetic dataset that follows the same schema,
realistic value distributions, and the same underlying approval logic
patterns (income, credit score, DTI, employment stability, defaults, etc.)
found in those real-world datasets.

To use a real dataset instead:
1. Download `application_record.csv` + `credit_record.csv` from
   https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
   (or the UCI Credit Approval dataset), place them in this `dataset/`
   folder, and adapt the column mapping in `train.py` -> `load_dataset()`.
2. Everything downstream (cleaning, feature engineering, training,
   evaluation, the Flask app) works unchanged as long as the final schema
   matches the columns produced below.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 12000

def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_samples)
    gender = rng.choice(["Male", "Female"], n_samples, p=[0.52, 0.48])
    education = rng.choice(
        ["High School", "Bachelors", "Masters", "PhD", "Other"],
        n_samples, p=[0.30, 0.40, 0.20, 0.05, 0.05]
    )
    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"],
        n_samples, p=[0.35, 0.45, 0.15, 0.05]
    )
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"],
        n_samples, p=[0.55, 0.20, 0.08, 0.12, 0.05]
    )
    residence_type = rng.choice(
        ["Owned", "Rented", "With Parents", "Mortgaged"],
        n_samples, p=[0.30, 0.35, 0.15, 0.20]
    )
    dependents = rng.poisson(1.1, n_samples).clip(0, 6)

    years_employed = np.where(
        np.isin(employment_status, ["Unemployed", "Student"]),
        0,
        rng.gamma(2.2, 2.8, n_samples).clip(0, 40)
    ).round(1)

    # Monetary values are generated directly in INR (₹). Typical monthly
    # income range: roughly ₹15,000 - ₹5,00,000 depending on education/role.
    base_income = rng.lognormal(mean=10.5, sigma=0.55, size=n_samples)
    edu_multiplier = pd.Series(education).map(
        {"High School": 0.85, "Bachelors": 1.15, "Masters": 1.45, "PhD": 1.7, "Other": 0.9}
    ).values
    emp_multiplier = pd.Series(employment_status).map(
        {"Employed": 1.1, "Self-Employed": 1.0, "Unemployed": 0.25, "Retired": 0.7, "Student": 0.35}
    ).values
    INR_SCALE = 83  # approx USD->INR conversion used only to set realistic magnitudes
    monthly_income = (base_income * edu_multiplier * emp_multiplier / 12 * INR_SCALE).round(2)
    monthly_income = np.clip(monthly_income, 15000, 5000000)

    bank_balance = (monthly_income * rng.uniform(0.5, 6, n_samples) + rng.normal(0, 2000 * INR_SCALE, n_samples)).round(2)
    bank_balance = np.clip(bank_balance, 0, None)

    monthly_expenses = (monthly_income * rng.uniform(0.25, 0.75, n_samples)).round(2)

    credit_history_years = np.clip(
        (age - 21) * rng.uniform(0.1, 0.9, n_samples), 0, 45
    ).round(1)

    previous_default = rng.choice([0, 1], n_samples, p=[0.85, 0.15])

    credit_score = (
        650
        + credit_history_years * 3.2
        + (monthly_income / 100) * 0.35
        - previous_default * 140
        + rng.normal(0, 45, n_samples)
    )
    credit_score = np.clip(credit_score, 300, 850).round(0)

    existing_loans = rng.poisson(0.8, n_samples).clip(0, 6)
    debt = (existing_loans * rng.uniform(1500, 9000, n_samples) * INR_SCALE
            + previous_default * rng.uniform(0, 5000, n_samples) * INR_SCALE).round(2)

    loan_amount = (monthly_income * rng.uniform(2, 14, n_samples)).round(2)

    dti = np.where(monthly_income > 0, (debt + monthly_expenses) / (monthly_income * 12 + 1), 1.0)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Income": monthly_income,
        "Education": education,
        "MaritalStatus": marital_status,
        "EmploymentStatus": employment_status,
        "CreditScore": credit_score.astype(int),
        "LoanAmount": loan_amount,
        "Debt": debt,
        "PreviousDefault": previous_default,
        "ResidenceType": residence_type,
        "Dependents": dependents,
        "BankBalance": bank_balance,
        "ExistingLoans": existing_loans,
        "MonthlyExpenses": monthly_expenses,
        "YearsEmployed": years_employed,
        "CreditHistoryYears": credit_history_years,
    })

    # --- Underlying "true" approval logic (used only to label the data) ---
    score = (
        0.0028 * (df["CreditScore"] - 300)
        + 0.00004 * df["Income"]
        + 0.06 * df["CreditHistoryYears"]
        + 0.05 * df["YearsEmployed"]
        - 1.6 * dti
        - 1.4 * df["PreviousDefault"]
        - 0.00002 * df["LoanAmount"]
        + 0.00003 * df["BankBalance"]
        - 0.12 * df["ExistingLoans"]
        - 0.05 * df["Dependents"]
        + np.where(df["EmploymentStatus"] == "Unemployed", -1.1, 0)
        + np.where(df["EmploymentStatus"] == "Student", -0.4, 0)
        + np.where(df["ResidenceType"] == "Owned", 0.25, 0)
    )
    score += rng.normal(0, 0.55, n_samples)  # noise so it isn't trivially separable
    prob_approved = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
    approved = (prob_approved > rng.uniform(0.35, 0.65, n_samples)).astype(int)

    df["Approved"] = approved

    # Inject a small amount of realistic missingness and a few duplicate rows
    for col in ["Income", "CreditScore", "BankBalance", "YearsEmployed"]:
        mask = rng.random(n_samples) < 0.015
        df.loc[mask, col] = np.nan

    dup_rows = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "credit_card_approval.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
    print(df["Approved"].value_counts(normalize=True))
