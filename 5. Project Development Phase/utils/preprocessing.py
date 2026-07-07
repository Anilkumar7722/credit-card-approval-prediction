"""
utils/preprocessing.py
-----------------------
Reusable data cleaning, feature engineering, and preprocessing utilities
shared by train.py, predict.py, and app.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "Age", "Income", "CreditScore", "LoanAmount", "Debt", "Dependents",
    "BankBalance", "ExistingLoans", "MonthlyExpenses", "YearsEmployed",
    "CreditHistoryYears",
]

CATEGORICAL_COLUMNS = [
    "Gender", "Education", "MaritalStatus", "EmploymentStatus", "ResidenceType",
]

BINARY_COLUMNS = ["PreviousDefault"]

TARGET_COLUMN = "Approved"

ENGINEERED_COLUMNS = [
    "DebtToIncomeRatio", "LoanToIncomeRatio", "DisposableIncome",
    "SavingsRatio", "CreditUtilizationRisk", "EmploymentStability",
]

FINAL_FEATURE_ORDER = NUMERIC_COLUMNS + BINARY_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and outliers."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", before - len(df))

    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Outlier clipping using the IQR method (keeps rows, caps extreme values)
    for col in ["Income", "LoanAmount", "Debt", "BankBalance", "MonthlyExpenses"]:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q1, upper=q3)

    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve model signal."""
    df = df.copy()
    eps = 1e-6

    df["DebtToIncomeRatio"] = df["Debt"] / (df["Income"] * 12 + eps)
    df["LoanToIncomeRatio"] = df["LoanAmount"] / (df["Income"] * 12 + eps)
    df["DisposableIncome"] = df["Income"] - df["MonthlyExpenses"]
    df["SavingsRatio"] = df["BankBalance"] / (df["Income"] * 12 + eps)
    df["CreditUtilizationRisk"] = (df["ExistingLoans"] + df["PreviousDefault"] * 2) / (df["CreditHistoryYears"] + 1)
    df["EmploymentStability"] = df["YearsEmployed"] / (df["Age"] - 17).clip(lower=1)

    for col in ENGINEERED_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


@dataclass
class Preprocessor:
    """Fits/stores encoders + scaler so preprocessing is 100% reproducible
    between training and inference."""

    label_encoders: dict = field(default_factory=dict)
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: list = field(default_factory=list)
    is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = clean_data(df)
        df = engineer_features(df)

        y = df[TARGET_COLUMN].astype(int).values

        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        X = df[FINAL_FEATURE_ORDER].copy()
        self.feature_names = FINAL_FEATURE_ORDER
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")

        df = df.copy()
        # clean_data() assumes an Approved column may be absent at inference time
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        df = engineer_features(df)

        for col in CATEGORICAL_COLUMNS:
            le = self.label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )
            # unseen categories -> map to the most frequent known class
            if (df[col] == -1).any():
                fallback = le.transform([le.classes_[0]])[0]
                df.loc[df[col] == -1, col] = fallback

        X = df[self.feature_names].copy()
        return self.scaler.transform(X)
