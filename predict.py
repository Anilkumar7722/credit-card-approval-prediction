"""
predict.py
----------
Loads the trained model + preprocessing artifacts and exposes a clean
`predict_single()` / `predict_batch()` API used by app.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import Preprocessor

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

REQUIRED_FIELDS = [
    "Age", "Gender", "Income", "Education", "MaritalStatus", "EmploymentStatus",
    "CreditScore", "LoanAmount", "Debt", "PreviousDefault", "ResidenceType",
    "Dependents", "BankBalance", "ExistingLoans", "MonthlyExpenses",
    "YearsEmployed", "CreditHistoryYears",
]


class CreditApprovalPredictor:
    """Thin, reusable wrapper around the trained model + preprocessor."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self._load()

    def _load(self) -> None:
        try:
            self.model = joblib.load(self.model_dir / "model.pkl")
            scaler = joblib.load(self.model_dir / "scaler.pkl")
            encoders = joblib.load(self.model_dir / "encoder.pkl")
            feature_names = joblib.load(self.model_dir / "feature_names.pkl")

            self.preprocessor = Preprocessor(
                label_encoders=encoders, scaler=scaler, feature_names=feature_names, is_fitted=True
            )
            logger.info("Model + preprocessing artifacts loaded from %s", self.model_dir)
        except FileNotFoundError as exc:
            logger.error("Model artifacts missing: %s. Run `python train.py` first.", exc)
            raise

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def _validate(self, data: dict) -> dict:
        missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in ("", None)]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        cleaned = dict(data)
        numeric_fields = [
            "Age", "Income", "CreditScore", "LoanAmount", "Debt", "Dependents",
            "BankBalance", "ExistingLoans", "MonthlyExpenses", "YearsEmployed",
            "CreditHistoryYears", "PreviousDefault",
        ]
        for f in numeric_fields:
            try:
                cleaned[f] = float(cleaned[f])
            except (TypeError, ValueError):
                raise ValueError(f"Field '{f}' must be numeric.")

        if not (18 <= cleaned["Age"] <= 100):
            raise ValueError("Age must be between 18 and 100.")
        if not (300 <= cleaned["CreditScore"] <= 850):
            raise ValueError("Credit score must be between 300 and 850.")
        if cleaned["Income"] < 0 or cleaned["LoanAmount"] < 0:
            raise ValueError("Income and loan amount cannot be negative.")

        return cleaned

    def predict_single(self, data: dict) -> dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Predictor not ready - model artifacts failed to load.")

        cleaned = self._validate(data)
        df = pd.DataFrame([cleaned])
        X = self.preprocessor.transform(df)

        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0] if hasattr(self.model, "predict_proba") else [1 - pred, pred]
        approval_probability = float(proba[1])

        risk_score = round((1 - approval_probability) * 100, 1)
        confidence = round(float(max(proba)) * 100, 1)

        return {
            "decision": "Approved" if pred == 1 else "Rejected",
            "approved": bool(pred),
            "approval_probability": round(approval_probability * 100, 1),
            "rejection_probability": round((1 - approval_probability) * 100, 1),
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
            "recommendation": self._recommendation(pred, cleaned, approval_probability),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready:
            raise RuntimeError("Predictor not ready - model artifacts failed to load.")

        results = []
        for _, row in df.iterrows():
            try:
                result = self.predict_single(row.to_dict())
            except ValueError as exc:
                result = {"decision": "Error", "error": str(exc)}
            results.append(result)

        out = df.copy()
        out["Prediction"] = [r.get("decision") for r in results]
        out["ApprovalProbability"] = [r.get("approval_probability") for r in results]
        out["RiskLevel"] = [r.get("risk_level") for r in results]
        return out

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        if risk_score < 25:
            return "Low"
        if risk_score < 55:
            return "Medium"
        return "High"

    @staticmethod
    def _recommendation(pred: int, data: dict, approval_probability: float) -> str:
        if pred == 1:
            if approval_probability > 0.85:
                return "Strong approval profile. Eligible for premium credit limits."
            return "Approved. Consider a standard credit limit with periodic review."

        reasons = []
        if data.get("CreditScore", 0) < 600:
            reasons.append("credit score below the recommended threshold")
        if data.get("PreviousDefault", 0) == 1:
            reasons.append("a previous default on record")
        if data.get("Debt", 0) > data.get("Income", 1) * 12 * 0.5:
            reasons.append("a high debt-to-income ratio")
        if not reasons:
            reasons.append("the overall risk profile from the submitted details")

        return f"Rejected due to {', '.join(reasons)}. Improving these factors may help on reapplication."


_predictor: CreditApprovalPredictor | None = None


def get_predictor() -> CreditApprovalPredictor:
    """Lazy singleton so the model is loaded once per process."""
    global _predictor
    if _predictor is None:
        _predictor = CreditApprovalPredictor()
    return _predictor


if __name__ == "__main__":
    sample = {
        "Age": 34, "Gender": "Male", "Income": 5200, "Education": "Bachelors",
        "MaritalStatus": "Married", "EmploymentStatus": "Employed", "CreditScore": 715,
        "LoanAmount": 18000, "Debt": 4200, "PreviousDefault": 0, "ResidenceType": "Owned",
        "Dependents": 1, "BankBalance": 22000, "ExistingLoans": 1, "MonthlyExpenses": 2100,
        "YearsEmployed": 6.5, "CreditHistoryYears": 9,
    }
    predictor = get_predictor()
    print(predictor.predict_single(sample))
