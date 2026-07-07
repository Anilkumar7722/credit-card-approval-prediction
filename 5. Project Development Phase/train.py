"""
train.py
--------
End-to-end training pipeline for the Credit Card Approval Prediction project.

Steps:
 1. Load the dataset (dataset/credit_card_approval.csv)
 2. Clean + engineer features (utils/preprocessing.py)
 3. Balance classes with SMOTE (falls back to random oversampling if
    imbalanced-learn isn't installed)
 4. Train and cross-validate a bank of classical + gradient-boosted models
 5. Tune the top candidate with GridSearchCV
 6. Evaluate on a held-out test set (accuracy, precision, recall, F1, ROC-AUC)
 7. Save the best model + scaler + encoders + metrics/feature-importance
    artifacts that the Flask dashboard reads at runtime

Run:
    python train.py
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from utils.preprocessing import Preprocessor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset" / "credit_card_approval.csv"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python dataset/generate_dataset.py` first, "
            "or place a real dataset with a matching schema at that path."
        )
    df = pd.read_csv(DATASET_PATH)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes with SMOTE; falls back to simple random oversampling
    with added Gaussian jitter if imbalanced-learn isn't available."""
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("Balanced classes with SMOTE -> %s", np.bincount(y_res))
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed; using manual oversampling fallback.")
        rng = np.random.default_rng(RANDOM_STATE)
        classes, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()
        X_parts, y_parts = [X], [y]
        for cls, cnt in zip(classes, counts):
            if cnt < majority_count:
                idx = np.where(y == cls)[0]
                extra_idx = rng.choice(idx, size=majority_count - cnt, replace=True)
                noise = rng.normal(0, 0.01, size=(len(extra_idx), X.shape[1]))
                X_parts.append(X[extra_idx] + noise)
                y_parts.append(np.full(len(extra_idx), cls))
        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)
        perm = rng.permutation(len(y_res))
        logger.info("Balanced classes with oversampling -> %s", np.bincount(y_res))
        return X_res[perm], y_res[perm]


def get_candidate_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed - skipping (will still run via requirements.txt in production).")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1)
    except ImportError:
        logger.warning("lightgbm not installed - skipping (will still run via requirements.txt in production).")

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(iterations=300, verbose=False, random_state=RANDOM_STATE)
    except ImportError:
        logger.warning("catboost not installed - skipping (will still run via requirements.txt in production).")

    return models


def evaluate_model(model, X_test, y_test, include_curves: bool = False) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    result = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if include_curves:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        step = max(1, len(fpr) // 60)
        result["roc_curve"] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}
        step2 = max(1, len(prec_curve) // 60)
        result["pr_curve"] = {
            "precision": prec_curve[::step2].tolist(),
            "recall": rec_curve[::step2].tolist(),
        }
    return result


def tune_best_model(name: str, model, X_train, y_train):
    """Run a small, targeted GridSearchCV around the winning model family."""
    param_grids = {
        "RandomForest": {"n_estimators": [200, 300], "max_depth": [8, 12, None], "min_samples_split": [2, 5]},
        "ExtraTrees": {"n_estimators": [200, 300], "max_depth": [8, 12, None]},
        "GradientBoosting": {"n_estimators": [150, 250], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
        "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
        "XGBoost": {"n_estimators": [200, 300], "max_depth": [4, 6], "learning_rate": [0.03, 0.1]},
        "LightGBM": {"n_estimators": [200, 300], "num_leaves": [31, 63]},
        "DecisionTree": {"max_depth": [6, 10, None], "min_samples_split": [2, 5]},
    }
    grid = param_grids.get(name)
    if grid is None:
        logger.info("No tuning grid defined for %s; using default hyperparameters.", name)
        return model

    logger.info("Running GridSearchCV for %s ...", name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(model, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info("Best params for %s: %s", name, search.best_params_)
    return search.best_estimator_


def main():
    start = time.time()
    df = load_dataset()

    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    logger.info("Feature matrix shape after preprocessing: %s", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    models = get_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    leaderboard = []
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()
        try:
            cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring="roc_auc", n_jobs=-1)
            model.fit(X_train_bal, y_train_bal)
        except Exception as exc:  # keep the leaderboard resilient to a single bad model
            logger.error("Model %s failed to train: %s", name, exc)
            continue

        metrics = evaluate_model(model, X_test, y_test, include_curves=False)
        elapsed = round(time.time() - t0, 2)
        trained_models[name] = model
        leaderboard.append({
            "model": name,
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
            "train_time_sec": elapsed,
            **metrics,
        })
        logger.info(
            "%-18s | CV ROC-AUC: %.4f | Test ROC-AUC: %.4f | Test F1: %.4f | %.2fs",
            name, cv_scores.mean(), metrics["roc_auc"], metrics["f1_score"], elapsed,
        )

    leaderboard_sorted = sorted(leaderboard, key=lambda r: r["roc_auc"], reverse=True)
    best_name = leaderboard_sorted[0]["model"]
    best_model = trained_models[best_name]
    logger.info("Best model before tuning: %s (Test ROC-AUC=%.4f)", best_name, leaderboard_sorted[0]["roc_auc"])

    tuned_model = tune_best_model(best_name, best_model, X_train_bal, y_train_bal)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test, include_curves=True)
    logger.info("Tuned %s -> Test ROC-AUC=%.4f, F1=%.4f", best_name, tuned_metrics["roc_auc"], tuned_metrics["f1_score"])

    # Feature importance (best-effort; not all model types expose it)
    feature_importance = None
    if hasattr(tuned_model, "feature_importances_"):
        feature_importance = dict(zip(preprocessor.feature_names, tuned_model.feature_importances_.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True))
    elif hasattr(tuned_model, "coef_"):
        coefs = np.abs(tuned_model.coef_[0])
        feature_importance = dict(zip(preprocessor.feature_names, coefs.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True))

    # --- Persist all artifacts ---
    joblib.dump(tuned_model, MODEL_DIR / "model.pkl")
    joblib.dump(preprocessor.scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(preprocessor.label_encoders, MODEL_DIR / "encoder.pkl")
    joblib.dump(preprocessor.feature_names, MODEL_DIR / "feature_names.pkl")

    metrics_payload = {
        "best_model": best_name,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_rows": int(df.shape[0]),
        "test_metrics": tuned_metrics,
        "leaderboard": leaderboard_sorted,
        "feature_importance": feature_importance,
        "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Saved model + artifacts to %s", MODEL_DIR)
    logger.info("Total training time: %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
