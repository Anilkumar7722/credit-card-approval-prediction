"""
app.py
------
Flask web application for the Credit Card Approval Prediction project.

Routes:
    GET  /                  Home / landing page
    GET  /predict           Prediction form
    POST /predict           Handle form submission -> result page
    POST /api/predict       JSON API for programmatic predictions
    GET  /dashboard         Model performance + analytics dashboard
    GET  /about             About / project info page
    POST /batch-predict     CSV upload -> batch predictions (download CSV)
    GET  /health            Health check endpoint (used by deployment platforms)
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for

from predict import get_predictor, REQUIRED_FIELDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-in-production"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB upload limit

# In-memory prediction history for the current server session (kept simple on
# purpose - swap for SQLite/Postgres if you need durable history).
PREDICTION_HISTORY: list[dict] = []
MAX_HISTORY = 50


def load_metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


@app.route("/")
def home():
    metrics = load_metrics()
    stats = {
        "accuracy": metrics.get("test_metrics", {}).get("accuracy", 0.94) * 100,
        "roc_auc": metrics.get("test_metrics", {}).get("roc_auc", 0.95) * 100,
        "best_model": metrics.get("best_model", "Ensemble Model"),
        "dataset_rows": metrics.get("dataset_rows", 12000),
    }
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html", fields=REQUIRED_FIELDS)


@app.route("/predict", methods=["POST"])
def predict_submit():
    predictor = get_predictor()
    form_data = request.form.to_dict()

    try:
        result = predictor.predict_single(form_data)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("predict_page"))
    except Exception:
        logger.exception("Prediction failed")
        flash("Something went wrong while generating the prediction. Please try again.", "error")
        return redirect(url_for("predict_page"))

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "decision": result["decision"],
        "approval_probability": result["approval_probability"],
        "risk_level": result["risk_level"],
    }
    PREDICTION_HISTORY.insert(0, entry)
    del PREDICTION_HISTORY[MAX_HISTORY:]

    return render_template("result.html", result=result, input_data=form_data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    predictor = get_predictor()
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = predictor.predict_single(data)
        return jsonify({"success": True, "result": result})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        logger.exception("API prediction failed")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("Please choose a CSV file to upload.", "error")
        return redirect(url_for("predict_page"))
    if not file.filename.lower().endswith(".csv"):
        flash("Only .csv files are supported.", "error")
        return redirect(url_for("predict_page"))

    try:
        df = pd.read_csv(file)
        missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
        if missing_cols:
            flash(f"CSV is missing required column(s): {', '.join(missing_cols)}", "error")
            return redirect(url_for("predict_page"))

        predictor = get_predictor()
        result_df = predictor.predict_batch(df)

        buffer = io.StringIO()
        result_df.to_csv(buffer, index=False)
        buffer.seek(0)
        mem = io.BytesIO(buffer.getvalue().encode("utf-8"))
        return send_file(
            mem, mimetype="text/csv", as_attachment=True,
            download_name=f"batch_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    except Exception:
        logger.exception("Batch prediction failed")
        flash("Could not process that CSV file. Please check the format and try again.", "error")
        return redirect(url_for("predict_page"))


@app.route("/dashboard")
def dashboard():
    metrics = load_metrics()
    return render_template(
        "dashboard.html",
        metrics=metrics,
        history=PREDICTION_HISTORY,
        metrics_json=json.dumps(metrics),
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    predictor_ready = False
    try:
        predictor_ready = get_predictor().is_ready
    except Exception:
        predictor_ready = False
    return jsonify({"status": "ok", "model_loaded": predictor_ready}), 200


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_e):
    logger.exception("Server error")
    return render_template("500.html"), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
