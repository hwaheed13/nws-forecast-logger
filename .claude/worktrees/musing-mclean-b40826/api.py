# api.py
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

from model_config import FEATURE_COLS, ACCU_NWS_FALLBACK, derive_bucket_probabilities

# Load models once at startup
with open("temp_model.pkl", "rb") as f:
    TEMP_MODEL = pickle.load(f)
with open("bucket_model.pkl", "rb") as f:
    BUCKET_INFO = pickle.load(f)

# Extract residual_std for Gaussian bucket derivation
if isinstance(BUCKET_INFO, dict) and "residual_std" in BUCKET_INFO:
    RESIDUAL_STD = BUCKET_INFO["residual_std"]
else:
    RESIDUAL_STD = 2.0  # fallback for old-style classifier


def prepare_features(raw):
    m = int(raw.get("month", 1))

    # Compute day-of-year cyclical features from target_date if available
    target_date = raw.get("target_date", "")
    if target_date:
        try:
            doy = datetime.strptime(target_date, "%Y-%m-%d").timetuple().tm_yday
        except ValueError:
            doy = datetime.now().timetuple().tm_yday
    else:
        doy = datetime.now().timetuple().tm_yday

    row = {
        # NWS forecast statistics
        "nws_first": float(raw.get("nws_first", np.nan)),
        "nws_last": float(raw.get("nws_last", np.nan)),
        "nws_max": float(raw.get("nws_max", np.nan)),
        "nws_min": float(raw.get("nws_min", np.nan)),
        "nws_mean": float(raw.get("nws_mean", np.nan)),
        "nws_spread": float(raw.get("nws_spread", 0)),
        "nws_std": float(raw.get("nws_std", 0)),
        "nws_trend": float(raw.get("nws_trend", 0)),
        "nws_count": int(raw.get("nws_count", 1)),
        "forecast_velocity": float(raw.get("forecast_velocity", 0)),
        "forecast_acceleration": float(raw.get("forecast_acceleration", 0)),

        # AccuWeather forecast statistics
        "accu_first": float(raw["accu_first"]) if raw.get("accu_first") is not None else np.nan,
        "accu_last": float(raw["accu_last"]) if raw.get("accu_last") is not None else np.nan,
        "accu_max": float(raw["accu_max"]) if raw.get("accu_max") is not None else np.nan,
        "accu_min": float(raw["accu_min"]) if raw.get("accu_min") is not None else np.nan,
        "accu_mean": float(raw["accu_mean"]) if raw.get("accu_mean") is not None else np.nan,
        "accu_spread": float(raw.get("accu_spread", 0)),
        "accu_std": float(raw.get("accu_std", 0)),
        "accu_trend": float(raw.get("accu_trend", 0)),
        "accu_count": int(raw.get("accu_count", 0)),

        # Cross-source features
        "nws_accu_spread": float(raw.get("nws_accu_spread", 0)),
        "nws_accu_mean_diff": float(raw.get("nws_accu_mean_diff", 0)),

        # Temporal features
        "day_of_year_sin": np.sin(2 * np.pi * doy / 365),
        "day_of_year_cos": np.cos(2 * np.pi * doy / 365),
        "month": m,
        "is_summer": int(m in [6, 7, 8]),
        "is_winter": int(m in [12, 1, 2]),

        # Rolling bias (must be provided by caller or default to 0)
        "rolling_bias_7d": float(raw.get("rolling_bias_7d", 0)),
        "rolling_bias_21d": float(raw.get("rolling_bias_21d", 0)),
        # Rolling ML self-error (how wrong the ML has been recently)
        "rolling_ml_error_7d": float(raw.get("rolling_ml_error_7d", 0)),

        # Data availability flag
        "has_accu_data": int(raw.get("has_accu_data", int(raw.get("accu_last") is not None))),
    }

    # Use model's own feature list when available (self-consistent with training)
    _cols = FEATURE_COLS
    if hasattr(TEMP_MODEL, "feature_names_in_"):
        _cols = list(TEMP_MODEL.feature_names_in_)
    for col in _cols:
        if col not in row:
            row[col] = np.nan
    X = pd.DataFrame([row], columns=_cols)

    # Fill NaN AccuWeather values with NWS equivalents
    for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
        if pd.isna(X.loc[0, accu_col]):
            X.loc[0, accu_col] = X.loc[0, nws_col]

    return X


app = Flask(__name__)


@app.post("/api/predict-ml")
def predict_ml():
    try:
        payload = request.get_json(force=True)
        X = prepare_features(payload)

        # Model predicts bias (actual - best_base); base = AccuWeather if available, else NWS
        predicted_bias = float(TEMP_MODEL.predict(X)[0])
        accu_last = payload.get("accu_last")
        if accu_last is not None:
            base = float(accu_last)
        else:
            base = float(payload.get("nws_last", payload.get("nws_mean", 0)))
        temp = base + predicted_bias
        bucket_dict = derive_bucket_probabilities(temp, RESIDUAL_STD)
        best_bucket = max(bucket_dict, key=bucket_dict.get)
        confidence = bucket_dict[best_bucket]

        # Top buckets sorted by probability
        probs = sorted(
            [{"bucket": b, "p": p} for b, p in bucket_dict.items()],
            key=lambda d: d["p"], reverse=True,
        )[:5]

        return jsonify({
            "temperature": round(temp, 2),
            "residual_std": round(RESIDUAL_STD, 2),
            "best_bucket": best_bucket,
            "confidence": round(confidence, 4),
            "bucket_probabilities": bucket_dict,
            "probs": probs,
            "should_bet": confidence >= 0.15,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/version")
def version():
    meta = {}
    try:
        with open("model_metadata.json", "r") as f:
            meta = json.load(f)
    except Exception:
        pass
    return {
        "service": "nws-ml-api",
        "model_type": meta.get("model_type"),
        "bucket_method": meta.get("bucket_method"),
        "trained_on": meta.get("trained_on"),
        "date_range": meta.get("date_range"),
        "num_days": meta.get("num_days"),
        "cv_mae": meta.get("model_performance", {}).get("cv_temperature_mae"),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
