# api.py
import json, pickle, numpy as np, pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

# Load models once at startup
with open("temp_model.pkl","rb") as f: TEMP_MODEL = pickle.load(f)
with open("bucket_model.pkl","rb") as f: BUCKET_MODEL = pickle.load(f)

FEATURE_COLS = [
    "nws_first","nws_last","nws_max","nws_min","nws_mean",
    "nws_spread","nws_std","nws_trend","nws_count",
    "forecast_velocity","forecast_acceleration",
    "accu_last","accu_mean","nws_accu_spread",
    "month","day_of_year","is_summer","is_winter",
    "prev_day_nws_error","rolling_nws_bias_7d",
]

def prepare_features(raw):
    m = int(raw.get("month", 1))
    doy = int(raw.get("day_of_year", 0))
    if doy == 0:
        # derive from target_date if provided
        td = raw.get("target_date", "")
        if td:
            try: doy = datetime.strptime(td, "%Y-%m-%d").timetuple().tm_yday
            except Exception: doy = 1
        else:
            doy = 1

    accu_last_val = float(raw["accu_last"]) if raw.get("accu_last") is not None else np.nan
    accu_mean_val = float(raw["accu_mean"]) if raw.get("accu_mean") is not None else accu_last_val
    nws_last_val = float(raw.get("nws_last", np.nan))

    row = {
        "nws_first": float(raw.get("nws_first", np.nan)),
        "nws_last": nws_last_val,
        "nws_max": float(raw.get("nws_max", np.nan)),
        "nws_min": float(raw.get("nws_min", np.nan)),
        "nws_mean": float(raw.get("nws_mean", np.nan)),
        "nws_spread": float(raw.get("nws_spread", 0)),
        "nws_std": float(raw.get("nws_std", 0)),
        "nws_trend": float(raw.get("nws_trend", 0)),
        "nws_count": int(raw.get("nws_count", 1)),
        "forecast_velocity": float(raw.get("forecast_velocity", 0)),
        "forecast_acceleration": float(raw.get("forecast_acceleration", 0)),
        "accu_last": accu_last_val if np.isfinite(accu_last_val) else nws_last_val,
        "accu_mean": accu_mean_val if np.isfinite(accu_mean_val) else nws_last_val,
        "nws_accu_spread": float(raw.get("nws_accu_spread", 0)),
        "month": m,
        "day_of_year": doy,
        "is_summer": int(m in [6,7,8]),
        "is_winter": int(m in [12,1,2]),
        "prev_day_nws_error": float(raw.get("prev_day_nws_error", 0)),
        "rolling_nws_bias_7d": float(raw.get("rolling_nws_bias_7d", 0)),
    }
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    # fill remaining NaN accu fields with NWS
    for col in ("accu_last", "accu_mean"):
        if pd.isna(X.loc[0, col]):
            X.loc[0, col] = X.loc[0, "nws_last"]
    return X

app = Flask(__name__)

@app.post("/api/predict-ml")
def predict_ml():
    try:
        payload = request.get_json(force=True)
        X = prepare_features(payload)

        temp = float(TEMP_MODEL.predict(X)[0])

        if hasattr(BUCKET_MODEL, "predict_proba"):
            proba = BUCKET_MODEL.predict_proba(X)[0]
            classes = list(BUCKET_MODEL.classes_)
            idx = int(np.argmax(proba))
            best_bucket = str(classes[idx])
            confidence = float(proba[idx])
            probs = sorted(
                [{"bucket": str(c), "p": float(p)} for c,p in zip(classes, proba)],
                key=lambda d: d["p"], reverse=True
            )[:3]
        else:
            best_bucket = str(BUCKET_MODEL.predict(X)[0])
            confidence = 0.0
            probs = None

        return jsonify({
            "temperature": round(temp, 2),
            "best_bucket": best_bucket,
            "confidence": round(confidence, 3),
            "probs": probs,
            "should_bet": confidence >= 0.65,
            "generated_at": datetime.utcnow().isoformat() + "Z"
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
        "models_loaded": bool(TEMP_MODEL) and bool(BUCKET_MODEL),
        "trained_on": meta.get("trained_on"),
        "date_range": meta.get("date_range"),
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
