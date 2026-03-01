# prediction_writer.py
from __future__ import annotations
import os, json, argparse, urllib.request, statistics
from datetime import date, datetime, timedelta
from typing import Optional

# reuse your existing helpers from nws_auto_logger.py (leave that file alone)
from nws_auto_logger import (
    now_nyc, today_nyc, _read_all_rows,
    _compute_avg_bias_excluding, _compute_today_pre_high_mean,
    _float_or_none, compute_today_gate_f,
)

# ML deps — optional, graceful skip if not installed
try:
    import pickle
    import numpy as np
    import pandas as pd
    HAS_ML = True
except ImportError:
    HAS_ML = False

MODEL_VERSION = os.environ.get("PREDICTION_MODEL_VERSION", "bcp_v2")

FEATURE_COLS = [
    "nws_first", "nws_last", "nws_max", "nws_min", "nws_mean",
    "nws_spread", "nws_std", "nws_trend", "nws_count",
    "forecast_velocity", "forecast_acceleration",
    "accu_last", "accu_mean", "nws_accu_spread",
    "month", "day_of_year", "is_summer", "is_winter",
    "prev_day_nws_error", "rolling_nws_bias_7d",
]

def _sb_endpoint():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE")
    return f"{url}/rest/v1/prediction_logs", key

def supabase_upsert(row: dict) -> None:
    endpoint, key = _sb_endpoint()
    data = json.dumps(row, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}?on_conflict=target_date,record_type,as_of",
        data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            _ = resp.read()
        print("✅ upsert:", {k: row.get(k) for k in ("record_type","target_date","as_of","bcp_f")})
    except Exception as e:
        if hasattr(e, "read"):
            try: print("❌ supabase:", getattr(e,'code','?'), e.read().decode("utf-8", "ignore"))
            except: print("❌ supabase:", e)
        else:
            print("❌ supabase:", e)

def _latest_forecast(rows: list[dict], date_iso: str, source: Optional[str]) -> Optional[float]:
    """source=None → NWS; source='accu' → AccuWeather"""
    cands = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast": continue
        if r.get("target_date") != date_iso: continue
        if _float_or_none(r.get("predicted_high")) is None: continue
        src = (r.get("source") or "").lower()
        if source == "accu":
            if src != "accuweather": continue
        else:
            if src == "accuweather": continue
        key = (r.get("timestamp") or r.get("forecast_time") or "")
        cands.append((key, float(r["predicted_high"])))
    if not cands: return None
    cands.sort(key=lambda kv: kv[0])
    return cands[-1][1]

def write_today_for_today(target_date_iso: Optional[str] = None) -> None:
    if not target_date_iso:
        target_date_iso = today_nyc().isoformat()
    rows, _ = _read_all_rows(include_accu=True)

    target_month = int(target_date_iso[5:7])
    avg_bias_excl_today = _compute_avg_bias_excluding(rows, target_date_iso, target_month=target_month)
    today_pre_mean      = _compute_today_pre_high_mean(rows, target_date_iso)
    if avg_bias_excl_today is None or today_pre_mean is None:
        print("⏭️ today_for_today: not enough data (need avg_bias_excl_today & today_pre_mean)."); return

    bcp = today_pre_mean + avg_bias_excl_today
    nws_latest  = _latest_forecast(rows, target_date_iso, source=None)
    accu_latest = _latest_forecast(rows, target_date_iso, source="accu")

    supabase_upsert({
        "as_of": now_nyc().isoformat(),
        "target_date": target_date_iso,
        "record_type": "today_for_today",
        "bcp_f": float(f"{bcp:.1f}"),
        "nws_latest_f": nws_latest,
        "accu_latest_f": accu_latest,
        "avg_bias_excl_today": avg_bias_excl_today,
        "today_pre_mean": today_pre_mean,
        "gate_f": compute_today_gate_f(),
        "model_version": MODEL_VERSION,
        "notes": "frozen at actual time",
        "source": "nws_auto_logger",
    })

def write_today_for_tomorrow(tomorrow_iso: Optional[str] = None) -> None:
    # default to local tomorrow if not provided
    if not tomorrow_iso:
        tomorrow_iso = (today_nyc() + timedelta(days=1)).isoformat()

    rows, _ = _read_all_rows(include_accu=True)

    target_month = int(tomorrow_iso[5:7])
    avg_bias_all   = _compute_avg_bias_excluding(rows, exclude_date_iso="", target_month=target_month)
    nws_latest_tm  = _latest_forecast(rows, tomorrow_iso, source=None)
    accu_latest_tm = _latest_forecast(rows, tomorrow_iso, source="accu")

    bcp_tm = None
    if nws_latest_tm is not None and avg_bias_all is not None:
        bcp_tm = float(f"{(nws_latest_tm + avg_bias_all):.1f}")

    supabase_upsert({
        "as_of": now_nyc().isoformat(),
        "target_date": tomorrow_iso,
        "record_type": "today_for_tomorrow",
        "bcp_f": bcp_tm,
        "nws_latest_f": nws_latest_tm,
        "accu_latest_f": accu_latest_tm,
        "avg_bias_excl_today": avg_bias_all,
        "today_pre_mean": None,
        "gate_f": None,
        "model_version": MODEL_VERSION,
        "notes": "snapshot from today",
        "source": "nws_auto_logger",
    })

def _get_nws_forecasts(rows: list[dict], date_iso: str) -> list[float]:
    """Return ordered list of NWS forecast values for a date."""
    pairs = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast": continue
        if r.get("target_date") != date_iso: continue
        if (r.get("source") or "").lower() == "accuweather": continue
        v = _float_or_none(r.get("predicted_high"))
        if v is None: continue
        ts = r.get("timestamp") or r.get("forecast_time") or ""
        pairs.append((ts, v))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


def _get_accu_forecasts(rows: list[dict], date_iso: str) -> list[float]:
    """Return ordered list of AccuWeather forecast values for a date."""
    pairs = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast": continue
        if r.get("target_date") != date_iso: continue
        if (r.get("source") or "").lower() != "accuweather": continue
        v = _float_or_none(r.get("predicted_high"))
        if v is None: continue
        ts = r.get("timestamp") or r.get("forecast_time") or ""
        pairs.append((ts, v))
    pairs.sort(key=lambda p: p[0])
    return [v for _, v in pairs]


def _get_daily_errors(rows: list[dict]) -> dict[str, float]:
    """Return {date_iso: actual - nws_last} for all dates with actuals."""
    # Collect actuals
    actuals = {}
    for r in rows:
        if r.get("forecast_or_actual") != "actual": continue
        d = r.get("cli_date") or r.get("target_date") or ""
        v = _float_or_none(r.get("actual_high"))
        if d and v is not None:
            actuals[d] = v

    # For each date with an actual, find nws_last
    errors = {}
    for d, actual in actuals.items():
        nws = _get_nws_forecasts(rows, d)
        if nws:
            errors[d] = actual - nws[-1]
    return errors


def build_features_for_date(rows: list[dict], target_date_iso: str,
                            accu_rows: Optional[list[dict]] = None) -> Optional[dict]:
    """Build the 20-feature vector for a single target date from CSV rows."""
    nws = _get_nws_forecasts(rows, target_date_iso)
    if not nws:
        return None

    # NWS aggregate features
    feat = {
        "nws_first": nws[0],
        "nws_last": nws[-1],
        "nws_max": max(nws),
        "nws_min": min(nws),
        "nws_mean": sum(nws) / len(nws),
        "nws_spread": max(nws) - min(nws),
        "nws_std": statistics.stdev(nws) if len(nws) > 1 else 0.0,
        "nws_trend": nws[-1] - nws[0] if len(nws) > 1 else 0.0,
        "nws_count": len(nws),
    }

    # Forecast velocity and acceleration (match train_models.py logic)
    if len(nws) >= 2:
        diffs = [nws[i+1] - nws[i] for i in range(len(nws)-1)]
        feat["forecast_velocity"] = sum(diffs) / len(diffs)
        if len(diffs) >= 2:
            acc = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            feat["forecast_acceleration"] = sum(acc) / len(acc)
        else:
            feat["forecast_acceleration"] = 0.0
    else:
        feat["forecast_velocity"] = 0.0
        feat["forecast_acceleration"] = 0.0

    # AccuWeather features
    accu = _get_accu_forecasts(accu_rows or rows, target_date_iso)
    feat["accu_last"] = accu[-1] if accu else feat["nws_last"]
    feat["accu_mean"] = (sum(accu) / len(accu)) if accu else feat["nws_mean"]
    feat["nws_accu_spread"] = abs(feat["nws_last"] - feat["accu_last"])

    # Calendar features
    d = date.fromisoformat(target_date_iso)
    feat["month"] = d.month
    feat["day_of_year"] = d.timetuple().tm_yday
    feat["is_summer"] = int(d.month in (6, 7, 8))
    feat["is_winter"] = int(d.month in (12, 1, 2))

    # Temporal error features (yesterday's and rolling errors)
    errors = _get_daily_errors(rows)
    yesterday = (d - timedelta(days=1)).isoformat()
    feat["prev_day_nws_error"] = errors.get(yesterday, 0.0)

    # Rolling 7-day bias: average of errors for the 7 days before target
    recent_errors = []
    for i in range(1, 8):
        past = (d - timedelta(days=i)).isoformat()
        if past in errors:
            recent_errors.append(errors[past])
    feat["rolling_nws_bias_7d"] = (sum(recent_errors) / len(recent_errors)) if recent_errors else 0.0

    return feat


def write_ml_predictions() -> None:
    """Load trained ML models, predict for today/tomorrow, write JSON."""
    if not HAS_ML:
        print("⏭️ ML deps not installed, skipping ML predictions")
        return

    # Check model files exist
    for fname in ("temp_model.pkl", "bucket_model.pkl"):
        if not os.path.exists(fname):
            print(f"⏭️ {fname} not found, skipping ML predictions")
            return

    # Validate feature count matches
    try:
        with open("model_metadata.json") as f:
            meta = json.load(f)
        model_features = meta.get("feature_columns", [])
        if len(model_features) != len(FEATURE_COLS):
            print(f"⏭️ Model has {len(model_features)} features, code expects {len(FEATURE_COLS)}. Retrain needed.")
            return
    except Exception:
        pass  # metadata missing is OK, proceed with caution

    # Load models
    with open("temp_model.pkl", "rb") as f:
        temp_model = pickle.load(f)
    with open("bucket_model.pkl", "rb") as f:
        bucket_model = pickle.load(f)

    rows, _ = _read_all_rows(include_accu=True)
    today_iso = today_nyc().isoformat()
    tomorrow_iso = (today_nyc() + timedelta(days=1)).isoformat()

    predictions = {}
    for target_iso in (today_iso, tomorrow_iso):
        feat = build_features_for_date(rows, target_iso)
        if feat is None:
            print(f"⏭️ No forecasts for {target_iso}, skipping ML prediction")
            continue

        X = pd.DataFrame([feat], columns=FEATURE_COLS)
        # Fill NaN accu fields with NWS (match training pipeline)
        for col in ("accu_last", "accu_mean"):
            if pd.isna(X.loc[0, col]):
                X.loc[0, col] = X.loc[0, "nws_last"]

        temp = float(temp_model.predict(X)[0])

        if hasattr(bucket_model, "predict_proba"):
            proba = bucket_model.predict_proba(X)[0]
            classes = list(bucket_model.classes_)
            idx = int(np.argmax(proba))
            best_bucket = str(classes[idx])
            confidence = float(proba[idx])
            top_buckets = sorted(
                [{"bucket": str(c), "p": round(float(p), 4)} for c, p in zip(classes, proba)],
                key=lambda d: d["p"], reverse=True
            )[:3]
        else:
            best_bucket = str(bucket_model.predict(X)[0])
            confidence = 0.0
            top_buckets = [{"bucket": best_bucket, "p": 0.0}]

        predictions[target_iso] = {
            "temperature": round(temp, 2),
            "bucket": best_bucket,
            "confidence": round(confidence, 4),
            "top_buckets": top_buckets,
        }
        print(f"🤖 ML prediction for {target_iso}: {temp:.1f}°F → {best_bucket} ({confidence:.0%})")

    if not predictions:
        print("⏭️ No ML predictions generated")
        return

    output = {
        "generated_at": now_nyc().isoformat(),
        "model_version": MODEL_VERSION,
        "predictions": predictions,
    }

    out_path = os.path.join("public", "ml_predictions.json")
    os.makedirs("public", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Wrote {out_path} with {len(predictions)} predictions")


def write_both_snapshots() -> None:
    try: write_today_for_today()
    except Exception as e: print("⚠️ write_today_for_today failed:", e)
    try: write_today_for_tomorrow()
    except Exception as e: print("⚠️ write_today_for_tomorrow failed:", e)
    try: write_ml_predictions()
    except Exception as e: print("⚠️ write_ml_predictions failed:", e)

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Write prediction snapshots to Supabase.")
    s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("today_for_today");    a.add_argument("--date")
    b = s.add_parser("today_for_tomorrow"); b.add_argument("--date")
    s.add_parser("both")
    args = p.parse_args()
    if args.cmd == "today_for_today":    write_today_for_today(args.date)
    elif args.cmd == "today_for_tomorrow": write_today_for_tomorrow(args.date)
    else: write_both_snapshots()

if __name__ == "__main__": _cli()
