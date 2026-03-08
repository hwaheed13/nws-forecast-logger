# prediction_writer.py
from __future__ import annotations
import os, json, argparse, pickle, math, urllib.request
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# reuse your existing helpers from nws_auto_logger.py (leave that file alone)
from nws_auto_logger import (
    now_nyc, today_nyc, _read_all_rows,
    _compute_avg_bias_excluding, _compute_today_pre_high_mean,
    _float_or_none, compute_today_gate_f,
)
from model_config import FEATURE_COLS, FEATURE_COLS_V2, ACCU_NWS_FALLBACK, derive_bucket_probabilities

MODEL_VERSION = os.environ.get("PREDICTION_MODEL_VERSION", "bcp_v1")

# Kalshi API base URL (public, no auth needed)
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Module-level city key — set by _cli() before write functions run
_CITY_KEY = "nyc"

# ---------------------------------------------------------------------------
# ML model inference
# ---------------------------------------------------------------------------
_ML_MODEL_CACHE: dict = {}


def _load_ml_models():
    """Load temp_model.pkl and bucket_model.pkl once (cached), using city prefix."""
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}temp"
    if cache_key not in _ML_MODEL_CACHE:
        try:
            with open(f"{prefix}temp_model.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            with open(f"{prefix}bucket_model.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}bucket"] = pickle.load(f)
        except FileNotFoundError:
            _ML_MODEL_CACHE[cache_key] = None
            _ML_MODEL_CACHE[f"{prefix}bucket"] = None
            print(f"⚠️ ML model files not found ({prefix}temp_model.pkl) — ML prediction will be skipped")
    return _ML_MODEL_CACHE.get(cache_key), _ML_MODEL_CACHE.get(f"{prefix}bucket")


def _load_v2_models():
    """Load v2 regression model and bucket classifier (cached)."""
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v2_temp"
    if cache_key not in _ML_MODEL_CACHE:
        try:
            with open(f"{prefix}temp_model_v2.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            with open(f"{prefix}bucket_model_v2.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}v2_bucket_info"] = pickle.load(f)
            # Load bucket classifier
            from train_classifier import BucketClassifier
            _ML_MODEL_CACHE[f"{prefix}v2_classifier"] = BucketClassifier.load(
                f"{prefix}bucket_classifier.pkl"
            )
            print(f"✅ Loaded v2 models (prefix='{prefix}')")
        except FileNotFoundError:
            _ML_MODEL_CACHE[cache_key] = None
            _ML_MODEL_CACHE[f"{prefix}v2_bucket_info"] = None
            _ML_MODEL_CACHE[f"{prefix}v2_classifier"] = None
        except Exception as e:
            print(f"⚠️ v2 model load error: {e}")
            _ML_MODEL_CACHE[cache_key] = None
            _ML_MODEL_CACHE[f"{prefix}v2_bucket_info"] = None
            _ML_MODEL_CACHE[f"{prefix}v2_classifier"] = None
    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v2_bucket_info"),
        _ML_MODEL_CACHE.get(f"{prefix}v2_classifier"),
    )


def _fetch_atmospheric_features(target_date_iso: str) -> dict:
    """Fetch atmospheric features from Open-Meteo for today/tomorrow."""
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        from open_meteo_client import get_atmospheric_features_live
        lat = cfg.get("open_meteo_lat", 40.7834)
        lon = cfg.get("open_meteo_lon", -73.965)
        tz = cfg.get("timezone", "America/New_York")
        features = get_atmospheric_features_live(lat, lon, target_date_iso, tz)
        n_valid = sum(1 for v in features.values()
                      if v is not None and not (isinstance(v, float) and math.isnan(v)))
        print(f"🌤️ Atmospheric features: {n_valid} valid values for {target_date_iso}")
        return features
    except Exception as e:
        print(f"⚠️ Atmospheric features fetch failed: {e}")
        return {}


def _fetch_kalshi_market_probs(target_date_iso: str) -> dict:
    """
    Fetch current Kalshi market probabilities for a target date.

    Returns dict like {"48-49": 0.32, "49-50": 0.41, ...} mapping
    bucket labels to market-implied probabilities (0-1).
    Returns empty dict on failure.
    """
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        series = cfg.get("kalshi_series", "KXHIGHNY")

        # Build event ticker: KXHIGHNY-26MAR07
        dt = datetime.strptime(target_date_iso, "%Y-%m-%d")
        yy = dt.strftime("%y")
        mon = ["JAN","FEB","MAR","APR","MAY","JUN",
               "JUL","AUG","SEP","OCT","NOV","DEC"][dt.month - 1]
        dd = dt.strftime("%d")
        event_ticker = f"{series}-{yy}{mon}{dd}"

        url = f"{KALSHI_API_BASE}/markets?event_ticker={event_ticker}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        markets = data.get("markets", [])
        if not markets:
            return {}

        # Filter for open/trading markets
        active = [m for m in markets
                  if str(m.get("status", "")).lower() in ("open", "trading", "active")]
        if not active:
            active = markets

        result = {}
        for m in active:
            # Compute implied probability from bid/ask midpoint
            bid = _parse_kalshi_price(m.get("yes_bid"))
            ask = _parse_kalshi_price(m.get("yes_ask"))
            if bid is not None and ask is not None:
                prob = (bid + ask) / 2
            elif bid is not None:
                prob = bid
            elif ask is not None:
                prob = ask
            else:
                continue

            # Parse bucket label from subtitle or title
            label = m.get("subtitle") or m.get("title") or ""
            bucket = _parse_kalshi_bucket(label)
            if bucket:
                result[bucket] = round(prob, 4)

        if result:
            print(f"📊 Kalshi market: {len(result)} buckets for {target_date_iso}")
            # Show top 3
            top = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
            for b, p in top:
                print(f"   {b}: {p:.0%}")

        return result

    except Exception as e:
        print(f"⚠️ Kalshi market fetch failed: {e}")
        return {}


def _parse_kalshi_price(raw) -> Optional[float]:
    """Parse a Kalshi price to 0-1 probability."""
    if raw is None:
        return None
    try:
        v = float(raw)
        if v > 1 and v <= 100:
            v = v / 100
        if 0 <= v <= 1:
            return v
    except (ValueError, TypeError):
        pass
    return None


def _parse_kalshi_bucket(label: str) -> Optional[str]:
    """
    Parse a Kalshi market label into our bucket format "48-49".
    Handles formats like:
      "48° to 49°" → "48-49"
      "47° or less" → None (skip edge buckets)
      "50° or more" → None (skip edge buckets)
    """
    import re
    clean = label.replace("**", "")

    # Range: "48° to 49°" or "48-49°"
    m = re.match(r".*?(\d+)°?\s*(?:to|-|–)\s*(\d+)°", clean)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Edge buckets — skip for now (our model doesn't predict these well)
    return None


def _map_ml_to_kalshi_buckets(
    ml_bucket_probs: dict,
    kalshi_buckets: dict,
) -> tuple[Optional[str], float, dict]:
    """
    Aggregate ML 1°F bucket probabilities into Kalshi's actual bucket structure.

    Our ML predicts 1°F buckets (e.g., "65-66" = [65, 66)°F).
    Kalshi uses 2°F buckets (e.g., "64-65" covers temps 64° AND 65°).

    For Kalshi bucket "X-Y": sum ML probs for 1°F buckets X→X+1, X+1→X+2, ..., Y→Y+1.

    Returns (best_kalshi_bucket, aggregated_confidence, kalshi_aligned_probs).
    """
    if not ml_bucket_probs or not kalshi_buckets:
        return None, 0.0, {}

    kalshi_aligned = {}
    for kalshi_label in kalshi_buckets:
        parts = kalshi_label.split("-")
        if len(parts) != 2:
            continue
        try:
            lo = int(parts[0])
            hi = int(parts[1])
        except ValueError:
            continue

        # Sum ML probs for 1°F buckets within this Kalshi range
        # Kalshi "64-65" covers integer temps 64 and 65 → our "64-65" + "65-66"
        agg_prob = 0.0
        for t in range(lo, hi + 1):
            ml_key = f"{t}-{t + 1}"
            agg_prob += ml_bucket_probs.get(ml_key, 0.0)

        kalshi_aligned[kalshi_label] = round(agg_prob, 4)

    if not kalshi_aligned:
        return None, 0.0, {}

    best = max(kalshi_aligned, key=kalshi_aligned.get)
    return best, kalshi_aligned[best], kalshi_aligned


def _compute_bet_signal(
    ml_confidence: float,
    ml_bucket: str,
    market_probs: dict,
) -> tuple[str, float]:
    """
    Compute bet signal by comparing model confidence vs market pricing.

    Returns (signal, edge) where:
      signal: "STRONG_BET" / "BET" / "LEAN" / "SKIP"
      edge: model confidence - market probability (positive = model sees value)
    """
    market_prob = market_probs.get(ml_bucket, 0.0)
    edge = ml_confidence - market_prob

    if ml_confidence >= 0.55 and edge >= 0.10:
        signal = "STRONG_BET"
    elif ml_confidence >= 0.40 and edge >= 0.05:
        signal = "BET"
    elif ml_confidence >= 0.30:
        signal = "LEAN"
    else:
        signal = "SKIP"

    return signal, round(edge, 4)


def _compute_ml_prediction(
    rows: list[dict], target_date_iso: str
) -> Optional[dict]:
    """
    Compute ML bias-corrected prediction for *target_date_iso* using the
    same 30 features as train_models.py (NWS stats, AccuWeather stats,
    cross-source, rolling bias, temporal).

    Returns {"ml_f": float, "ml_bucket": str, "ml_confidence": float}
    or None if insufficient data or models missing.
    """
    temp_model, bucket_info = _load_ml_models()
    v2_temp_model_check, _, v2_classifier_check = _load_v2_models()

    # Need at least v1 OR v2 models
    if temp_model is None and v2_classifier_check is None:
        return None

    # --- 1. Collect NWS forecasts for target_date ---
    nws_fc = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast":
            continue
        if r.get("target_date") != target_date_iso:
            continue
        src = (r.get("source") or "").lower()
        if src == "accuweather":
            continue
        ph = _float_or_none(r.get("predicted_high"))
        if ph is None:
            continue
        ts = r.get("timestamp") or r.get("forecast_time") or ""
        nws_fc.append((ts, ph))

    if not nws_fc:
        print(f"⚠️ ML: no NWS forecasts for {target_date_iso}")
        return None

    nws_fc.sort(key=lambda x: x[0])
    nws_vals = np.array([v for _, v in nws_fc])

    # --- 2. Collect AccuWeather forecasts for target_date ---
    accu_fc = []
    for r in rows:
        if r.get("forecast_or_actual") != "forecast":
            continue
        if r.get("target_date") != target_date_iso:
            continue
        src = (r.get("source") or "").lower()
        if src != "accuweather":
            continue
        ph = _float_or_none(r.get("predicted_high"))
        if ph is None:
            continue
        ts = r.get("timestamp") or r.get("forecast_time") or ""
        accu_fc.append((ts, ph))

    accu_fc.sort(key=lambda x: x[0])
    accu_vals = np.array([v for _, v in accu_fc]) if accu_fc else np.array([])
    has_accu = len(accu_vals) > 0

    # --- 3. NWS features ---
    features: dict = {
        "nws_first": float(nws_vals[0]),
        "nws_last": float(nws_vals[-1]),
        "nws_max": float(nws_vals.max()),
        "nws_min": float(nws_vals.min()),
        "nws_mean": float(nws_vals.mean()),
        "nws_spread": float(nws_vals.max() - nws_vals.min()),
        "nws_std": float(nws_vals.std()) if len(nws_vals) > 1 else 0.0,
        "nws_trend": float(nws_vals[-1] - nws_vals[0]) if len(nws_vals) > 1 else 0.0,
        "nws_count": len(nws_vals),
        "forecast_velocity": float(np.diff(nws_vals).mean()) if len(nws_vals) > 1 else 0.0,
        "forecast_acceleration": (
            float(np.diff(np.diff(nws_vals)).mean()) if len(nws_vals) > 2 else 0.0
        ),
    }

    # --- 4. AccuWeather features ---
    if has_accu:
        features.update({
            "accu_first": float(accu_vals[0]),
            "accu_last": float(accu_vals[-1]),
            "accu_max": float(accu_vals.max()),
            "accu_min": float(accu_vals.min()),
            "accu_mean": float(accu_vals.mean()),
            "accu_spread": float(accu_vals.max() - accu_vals.min()),
            "accu_std": float(accu_vals.std()) if len(accu_vals) > 1 else 0.0,
            "accu_trend": float(accu_vals[-1] - accu_vals[0]) if len(accu_vals) > 1 else 0.0,
            "accu_count": len(accu_vals),
        })
    else:
        features.update({
            "accu_first": np.nan, "accu_last": np.nan,
            "accu_max": np.nan, "accu_min": np.nan, "accu_mean": np.nan,
            "accu_spread": 0.0, "accu_std": 0.0, "accu_trend": 0.0,
            "accu_count": 0,
        })

    # --- 5. Cross-source features ---
    if has_accu:
        features["nws_accu_spread"] = abs(features["nws_last"] - features["accu_last"])
        features["nws_accu_mean_diff"] = features["nws_mean"] - features["accu_mean"]
    else:
        features["nws_accu_spread"] = 0.0
        features["nws_accu_mean_diff"] = 0.0

    # --- 6. Temporal features ---
    doy = datetime.strptime(target_date_iso, "%Y-%m-%d").timetuple().tm_yday
    month = int(target_date_iso.split("-")[1])
    features["day_of_year_sin"] = math.sin(2 * math.pi * doy / 365)
    features["day_of_year_cos"] = math.cos(2 * math.pi * doy / 365)
    features["month"] = month
    features["is_summer"] = int(month in (6, 7, 8))
    features["is_winter"] = int(month in (12, 1, 2))

    # --- 7. Rolling bias from strictly prior completed days ---
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        d = r.get("cli_date") if r.get("forecast_or_actual") == "actual" else r.get("target_date")
        if d:
            by_date.setdefault(d, []).append(r)

    daily_biases: list[float] = []
    for d in sorted(by_date.keys()):
        if d >= target_date_iso:
            continue  # strictly prior days only
        rs = by_date[d]
        # find actual
        actual_high = None
        for x in rs:
            if x.get("forecast_or_actual") == "actual":
                actual_high = _float_or_none(x.get("actual_high"))
                if actual_high is not None:
                    break
        if actual_high is None:
            continue
        # NWS forecast mean for this day
        fc_vals = []
        for x in rs:
            if x.get("forecast_or_actual") != "forecast":
                continue
            src = (x.get("source") or "").lower()
            if src == "accuweather":
                continue
            ph = _float_or_none(x.get("predicted_high"))
            if ph is not None:
                fc_vals.append(ph)
        if fc_vals:
            daily_biases.append(actual_high - sum(fc_vals) / len(fc_vals))

    features["rolling_bias_7d"] = float(np.mean(daily_biases[-7:])) if daily_biases else 0.0
    features["rolling_bias_21d"] = float(np.mean(daily_biases[-21:])) if daily_biases else 0.0

    # --- 8. Data availability flag ---
    features["has_accu_data"] = int(has_accu)

    # --- 9. Build DataFrame, fill NaN AccuWeather with NWS fallbacks ---
    X = pd.DataFrame([features])[FEATURE_COLS]
    for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
        if pd.isna(X.loc[0, accu_col]):
            X.loc[0, accu_col] = X.loc[0, nws_col]

    # Base forecast: best available source
    base = features["accu_last"] if has_accu else features["nws_last"]
    base_src = "accu" if has_accu else "nws"

    # --- 10. v1 regression prediction (if v1 models available) ---
    result = {}
    if temp_model is not None:
        predicted_bias = float(temp_model.predict(X)[0])
        ml_temp = base + predicted_bias

        residual_std = 2.0
        if isinstance(bucket_info, dict) and "residual_std" in bucket_info:
            residual_std = bucket_info["residual_std"]

        bucket_dict = derive_bucket_probabilities(ml_temp, residual_std)
        best_bucket = max(bucket_dict, key=bucket_dict.get)
        confidence = bucket_dict[best_bucket]

        print(f"🤖 ML v1 prediction for {target_date_iso}: {ml_temp:.1f}°F "
              f"(base={base_src}={base:.0f}, bias={predicted_bias:+.2f}, "
              f"bucket={best_bucket}, conf={confidence:.2%})")

        result = {
            "ml_f": round(ml_temp, 1),
            "ml_bucket": best_bucket,
            "ml_confidence": round(confidence, 4),
        }

    # --- 11. Try v2 models (atmospheric + classifier) ---
    v2_temp_model, v2_bucket_info, v2_classifier = _load_v2_models()
    if v2_classifier is not None:
        try:
            # Fetch atmospheric features
            atm_features = _fetch_atmospheric_features(target_date_iso)

            # Merge atmospheric features into the feature dict
            v2_features = dict(features)
            v2_features.update(atm_features)

            # Build v2 feature DataFrame
            X_v2 = pd.DataFrame([v2_features])
            # Add any missing v2 columns as NaN
            for col in FEATURE_COLS_V2:
                if col not in X_v2.columns:
                    X_v2[col] = np.nan
            X_v2 = X_v2[FEATURE_COLS_V2]
            # Fill AccuWeather fallbacks
            for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
                if pd.isna(X_v2.loc[0, accu_col]):
                    X_v2.loc[0, accu_col] = X_v2.loc[0, nws_col]

            # v2 regression prediction (if v2 regression model available)
            if v2_temp_model is not None:
                v2_bias = float(v2_temp_model.predict(X_v2)[0])
                v2_temp = base + v2_bias
            else:
                # No regression model — use raw forecast as center
                v2_temp = float(base)

            # v2 classifier bucket prediction
            # Use 11 candidates (±5) to cover Kalshi's full range
            bucket_probs = v2_classifier.predict_bucket_probs(
                features=v2_features,
                center_temp=v2_temp,
                accu_last=features.get("accu_last") if has_accu else None,
                nws_last=features.get("nws_last"),
                n_candidates=11,
            )

            if bucket_probs:
                v2_best = bucket_probs[0]
                result["ml_f"] = round(v2_temp, 1)
                result["ml_bucket"] = v2_best["bucket"]
                result["ml_confidence"] = v2_best["probability"]
                result["ml_bucket_probs"] = json.dumps(
                    {bp["bucket"]: bp["probability"] for bp in bucket_probs}
                )
                result["ml_version"] = "v2_atm_classifier"

                print(f"🧠 ML v2 prediction for {target_date_iso}: {v2_temp:.1f}°F "
                      f"→ bucket={v2_best['bucket']} ({v2_best['probability']:.0%})")
                if len(bucket_probs) > 1:
                    runner = bucket_probs[1]
                    print(f"   Runner-up: {runner['bucket']} ({runner['probability']:.0%})")
        except Exception as e:
            print(f"⚠️ v2 prediction failed, using v1: {e}")

    return result

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
        f"{endpoint}?on_conflict=idempotency_key",
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
        print("✅ upsert:", {k: row.get(k) for k in ("lead_used","target_date","timestamp","prediction_value","ml_f")})
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

    avg_bias_excl_today = _compute_avg_bias_excluding(rows, target_date_iso)
    today_pre_mean      = _compute_today_pre_high_mean(rows, target_date_iso)
    nws_latest  = _latest_forecast(rows, target_date_iso, source=None)
    accu_latest = _latest_forecast(rows, target_date_iso, source="accu")

    bcp = None
    if avg_bias_excl_today is not None and today_pre_mean is not None:
        bcp = today_pre_mean + avg_bias_excl_today

    # ML model prediction (graceful — returns None if models missing or no data)
    ml = _compute_ml_prediction(rows, target_date_iso)

    if bcp is None and ml is None:
        print("⏭️ today_for_today: no BCP data and no ML prediction available."); return

    ts = now_nyc().isoformat()
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:today_for_today:{target_date_iso}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": target_date_iso,
        "lead_used": "today_for_today",
        "model_name": MODEL_VERSION,
        "prediction_value": float(f"{bcp:.1f}") if bcp is not None else None,
        "nws_d0": nws_latest,
        "accuweather": accu_latest,
        "rep_forecast": today_pre_mean,
        "bias_applied": avg_bias_excl_today,
        "version": MODEL_VERSION,
        "recommendation": "frozen at actual time",
        "source_card": "nws_auto_logger",
        "city": _CITY_KEY,
    }
    if ml:
        payload["ml_f"] = ml["ml_f"]
        payload["ml_bucket"] = ml["ml_bucket"]
        payload["ml_confidence"] = ml["ml_confidence"]
        if ml.get("ml_bucket_probs"):
            payload["ml_bucket_probs"] = ml["ml_bucket_probs"]
        if ml.get("ml_version"):
            payload["ml_version"] = ml["ml_version"]

    # Fetch Kalshi market odds + compute bet signal
    market_probs = _fetch_kalshi_market_probs(target_date_iso)
    if market_probs:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    # Map ML 1°F buckets → Kalshi's actual bucket structure
    if ml and market_probs and ml.get("ml_bucket_probs"):
        raw_probs = json.loads(ml["ml_bucket_probs"]) if isinstance(ml["ml_bucket_probs"], str) else ml["ml_bucket_probs"]
        kalshi_bucket, kalshi_conf, kalshi_aligned = _map_ml_to_kalshi_buckets(raw_probs, market_probs)
        if kalshi_bucket:
            payload["ml_bucket"] = kalshi_bucket
            payload["ml_confidence"] = kalshi_conf
            print(f"🎯 Kalshi-aligned bucket: {kalshi_bucket} ({kalshi_conf:.0%})")
            # Bet signal uses Kalshi-aligned confidence
            signal, edge = _compute_bet_signal(kalshi_conf, kalshi_bucket, market_probs)
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
    elif ml and market_probs:
        signal, edge = _compute_bet_signal(
            ml["ml_confidence"], ml["ml_bucket"], market_probs
        )
        payload["bet_signal"] = signal
        payload["ml_edge"] = edge
        print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")

    supabase_upsert(payload)

def write_today_for_tomorrow(tomorrow_iso: Optional[str] = None) -> None:
    # default to local tomorrow if not provided
    if not tomorrow_iso:
        tomorrow_iso = (today_nyc() + timedelta(days=1)).isoformat()

    rows, _ = _read_all_rows(include_accu=True)

    avg_bias_all   = _compute_avg_bias_excluding(rows, exclude_date_iso="")
    nws_latest_tm  = _latest_forecast(rows, tomorrow_iso, source=None)
    accu_latest_tm = _latest_forecast(rows, tomorrow_iso, source="accu")

    bcp_tm = None
    if nws_latest_tm is not None and avg_bias_all is not None:
        bcp_tm = float(f"{(nws_latest_tm + avg_bias_all):.1f}")

    # ML model prediction (graceful — returns None if models missing or no data)
    ml = _compute_ml_prediction(rows, tomorrow_iso)

    ts = now_nyc().isoformat()
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:today_for_tomorrow:{tomorrow_iso}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": tomorrow_iso,
        "lead_used": "today_for_tomorrow",
        "model_name": MODEL_VERSION,
        "prediction_value": bcp_tm,
        "nws_d1": nws_latest_tm,
        "accuweather": accu_latest_tm,
        "bias_applied": avg_bias_all,
        "rep_forecast": None,
        "version": MODEL_VERSION,
        "recommendation": "snapshot from today",
        "source_card": "nws_auto_logger",
        "city": _CITY_KEY,
    }
    if ml:
        payload["ml_f"] = ml["ml_f"]
        payload["ml_bucket"] = ml["ml_bucket"]
        payload["ml_confidence"] = ml["ml_confidence"]
        if ml.get("ml_bucket_probs"):
            payload["ml_bucket_probs"] = ml["ml_bucket_probs"]
        if ml.get("ml_version"):
            payload["ml_version"] = ml["ml_version"]

    # Fetch Kalshi market odds + compute bet signal
    market_probs = _fetch_kalshi_market_probs(tomorrow_iso)
    if market_probs:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    # Map ML 1°F buckets → Kalshi's actual bucket structure
    if ml and market_probs and ml.get("ml_bucket_probs"):
        raw_probs = json.loads(ml["ml_bucket_probs"]) if isinstance(ml["ml_bucket_probs"], str) else ml["ml_bucket_probs"]
        kalshi_bucket, kalshi_conf, kalshi_aligned = _map_ml_to_kalshi_buckets(raw_probs, market_probs)
        if kalshi_bucket:
            payload["ml_bucket"] = kalshi_bucket
            payload["ml_confidence"] = kalshi_conf
            print(f"🎯 Kalshi-aligned bucket: {kalshi_bucket} ({kalshi_conf:.0%})")
            signal, edge = _compute_bet_signal(kalshi_conf, kalshi_bucket, market_probs)
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
    elif ml and market_probs:
        signal, edge = _compute_bet_signal(
            ml["ml_confidence"], ml["ml_bucket"], market_probs
        )
        payload["bet_signal"] = signal
        payload["ml_edge"] = edge
        print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")

    supabase_upsert(payload)

def write_both_snapshots() -> None:
    try: write_today_for_today()
    except Exception as e: print("⚠️ write_today_for_today failed:", e)
    try: write_today_for_tomorrow()
    except Exception as e: print("⚠️ write_today_for_tomorrow failed:", e)

def _cli():
    import argparse
    from nws_auto_logger import set_city
    from city_config import DEFAULT_CITY

    p = argparse.ArgumentParser(description="Write prediction snapshots to Supabase.")
    p.add_argument("--city", default=os.environ.get("CITY", DEFAULT_CITY),
                   help="City key (nyc, lax, etc.)")
    s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("today_for_today");    a.add_argument("--date")
    b = s.add_parser("today_for_tomorrow"); b.add_argument("--date")
    s.add_parser("both")
    args = p.parse_args()

    set_city(args.city)
    global _CITY_KEY
    _CITY_KEY = args.city
    print(f"[prediction_writer] city={args.city}")

    if args.cmd == "today_for_today":    write_today_for_today(args.date)
    elif args.cmd == "today_for_tomorrow": write_today_for_tomorrow(args.date)
    else: write_both_snapshots()

if __name__ == "__main__": _cli()
