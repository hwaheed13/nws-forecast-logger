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
from model_config import (
    FEATURE_COLS, FEATURE_COLS_V2, ACCU_NWS_FALLBACK,
    ATM_PREDICTOR_INPUT_COLS, derive_bucket_probabilities,
)

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
    """Load v2 regression model and bucket classifier (cached).
    Regression model is optional — classifier can work alone."""
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v2_temp"
    if cache_key not in _ML_MODEL_CACHE:
        # Initialize all to None
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v2_bucket_info"] = None
        _ML_MODEL_CACHE[f"{prefix}v2_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v2_atm_predictor"] = None

        # v2 regression model (optional)
        try:
            with open(f"{prefix}temp_model_v2.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            with open(f"{prefix}bucket_model_v2.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}v2_bucket_info"] = pickle.load(f)
        except FileNotFoundError:
            pass  # regression model optional

        # Atmospheric predictor (first-stage model, trained on 1,278 historical days)
        try:
            with open(f"{prefix}atm_predictor.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}v2_atm_predictor"] = pickle.load(f)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ atm predictor load error: {e}")

        # Bucket classifier (the important one)
        try:
            from train_classifier import BucketClassifier
            _ML_MODEL_CACHE[f"{prefix}v2_classifier"] = BucketClassifier.load(
                f"{prefix}bucket_classifier.pkl"
            )
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v2 classifier load error: {e}")

        loaded = []
        if _ML_MODEL_CACHE[cache_key] is not None:
            loaded.append("regression")
        if _ML_MODEL_CACHE[f"{prefix}v2_atm_predictor"] is not None:
            loaded.append("atm_predictor")
        if _ML_MODEL_CACHE[f"{prefix}v2_classifier"] is not None:
            loaded.append("classifier")
        if loaded:
            print(f"✅ Loaded v2 models: {', '.join(loaded)} (prefix='{prefix}')")
    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v2_bucket_info"),
        _ML_MODEL_CACHE.get(f"{prefix}v2_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v2_atm_predictor"),
    )


def _load_v3_model():
    """Load v3 unified regression model + atmospheric predictor (cached).
    v3 predicts actual_high directly from all features — no classifier needed."""
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v3_model"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v3_atm_predictor"] = None

        try:
            with open(f"{prefix}temp_model_v3.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v3 unified model (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v3 model load error: {e}")

        # Atmospheric predictor (needed to compute atm_predicted_high feature)
        try:
            with open(f"{prefix}atm_predictor.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}v3_atm_predictor"] = pickle.load(f)
        except FileNotFoundError:
            pass

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v3_atm_predictor"),
    )


def _fetch_observed_high_so_far(target_date_iso: str) -> tuple:
    """
    Fetch today's observed high temperature from NWS station observations.

    Returns (observed_high_f, obs_hour_local) or (None, None).
      - observed_high_f: highest temp (°F) observed today
      - obs_hour_local: local hour (0-23) when that high was observed

    Only fetches for today's date (not tomorrow/past).
    Used for:
      1. Safety floor guard (drop impossible buckets)
      2. Forecast exceedance detection (shift center when forecast is busted)
    """
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        station = cfg.get("obs_station", "KNYC")
        tz_name = cfg.get("timezone", "America/New_York")

        today = today_nyc().isoformat()
        if target_date_iso != today:
            return None, None

        url = f"https://api.weather.gov/stations/{station}/observations?limit=100"
        req = urllib.request.Request(url, headers={
            "Accept": "application/geo+json",
            "User-Agent": "nws-forecast-logger/1.0",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        obs_features = data.get("features", [])
        if not obs_features:
            return None, None

        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)

        high_f = None
        high_hour = None
        for f in obs_features:
            props = f.get("properties", {})
            ts = props.get("timestamp", "")
            if not ts:
                continue
            obs_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            obs_local = obs_dt.astimezone(tz)
            if obs_local.strftime("%Y-%m-%d") != target_date_iso:
                continue
            temp_c = props.get("temperature", {}).get("value")
            if temp_c is None:
                continue
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            if high_f is None or temp_f > high_f:
                high_f = temp_f
                high_hour = obs_local.hour

        if high_f is not None:
            high_f = round(high_f, 1)
            print(f"🌡️ Observed high so far: {high_f}°F at {high_hour}:00 local")
        return high_f, high_hour
    except Exception as e:
        print(f"⚠️ Could not fetch observed high: {e}")
        return None, None


def _adjust_center_for_exceedance(
    center_temp: float,
    observed_high: float,
    obs_hour: Optional[int],
) -> float:
    """
    When the observed temperature EXCEEDS the forecast center, the forecast
    is wrong. Shift the center up based on the observed temp + estimated
    remaining afternoon heating.

    This is the key fix: if NWS says 66 but it's already 67 at 1pm,
    we shift the center to ~69 so the classifier generates the right
    candidate buckets (68-69, 70-71, etc.).
    """
    if observed_high is None or observed_high <= center_temp:
        return center_temp  # forecast not exceeded, no adjustment

    if obs_hour is None:
        obs_hour = 12  # default assumption

    # Only trust exceedance during daytime heating hours (8 AM – 5 PM).
    # Overnight warmth (e.g., 51°F at midnight from prior day's warm air)
    # does NOT indicate the day's high will exceed the forecast — temps
    # often DROP during the day after a warm front passes.
    if obs_hour < 8 or obs_hour >= 17:
        print(f"ℹ️ Observed {observed_high}°F > forecast {center_temp:.1f}°F "
              f"at {obs_hour}:00 — ignoring (outside daytime heating window)")
        return center_temp

    # Estimate remaining heating based on hour of day
    # In spring/fall, peak heating is typically 2-4pm local
    remaining_heat = {
        8: 5.0, 9: 4.5, 10: 4.0, 11: 3.5, 12: 2.5,
        13: 2.0, 14: 1.5, 15: 1.0, 16: 0.5,
    }
    extra = remaining_heat.get(obs_hour, 1.0)

    adjusted = observed_high + extra
    print(f"⚡ Exceedance: observed {observed_high}°F > forecast {center_temp:.1f}°F "
          f"at {obs_hour}:00 → adjusted center to {adjusted:.1f}°F (+{extra}°F est. remaining)")
    return adjusted


def _apply_observed_floor(bucket_probs: dict, observed_high: float) -> dict:
    """
    Safety guard: zero out buckets where the UPPER bound is below the
    observed high. If it's already 66.3°F, bucket "64-65" (upper=65) is
    impossible. Renormalize remaining buckets.

    This is NOT an ML feature — it's a post-prediction sanity check
    to prevent embarrassing signals.
    """
    if not bucket_probs or observed_high is None:
        return bucket_probs

    floor = int(observed_high)  # 66.3 → 66
    filtered = {}
    dropped = []
    for bucket, prob in bucket_probs.items():
        parts = bucket.split("-")
        if len(parts) != 2:
            filtered[bucket] = prob
            continue
        try:
            bucket_hi = int(parts[1])
        except ValueError:
            filtered[bucket] = prob
            continue
        # Keep if upper bound > observed floor
        # If observed is 67.2, bucket "66-67" (upper=67) is likely impossible
        if bucket_hi > floor:
            filtered[bucket] = prob
        else:
            dropped.append(bucket)

    if dropped:
        total = sum(filtered.values())
        if total > 0:
            filtered = {k: round(v / total, 4) for k, v in filtered.items()}
        print(f"🛡️ Floor guard: dropped {len(dropped)} impossible bucket(s) below {floor}°F")

    return filtered


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
            # Kalshi API returns prices in _dollars fields (0-1 scale)
            # Fall back to legacy field names for backwards compat
            bid_d = _parse_kalshi_price(m.get("yes_bid_dollars"))
            bid = bid_d if bid_d is not None else _parse_kalshi_price(m.get("yes_bid"))
            ask_d = _parse_kalshi_price(m.get("yes_ask_dollars"))
            ask = ask_d if ask_d is not None else _parse_kalshi_price(m.get("yes_ask"))
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
    Parse a Kalshi market label into our bucket format.
    Handles formats like:
      "48° to 49°" → "48-49"       (standard range bucket)
      "47° or less" → "<=47"       (lower edge bucket)
      "50° or more" → ">=50"       (upper edge bucket)
      "Below 47°"   → "<=47"
      "Above 70°"   → ">=70"
    """
    import re
    clean = label.replace("**", "").strip()

    # Range: "48° to 49°" or "48-49°"
    m = re.match(r".*?(\d+)°?\s*(?:to|-|–)\s*(\d+)°", clean)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Upper edge: "70° or more", "70° or above", "Above 70°", "70° or higher"
    m = re.search(r"(\d+)°?\s*or\s*(?:more|above|higher|greater)", clean, re.IGNORECASE)
    if m:
        return f">={m.group(1)}"
    m = re.search(r"(?:above|over|higher\s+than|more\s+than)\s*(\d+)°?", clean, re.IGNORECASE)
    if m:
        return f">={m.group(1)}"

    # Lower edge: "47° or less", "47° or below", "Below 47°", "47° or lower"
    m = re.search(r"(\d+)°?\s*or\s*(?:less|below|lower|fewer)", clean, re.IGNORECASE)
    if m:
        return f"<={m.group(1)}"
    m = re.search(r"(?:below|under|less\s+than|lower\s+than)\s*(\d+)°?", clean, re.IGNORECASE)
    if m:
        return f"<={m.group(1)}"

    return None


def _find_kalshi_bucket_for_temp(predicted_temp: float, kalshi_buckets: dict) -> Optional[str]:
    """
    Given a predicted temperature, find which Kalshi bucket contains it.

    Kalshi buckets change daily. Structure is always:
      - "<=X" (lower edge: X and below)
      - "A-B" (range: covers integer temps A through B inclusive)
      - ">=Y" (upper edge: Y and above)

    The predicted_temp is rounded to nearest integer, then we find
    which bucket that integer falls into.
    """
    temp_int = int(round(predicted_temp))

    for label in kalshi_buckets:
        # Upper edge: ">=70" means 70 and above
        if label.startswith(">="):
            try:
                threshold = int(label[2:])
                if temp_int >= threshold:
                    return label
            except ValueError:
                continue

        # Lower edge: "<=47" means 47 and below
        elif label.startswith("<="):
            try:
                threshold = int(label[2:])
                if temp_int <= threshold:
                    return label
            except ValueError:
                continue

        # Standard range: "68-69" means integer temps 68 and 69
        elif "-" in label:
            parts = label.split("-")
            if len(parts) == 2:
                try:
                    lo, hi = int(parts[0]), int(parts[1])
                    if lo <= temp_int <= hi:
                        return label
                except ValueError:
                    continue

    return None


def _map_ml_to_kalshi_buckets(
    ml_bucket_probs: dict,
    kalshi_buckets: dict,
) -> tuple[Optional[str], float, dict]:
    """
    Aggregate ML 1°F bucket probabilities into Kalshi's actual bucket structure.

    Our ML predicts 1°F buckets (e.g., "65-66" = [65, 66)°F, meaning high = 65).
    Kalshi uses varying bucket sizes that change daily:
      - "<=47": all temps 47 and below
      - "48-49": integer temps 48 and 49 (our ML "48-49" + "49-50")
      - ">=70": all temps 70 and above

    Returns (best_kalshi_bucket, aggregated_confidence, kalshi_aligned_probs).
    """
    if not ml_bucket_probs or not kalshi_buckets:
        return None, 0.0, {}

    kalshi_aligned = {}
    for kalshi_label in kalshi_buckets:
        agg_prob = 0.0

        # Upper edge bucket: ">=70" — sum all ML probs for temps 70+
        if kalshi_label.startswith(">="):
            try:
                threshold = int(kalshi_label[2:])
                for ml_key, prob in ml_bucket_probs.items():
                    parts = ml_key.split("-")
                    if len(parts) == 2:
                        ml_lo = int(parts[0])
                        if ml_lo >= threshold:
                            agg_prob += prob
            except ValueError:
                continue

        # Lower edge bucket: "<=47" — sum all ML probs for temps 47 and below
        elif kalshi_label.startswith("<="):
            try:
                threshold = int(kalshi_label[2:])
                for ml_key, prob in ml_bucket_probs.items():
                    parts = ml_key.split("-")
                    if len(parts) == 2:
                        ml_lo = int(parts[0])
                        # ML bucket "47-48" means high=47, which is <=47
                        if ml_lo <= threshold:
                            agg_prob += prob
            except ValueError:
                continue

        # Standard range bucket: "68-69"
        elif "-" in kalshi_label:
            parts = kalshi_label.split("-")
            if len(parts) != 2:
                continue
            try:
                lo = int(parts[0])
                hi = int(parts[1])
            except ValueError:
                continue
            # Kalshi "68-69" covers integer temps 68 and 69
            # → sum our ML "68-69" (high=68) + "69-70" (high=69)
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
    Compute bet signal based on ML model confidence AND edge over market.

    Returns (signal, edge) where:
      signal: "STRONG_BET" / "BET" / "LEAN" / "SKIP"
      edge: model confidence - market probability
    """
    market_prob = market_probs.get(ml_bucket, 0.0)
    edge = ml_confidence - market_prob

    if ml_confidence >= 0.65 and edge >= 0.20:
        signal = "STRONG_BET"
    elif ml_confidence >= 0.55 and edge >= 0.15:
        signal = "BET"
    elif ml_confidence >= 0.45 and edge >= 0.10:
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
    v2_temp_model_check, _, v2_classifier_check, _ = _load_v2_models()

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

    # Base forecast: prefer AccuWeather (lower MAE historically) but guard against
    # stale data.  When NWS and AccuWeather disagree by >8°F, AccuWeather is likely
    # outdated (e.g., yesterday's warm weather not yet revised).  Fall back to NWS
    # in that case — a small bias mismatch is far better than a 15°F wrong anchor.
    if has_accu:
        spread = abs(features["nws_last"] - features["accu_last"])
        if spread > 8.0:
            print(f"⚠️ NWS-AccuWeather spread={spread:.0f}°F > 8°F — "
                  f"AccuWeather likely stale ({features['accu_last']:.0f}°F vs NWS {features['nws_last']:.0f}°F). "
                  f"Using NWS as base.")
            base = features["nws_last"]
            base_src = "nws_last (accu stale)"
        else:
            base = features["accu_last"]
            base_src = "accu_last"
    else:
        base = features["nws_last"]
        base_src = "nws_last"

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
    v2_temp_model, v2_bucket_info, v2_classifier, v2_atm_predictor = _load_v2_models()
    if v2_classifier is not None:
        try:
            # Fetch atmospheric features
            atm_features = _fetch_atmospheric_features(target_date_iso)

            # Merge atmospheric features into the feature dict
            v2_features = dict(features)
            v2_features.update(atm_features)

            # Run atmospheric predictor (first-stage model) if available
            if v2_atm_predictor is not None:
                try:
                    atm_model_data = v2_atm_predictor
                    atm_model = atm_model_data["model"]
                    atm_input_cols = atm_model_data["features"]
                    atm_input = pd.DataFrame([v2_features])
                    for col in atm_input_cols:
                        if col not in atm_input.columns:
                            atm_input[col] = np.nan
                    atm_pred = float(atm_model.predict(atm_input[atm_input_cols])[0])
                    v2_features["atm_predicted_high"] = atm_pred
                    v2_features["atm_vs_forecast_diff"] = features["nws_last"] - atm_pred
                    print(f"🌍 Atmospheric predictor: {atm_pred:.1f}°F "
                          f"(NWS diff: {features['nws_last'] - atm_pred:+.1f}°F)")
                except Exception as e:
                    print(f"⚠️ Atmospheric predictor failed: {e}")
                    v2_features["atm_predicted_high"] = np.nan
                    v2_features["atm_vs_forecast_diff"] = np.nan
            else:
                v2_features["atm_predicted_high"] = np.nan
                v2_features["atm_vs_forecast_diff"] = np.nan

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

            # v2 center temp: prefer atmospheric predictor (1,278 days, knows
            # spring) over regression model (239 days, 15 spring days).
            # Fall back to regression or forecast average when atm unavailable.
            atm_pred_val = v2_features.get("atm_predicted_high")
            has_atm = (atm_pred_val is not None
                       and not (isinstance(atm_pred_val, float) and math.isnan(atm_pred_val)))

            if v2_temp_model is not None:
                v2_bias = float(v2_temp_model.predict(X_v2)[0])
                v2_temp = base + v2_bias
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"(regression: base={base:.0f} + bias={v2_bias:+.1f})")
            else:
                # No atm predictor and no regression — use forecast average
                all_forecasts = [features["nws_last"]]
                if has_accu and not np.isnan(features["accu_last"]):
                    all_forecasts.append(features["accu_last"])
                v2_temp = float(np.mean(all_forecasts))
                accu_note = f", AccuWx={features['accu_last']:.0f}" if has_accu else ""
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"(forecast avg: NWS={features['nws_last']:.0f}{accu_note})")

            # Exceedance check: if observed temp already exceeded forecast,
            # shift center up (physics-based, not market-based)
            obs_high, obs_hour = _fetch_observed_high_so_far(target_date_iso)
            if obs_high is not None:
                v2_temp = _adjust_center_for_exceedance(v2_temp, obs_high, obs_hour)

            # v2 classifier bucket prediction
            # Use 15 candidates (±7) to cover full Kalshi range and handle
            # source disagreements (e.g., AccuWeather says 77, NWS says 88)
            bucket_probs = v2_classifier.predict_bucket_probs(
                features=v2_features,
                center_temp=v2_temp,
                accu_last=features.get("accu_last") if has_accu else None,
                nws_last=features.get("nws_last"),
                n_candidates=15,
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

_LOCK_NOT_FOUND = "not_found"
_LOCK_ERROR = "error"

def _fetch_existing_prediction(target_date_iso: str):
    """
    Check if a prediction already exists in Supabase for this target date.

    Uses a lead-agnostic idempotency key so the first prediction for a given
    target date wins (tomorrow's prediction locks out today's for the same date).

    Returns:
      dict  – existing prediction row (lock the ML fields)
      _LOCK_NOT_FOUND – no row exists yet (compute fresh ML prediction)
      _LOCK_ERROR – network/auth error (skip ML to avoid overwriting)
    """
    try:
        endpoint, key = _sb_endpoint()
        idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{target_date_iso}"
        url = (f"{endpoint}?idempotency_key=eq.{idem_key}"
               f"&select=ml_f,ml_bucket,ml_confidence,ml_bucket_probs,ml_version,kalshi_market_snapshot")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        if rows and rows[0].get("ml_f") is not None:
            # Allow force-recompute for specific dates (e.g., after code fix)
            force_dates = os.environ.get("FORCE_RECOMPUTE_DATES", "")
            if target_date_iso in force_dates.split(","):
                print(f"🔓 Force-recompute enabled for {target_date_iso}")
                return _LOCK_NOT_FOUND
            return rows[0]
        return _LOCK_NOT_FOUND
    except Exception as e:
        print(f"⚠️ Could not check existing prediction: {e}")
        return _LOCK_ERROR


def score_yesterday_prediction(rows: list[dict]) -> None:
    """
    Score yesterday's ML prediction against the actual high.
    Updates the prediction_logs row with ml_result ('WIN' or 'MISS')
    and ml_actual_high.
    """
    today = today_nyc()
    yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get yesterday's actual high from CSV
    actual_high = None
    for r in rows:
        if (r.get("forecast_or_actual") == "actual" and
            r.get("cli_date") == yesterday_iso):
            actual_high = _float_or_none(r.get("actual_high"))
            if actual_high is not None:
                break
    if actual_high is None:
        return  # No actual yet — nothing to score

    # Fetch yesterday's prediction from Supabase
    try:
        endpoint, key = _sb_endpoint()
        idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{yesterday_iso}"
        url = (f"{endpoint}?idempotency_key=eq.{idem_key}"
               f"&select=ml_bucket,ml_f,ml_result")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            pred_rows = json.loads(resp.read().decode("utf-8"))

        if not pred_rows or not pred_rows[0].get("ml_bucket"):
            return
        pred = pred_rows[0]

        # Already scored?
        if pred.get("ml_result"):
            return

        # Check if actual falls in the ML bucket
        ml_bucket = pred["ml_bucket"]
        parts = ml_bucket.split("-") if "-" in ml_bucket else None
        if parts and len(parts) == 2:
            try:
                lo, hi = int(parts[0]), int(parts[1])
                actual_int = int(round(actual_high))
                is_win = lo <= actual_int <= hi
            except ValueError:
                is_win = False
        else:
            is_win = False

        result = "WIN" if is_win else "MISS"

        # Update prediction_logs with result
        patch = json.dumps({
            "ml_result": result,
            "ml_actual_high": actual_high,
        }).encode("utf-8")
        patch_url = f"{endpoint}?idempotency_key=eq.{idem_key}"
        patch_req = urllib.request.Request(
            patch_url, data=patch, method="PATCH",
            headers={
                "Content-Type": "application/json",
                "apikey": key, "Authorization": f"Bearer {key}",
                "Prefer": "return=minimal",
            },
        )
        with urllib.request.urlopen(patch_req, timeout=10) as resp:
            _ = resp.read()
        print(f"{'✅' if is_win else '❌'} Yesterday ({yesterday_iso}): ML={ml_bucket}, "
              f"Actual={actual_high}°F → {result}")

    except Exception as e:
        print(f"⚠️ Could not score yesterday's prediction: {e}")


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

    # Check if ML prediction is already locked for today.
    # The ML prediction should be stable — only computed once per day.
    # Subsequent runs only refresh Kalshi market data + bet signals.
    existing = _fetch_existing_prediction(target_date_iso)
    if isinstance(existing, dict):
        print(f"🔒 ML prediction already locked: {existing['ml_f']}°F → {existing.get('ml_bucket')}")
        ml = {
            "ml_f": existing["ml_f"],
            "ml_bucket": existing["ml_bucket"],
            "ml_confidence": existing["ml_confidence"],
            "ml_bucket_probs": existing.get("ml_bucket_probs"),
            "ml_version": existing.get("ml_version"),
        }
    elif existing == _LOCK_ERROR:
        # Network error checking lock — skip ML to avoid overwriting a locked prediction
        print("⚠️ Supabase unreachable — skipping ML recomputation to protect locked prediction")
        ml = None
    else:
        # First run of the day (_LOCK_NOT_FOUND) — compute ML prediction
        ml = _compute_ml_prediction(rows, target_date_iso)

    if bcp is None and ml is None:
        print("⏭️ today_for_today: no BCP data and no ML prediction available."); return

    ts = now_nyc().isoformat()
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{target_date_iso}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": target_date_iso,
        "lead_used": "today_for_today",
        "model_name": MODEL_VERSION,
        "prediction_value": float(f"{bcp:.1f}") if bcp is not None else (ml["ml_f"] if ml else 0.0),
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

    is_locked = isinstance(existing, dict)

    # Map ML prediction → Kalshi's actual bucket structure for today
    if ml and market_probs:
        if is_locked:
            # Prediction is locked — do NOT re-map the bucket or apply observed
            # floor.  Only refresh the bet signal (ML confidence vs live market).
            print(f"🔒 Bucket locked: {payload.get('ml_bucket')} — refreshing bet signal only")
            signal, edge = _compute_bet_signal(
                payload.get("ml_confidence", 0), payload.get("ml_bucket", ""), market_probs
            )
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
        else:
            # First run — map ML probs to Kalshi buckets
            direct_bucket = _find_kalshi_bucket_for_temp(ml["ml_f"], market_probs)
            if direct_bucket:
                print(f"🎯 Direct map: {ml['ml_f']:.1f}°F → Kalshi bucket '{direct_bucket}'")

            if ml.get("ml_bucket_probs"):
                raw_probs = json.loads(ml["ml_bucket_probs"]) if isinstance(ml["ml_bucket_probs"], str) else ml["ml_bucket_probs"]

                kalshi_bucket, kalshi_conf, kalshi_aligned = _map_ml_to_kalshi_buckets(raw_probs, market_probs)
                if kalshi_bucket:
                    payload["ml_bucket"] = kalshi_bucket
                    payload["ml_confidence"] = kalshi_conf
                    print(f"🎯 Kalshi prob-aligned bucket: {kalshi_bucket} ({kalshi_conf:.0%})")

                    if direct_bucket and kalshi_bucket != direct_bucket:
                        print(f"ℹ️ Direct map ({direct_bucket}) differs from prob-aligned ({kalshi_bucket}) "
                              f"— keeping prob-aligned (higher expected accuracy)")

                    signal, edge = _compute_bet_signal(
                        payload["ml_confidence"], payload["ml_bucket"], market_probs
                    )
                    payload["bet_signal"] = signal
                    payload["ml_edge"] = edge
                    print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
            elif direct_bucket:
                payload["ml_bucket"] = direct_bucket
                signal, edge = _compute_bet_signal(
                    ml["ml_confidence"], direct_bucket, market_probs
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

    # Check if ML prediction is already locked for tomorrow.
    # Only lock when Kalshi market data is available — otherwise keep recomputing
    # so the prediction gets mapped to real Kalshi buckets.
    existing = _fetch_existing_prediction(tomorrow_iso)
    tomorrow_market_probs = _fetch_kalshi_market_probs(tomorrow_iso)
    if isinstance(existing, dict):
        if existing.get("kalshi_market_snapshot") or tomorrow_market_probs:
            # Locked WITH Kalshi data — keep it
            print(f"🔒 ML prediction already locked: {existing['ml_f']}°F → {existing.get('ml_bucket')}")
            ml = {
                "ml_f": existing["ml_f"],
                "ml_bucket": existing["ml_bucket"],
                "ml_confidence": existing["ml_confidence"],
                "ml_bucket_probs": existing.get("ml_bucket_probs"),
                "ml_version": existing.get("ml_version"),
            }
        else:
            # Locked WITHOUT Kalshi data — recompute so we can map to real buckets
            print(f"🔄 Recomputing tomorrow's prediction — previous had no Kalshi market data")
            ml = _compute_ml_prediction(rows, tomorrow_iso)
    elif existing == _LOCK_ERROR:
        print("⚠️ Supabase unreachable — skipping ML recomputation to protect locked prediction")
        ml = None
    else:
        # First run — compute ML prediction
        ml = _compute_ml_prediction(rows, tomorrow_iso)

    if bcp_tm is None and ml is None:
        print("⏭️ today_for_tomorrow: no BCP data and no ML prediction available."); return

    ts = now_nyc().isoformat()
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{tomorrow_iso}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": tomorrow_iso,
        "lead_used": "today_for_tomorrow",
        "model_name": MODEL_VERSION,
        "prediction_value": bcp_tm if bcp_tm is not None else (ml["ml_f"] if ml else None),
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

    # Use already-fetched Kalshi market odds (from lock check above)
    market_probs = tomorrow_market_probs
    if market_probs:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    is_locked_tm = isinstance(existing, dict)

    # Map ML prediction → Kalshi's actual bucket structure for tomorrow
    if ml and market_probs:
        if is_locked_tm:
            print(f"🔒 Bucket locked: {payload.get('ml_bucket')} — refreshing bet signal only")
            signal, edge = _compute_bet_signal(
                payload.get("ml_confidence", 0), payload.get("ml_bucket", ""), market_probs
            )
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
        else:
            direct_bucket = _find_kalshi_bucket_for_temp(ml["ml_f"], market_probs)
            if direct_bucket:
                print(f"🎯 Direct map: {ml['ml_f']:.1f}°F → Kalshi bucket '{direct_bucket}'")

            if ml.get("ml_bucket_probs"):
                raw_probs = json.loads(ml["ml_bucket_probs"]) if isinstance(ml["ml_bucket_probs"], str) else ml["ml_bucket_probs"]
                kalshi_bucket, kalshi_conf, kalshi_aligned = _map_ml_to_kalshi_buckets(raw_probs, market_probs)
                if kalshi_bucket:
                    payload["ml_bucket"] = kalshi_bucket
                    payload["ml_confidence"] = kalshi_conf
                    print(f"🎯 Kalshi prob-aligned bucket: {kalshi_bucket} ({kalshi_conf:.0%})")

                    if direct_bucket and kalshi_bucket != direct_bucket:
                        print(f"ℹ️ Direct map ({direct_bucket}) differs from prob-aligned ({kalshi_bucket}) "
                              f"— keeping prob-aligned (higher expected accuracy)")

                    signal, edge = _compute_bet_signal(
                        payload["ml_confidence"], payload["ml_bucket"], market_probs
                    )
                    payload["bet_signal"] = signal
                    payload["ml_edge"] = edge
                    print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")
            elif direct_bucket:
                payload["ml_bucket"] = direct_bucket
                signal, edge = _compute_bet_signal(
                    ml["ml_confidence"], direct_bucket, market_probs
                )
                payload["bet_signal"] = signal
                payload["ml_edge"] = edge
                print(f"🎯 Bet signal: {signal} (edge={edge:+.0%})")

    supabase_upsert(payload)

def write_both_snapshots() -> None:
    rows = _read_all_rows()
    # Score yesterday's prediction against actual high
    try: score_yesterday_prediction(rows)
    except Exception as e: print("⚠️ score_yesterday_prediction failed:", e)
    try: write_today_for_today()
    except Exception as e: print("⚠️ write_today_for_today failed:", e)
    try: write_today_for_tomorrow()
    except Exception as e: print("⚠️ write_today_for_tomorrow failed:", e)
    # Compute and store server-side ensemble weights
    try: compute_ensemble_weights()
    except Exception as e: print("⚠️ compute_ensemble_weights failed:", e)


# ---------------------------------------------------------------------------
# Server-side ensemble weight learning (Hedge algorithm)
# ---------------------------------------------------------------------------

def _fetch_ml_predictions_history() -> dict:
    """
    Fetch all scored ML predictions from Supabase for the current city.
    Returns {target_date: ml_f} for rows that have ml_f and ml_actual_high.
    """
    try:
        endpoint, key = _sb_endpoint()
        url = (f"{endpoint}?city=eq.{_CITY_KEY}"
               f"&ml_f=not.is.null"
               f"&ml_actual_high=not.is.null"
               f"&select=target_date,ml_f")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        result = {}
        for r in rows:
            td = r.get("target_date")
            ml_f = r.get("ml_f")
            if td and ml_f is not None:
                result[td] = float(ml_f)
        print(f"📊 Loaded {len(result)} ML prediction history rows from Supabase")
        return result
    except Exception as e:
        print(f"⚠️ Could not fetch ML prediction history: {e}")
        return {}


def compute_ensemble_weights() -> Optional[dict]:
    """
    Compute server-side ensemble weights using an exponentially-weighted
    Hedge algorithm over all historical days with actuals.

    Candidates (matching the dashboard):
      - nws_last: latest NWS forecast for the target date
      - accu_last: latest AccuWeather forecast
      - nws_mean: mean of all NWS pre-high forecasts (like series_rep)
      - ml_prediction: ML bias-corrected prediction from Supabase

    Stores result in Supabase prediction_logs with a special idempotency key.
    Returns the weight dict or None on failure.
    """
    rows, _ = _read_all_rows(include_accu=True)

    # Fetch ML predictions from Supabase
    ml_history = _fetch_ml_predictions_history()

    # Group rows by target date
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        d = (r.get("cli_date") if r.get("forecast_or_actual") == "actual"
             else r.get("target_date"))
        if d:
            by_date.setdefault(d, []).append(r)

    # Hedge algorithm parameters (match client-side)
    ETA = 0.05          # learning rate
    MIX_GAMMA = 0.02    # uniform mixing to prevent collapse
    DECAY = 0.995       # daily weight decay (recency)
    CANDIDATE_NAMES = ["nws_last", "accu_last", "nws_mean", "ml_prediction"]

    # Initialize uniform weights
    n_cands = len(CANDIDATE_NAMES)
    weights = {name: 1.0 / n_cands for name in CANDIDATE_NAMES}

    # Track bias via EWMA
    bias_ewma = 0.0
    bias_alpha = 1.0 - 0.5 ** (1.0 / 14.0)  # 14-day half-life
    rmse_var = 4.0  # initial variance (2^2)
    rmse_alpha = 1.0 - 0.5 ** (1.0 / 21.0)  # 21-day half-life

    num_days = 0

    from nws_auto_logger import _minutes_from_hhmm_ampm, _minutes_from_forecast_time_cell

    for d in sorted(by_date.keys()):
        rs = by_date[d]
        # Find actual high
        actual_high = None
        high_time_str = None
        for x in rs:
            if x.get("forecast_or_actual") == "actual":
                actual_high = _float_or_none(x.get("actual_high"))
                high_time_str = (x.get("high_time") or "").strip()
                if actual_high is not None:
                    break
        if actual_high is None:
            continue

        # Parse high time for pre-high filtering
        high_min = _minutes_from_hhmm_ampm(high_time_str) if high_time_str else None

        # Collect NWS forecasts (pre-high only)
        nws_vals = []
        nws_last_val = None
        for x in rs:
            if x.get("forecast_or_actual") != "forecast":
                continue
            src = (x.get("source") or "").lower()
            if src == "accuweather":
                continue
            ph = _float_or_none(x.get("predicted_high"))
            if ph is None:
                continue
            if high_min is not None:
                fc_min = _minutes_from_forecast_time_cell(x.get("forecast_time") or "")
                if fc_min is not None and fc_min > high_min:
                    continue
            nws_vals.append(ph)
        if nws_vals:
            nws_last_val = nws_vals[-1]  # last pre-high forecast

        # Collect AccuWeather forecasts
        accu_last_val = None
        for x in rs:
            if x.get("forecast_or_actual") != "forecast":
                continue
            src = (x.get("source") or "").lower()
            if src != "accuweather":
                continue
            ph = _float_or_none(x.get("predicted_high"))
            if ph is not None:
                accu_last_val = ph  # take last one

        # NWS mean
        nws_mean_val = (sum(nws_vals) / len(nws_vals)) if nws_vals else None

        # ML prediction
        ml_val = ml_history.get(d)

        # Build candidate dict for this day
        candidates = {}
        if nws_last_val is not None:
            candidates["nws_last"] = nws_last_val
        if accu_last_val is not None:
            candidates["accu_last"] = accu_last_val
        if nws_mean_val is not None:
            candidates["nws_mean"] = nws_mean_val
        if ml_val is not None:
            candidates["ml_prediction"] = ml_val

        if len(candidates) < 2:
            continue  # need at least 2 candidates

        # Decay weights (recency preference)
        for name in weights:
            weights[name] *= DECAY

        # Uniform mixing to prevent collapse
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] = (1.0 - MIX_GAMMA) * (weights[name] / total) + MIX_GAMMA / n_cands

        # Compute weighted ensemble prediction (pre-bias)
        active_names = [n for n in CANDIDATE_NAMES if n in candidates]
        denom = sum(weights[n] for n in active_names)
        if denom <= 0:
            continue
        y0 = sum(weights[n] * candidates[n] for n in active_names) / denom
        yhat = y0 + bias_ewma
        err = actual_high - yhat

        # Hedge update: penalize each candidate by its absolute error
        for name in active_names:
            loss = abs(candidates[name] - actual_high)
            weights[name] *= math.exp(-ETA * loss)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] /= total

        # Update bias EWMA (clipped to +/-6)
        clipped_err = max(-6.0, min(6.0, err))
        bias_ewma = (1.0 - bias_alpha) * bias_ewma + bias_alpha * clipped_err

        # Update RMSE EWMA
        sq = min(36.0, err * err)
        rmse_var = (1.0 - rmse_alpha) * rmse_var + rmse_alpha * sq

        num_days += 1

    if num_days < 5:
        print(f"⏭️ Ensemble weights: only {num_days} days with data, need >= 5")
        return None

    rmse = math.sqrt(max(0, rmse_var))

    # Final normalization
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 6) for k, v in weights.items()}

    result = {
        "weights": weights,
        "bias": round(bias_ewma, 4),
        "rmse": round(rmse, 3),
        "computed_on": today_nyc().isoformat(),
        "num_days": num_days,
    }

    print(f"📊 Ensemble weights ({_CITY_KEY}, {num_days} days):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"   {name}: {w:.1%}")
    print(f"   bias: {bias_ewma:+.3f}, RMSE: {rmse:.2f}")

    # Store in Supabase with special idempotency key
    ts = now_nyc().isoformat()
    idem_key = f"{_CITY_KEY}:ensemble_weights"
    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": "9999-12-31",
        "lead_used": "ensemble_weights",
        "model_name": "hedge_server",
        "prediction_value": 0,
        "version": "hedge_v1",
        "source_card": "ensemble_weights",
        "city": _CITY_KEY,
        "recommendation": json.dumps(result),
    }
    supabase_upsert(payload)

    return result


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
