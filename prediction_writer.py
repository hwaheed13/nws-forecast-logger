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
    FEATURE_COLS, FEATURE_COLS_V2, FEATURE_COLS_V4, ACCU_NWS_FALLBACK,
    ATM_PREDICTOR_INPUT_COLS, OBSERVATION_COLS, derive_bucket_probabilities,
)

MODEL_VERSION = os.environ.get("PREDICTION_MODEL_VERSION", "bcp_v1")


def _ts_hour(ts_str: str) -> int | None:
    """Extract local hour (0-23) from a CSV timestamp string.

    Handles formats produced by nws_auto_logger.py, e.g.:
      '2026-04-06 08:30:00 EDT'   → 8
      '2026-04-06T14:45:00'       → 14
      '2026-04-06 14:45:00'       → 14

    Returns None on any parse failure (treated as unknown, NOT < 9).
    """
    try:
        s = str(ts_str).strip()
        # Strip tz suffix (EDT, EST, PDT, PST, ET, PT, UTC)
        for sfx in (" EDT", " EST", " PDT", " PST", " ET", " PT", " UTC"):
            if s.endswith(sfx):
                s = s[: -len(sfx)].strip()
                break
        # Normalize 'T' separator → space
        s = s.replace("T", " ")
        # Hour lives at position 11-12 of 'YYYY-MM-DD HH:MM:SS'
        return int(s[11:13])
    except (ValueError, IndexError, TypeError):
        return None

# Kalshi API base URL (public, no auth needed)
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Module-level city key — set by _cli() before write functions run
_CITY_KEY = "nyc"

# D0 prediction cutoff hour (local time per city, expressed in ET for NYC,
# PT for LAX). After this hour we freeze the ML prediction to prevent
# AccuWeather retroactively echoing the observed high and intraday curve
# features from leaking actual temperatures into the forecast.
_D0_CUTOFF_HOUR_LOCAL = {
    "nyc": 14,  # 2pm ET  — NYC high typically peaks 1–3pm
    "lax": 14,  # 2pm PT  — LA  high typically peaks 2–4pm PT
}

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


def _load_v4_models():
    """Load v4 models: regression + bucket_info + classifier + atm_predictor (cached).
    v4 = v2 architecture + 12 observation features (84 total)."""
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v4_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v4_bucket_info"] = None
        _ML_MODEL_CACHE[f"{prefix}v4_classifier"] = None

        try:
            with open(f"{prefix}temp_model_v4.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v4 regression model (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v4 regression load error: {e}")

        try:
            with open(f"{prefix}bucket_model_v4.pkl", "rb") as f:
                _ML_MODEL_CACHE[f"{prefix}v4_bucket_info"] = pickle.load(f)
        except FileNotFoundError:
            pass

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bucket_classifier_v4.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v4_classifier"] = BucketClassifier.load(
                    f"{prefix}bucket_classifier_v4.pkl"
                )
                print(f"✅ Loaded v4 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v4 classifier load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v4_bucket_info"),
        _ML_MODEL_CACHE.get(f"{prefix}v4_classifier"),
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


def _fetch_mos_forecast(target_date_iso: str) -> Optional[float]:
    """
    Fetch MOS (Model Output Statistics) max temperature forecast from NWS MEX product.

    The MEX product contains GFS MOS guidance with X/N (max/min) temperature rows.
    Parses the text product to extract the max temperature for the target date.

    Returns the MOS max temp as a float, or None on failure.
    """
    import re

    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        issuedby = cfg.get("cli_issuedby", "NYC")

        url = f"https://forecast.weather.gov/product.php?site=NWS&issuedby={issuedby}&product=MEX"
        req = urllib.request.Request(url, headers={
            "User-Agent": "nws-forecast-logger/1.0",
            "Accept": "text/html",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Extract the pre-formatted text content from the HTML
        pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
        if not pre_match:
            print(f"  MOS: no <pre> block found in MEX product")
            return None
        text = pre_match.group(1)

        # Parse target date components
        target_dt = datetime.strptime(target_date_iso, "%Y-%m-%d")
        target_month_abbr = target_dt.strftime("%b").upper()  # e.g., "MAR"
        target_day = target_dt.day

        # Find the DT line that contains our target date
        # Format: "DT /MAR  26            /MAR  27            /"
        # or similar with the month abbreviation and day number
        lines = text.split("\n")

        # Strategy: find the DT header row with our date, then find X/N row
        mos_max = None
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()

            # Look for DT row containing our target date
            if not line_upper.startswith("DT"):
                continue

            # Check if our target date's month+day appears in this DT line
            # Pattern: /MON  DD where MON is 3-letter month abbreviation
            dt_pattern = rf"/{target_month_abbr}\s+{target_day}\b"
            dt_matches = list(re.finditer(dt_pattern, line_upper))
            if not dt_matches:
                continue

            # Found our date in the DT line. Now find the X/N row below it.
            # The X/N row has the max (X) and min (N) temperatures.
            for j in range(i + 1, min(i + 10, len(lines))):
                xn_line = lines[j].strip()
                if xn_line.startswith("X/N"):
                    # Extract numbers from the X/N line
                    # The values are space-separated after "X/N"
                    xn_values = re.findall(r'-?\d+', xn_line[3:])
                    if xn_values:
                        # Determine which column corresponds to our date.
                        # The DT line has date blocks separated by "/".
                        # Count which date block our match is in.
                        dt_content = line_upper
                        # Split by "/" to get date groups
                        date_groups = [g.strip() for g in dt_content.split("/") if g.strip()]

                        # Find which group index contains our target date
                        target_group_idx = None
                        for gi, group in enumerate(date_groups):
                            if re.search(rf"{target_month_abbr}\s+{target_day}\b", group):
                                target_group_idx = gi
                                break

                        if target_group_idx is not None and target_group_idx < len(date_groups):
                            # In X/N row, values alternate: max for day1, min for day1, max for day2, min for day2...
                            # But the DT header starts with "DT" then date groups.
                            # The first date group (index 0 after removing "DT") corresponds to X/N values.
                            # First number is max for first date, second is min for between dates, etc.
                            # Adjust: DT line has "DT" prefix in group 0
                            adj_idx = target_group_idx
                            if date_groups[0].startswith("DT"):
                                adj_idx = target_group_idx  # DT is part of group 0

                            # The X/N values: first value = max of first date period,
                            # second value = min of overnight, etc.
                            # For the Nth date group (0-indexed), max is at index N*2
                            # and min is at index N*2+1
                            val_idx = adj_idx  # max temp index
                            if val_idx < len(xn_values):
                                mos_max = float(xn_values[val_idx])
                                print(f"🌡️ MOS forecast: {mos_max:.0f}°F max for {target_date_iso}")
                                return mos_max

                    # If we couldn't parse columns, try simpler: first number is the max
                    if xn_values:
                        mos_max = float(xn_values[0])
                        print(f"🌡️ MOS forecast: {mos_max:.0f}°F max for {target_date_iso} (first value)")
                        return mos_max
                    break
            break

        if mos_max is None:
            print(f"  MOS: could not find max temp for {target_date_iso} in MEX product")
        return mos_max

    except Exception as e:
        print(f"⚠️ MOS forecast fetch failed: {e}")
        return None


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
    rows: list[dict], target_date_iso: str,
    prefetched_atm: Optional[dict] = None,
) -> Optional[dict]:
    """
    Compute ML bias-corrected prediction for *target_date_iso* using the
    same 30 features as train_models.py (NWS stats, AccuWeather stats,
    cross-source, rolling bias, temporal).

    prefetched_atm: if provided, skip the _fetch_atmospheric_features() call
        and use this dict instead. Used when the caller already fetched live
        atmospheric data for the intraday shift comparison, to avoid a redundant
        set of 3 Open-Meteo API calls (~15-20 seconds each run).

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

    # rolling_ml_error_7d: mean(actual - ml_f) over last 7 days
    # Captures systematic ML model bias in recent predictions
    try:
        endpoint, key = _sb_endpoint()
        ml_hist_url = (
            f"{endpoint}"
            f"?city=eq.{_CITY_KEY}"
            f"&lead_used=in.(today_for_today,D0)"
            f"&ml_f=not.is.null"
            f"&target_date=lt.{target_date_iso}"
            f"&order=target_date.desc"
            f"&limit=14"
            f"&select=target_date,ml_f"
        )
        ml_hist_req = urllib.request.Request(
            ml_hist_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(ml_hist_req, timeout=10) as _resp:
            ml_hist_rows = json.loads(_resp.read().decode("utf-8"))
        ml_errors = []
        actuals_by_date = {}
        for r in rows:
            if r.get("forecast_or_actual") == "actual":
                d = r.get("cli_date") or r.get("target_date")
                ah = _float_or_none(r.get("actual_high"))
                if d and ah is not None:
                    actuals_by_date[d] = ah
        for mrow in ml_hist_rows:
            d = str(mrow.get("target_date", ""))[:10]
            mf = _float_or_none(mrow.get("ml_f"))
            ah = actuals_by_date.get(d)
            if mf is not None and ah is not None:
                ml_errors.append(ah - mf)
            if len(ml_errors) >= 7:
                break
        features["rolling_ml_error_7d"] = float(np.mean(ml_errors)) if ml_errors else 0.0
    except Exception:
        features["rolling_ml_error_7d"] = 0.0

    # --- 7b. Overnight carryover detection features ---
    # prev_day_high: yesterday's actual high from CSV rows
    prev_day_high = None
    try:
        prev_dt = datetime.strptime(target_date_iso, "%Y-%m-%d") - timedelta(days=1)
        prev_date_str = prev_dt.strftime("%Y-%m-%d")
        for r in rows:
            if r.get("forecast_or_actual") == "actual" and r.get("cli_date") == prev_date_str:
                ah = _float_or_none(r.get("actual_high"))
                if ah is not None:
                    prev_day_high = ah
                    break
    except (ValueError, TypeError):
        pass

    # If no actual yet (CLI/DSM not recorded) and we're predicting tomorrow,
    # use today's obs_max_so_far as a real-time proxy.
    # After 12 PM ET the daily max is usually established; before that we
    # use the latest NWS forecast for today as a conservative fallback.
    if prev_day_high is None:
        try:
            today_iso = today_nyc().isoformat()
            prev_dt2 = datetime.strptime(target_date_iso, "%Y-%m-%d") - timedelta(days=1)
            if prev_dt2.strftime("%Y-%m-%d") == today_iso:
                # We're predicting tomorrow; today's actual not yet recorded
                today_obs = _query_supabase_observations(today_iso)
                if today_obs:
                    max_temps = [r["temp_f"] for r in today_obs if r.get("temp_f") is not None]
                    if max_temps:
                        obs_max = max(max_temps)
                        current_hour = now_nyc().hour
                        if current_hour >= 14:
                            # After 2 PM: obs max is likely the final high
                            prev_day_high = obs_max
                            print(f"  📍 prev_day_high proxy (obs_max after 2PM): {obs_max:.1f}°F")
                        elif current_hour >= 9:
                            # 9 AM–2 PM: use obs_max as lower bound proxy (high still building)
                            # Blend with NWS forecast for today as upper estimate
                            today_nws = _latest_forecast(rows, today_iso, source=None)
                            if today_nws is not None:
                                prev_day_high = max(obs_max, float(today_nws) * 0.5 + obs_max * 0.5)
                            else:
                                prev_day_high = obs_max
                            print(f"  📍 prev_day_high proxy (obs_max+NWS blend 9-2PM): {prev_day_high:.1f}°F")
        except Exception:
            pass

    features["prev_day_high"] = prev_day_high if prev_day_high is not None else np.nan
    # prev_day_temp_drop: large positive = potential overnight carryover
    if prev_day_high is not None:
        features["prev_day_temp_drop"] = prev_day_high - features["nws_last"]
    else:
        features["prev_day_temp_drop"] = np.nan
    # midnight_temp: filled from atmospheric features later (Open-Meteo live data)
    features["midnight_temp"] = np.nan

    # --- 8. Data availability flag ---
    features["has_accu_data"] = int(has_accu)

    # --- 8b. MOS max temp — set to NaN initially, filled in v2 path ---
    features["mos_max_temp"] = np.nan

    # --- 8c. Intraday forecast revision features ---
    # How much did each agency revise its forecast after 9 AM local time?
    # For D0 predictions (same-day), nws_fc contains both pre- and post-9am rows.
    # For D1 predictions (next-day), forecasts were issued yesterday — all rows
    # are from a prior calendar day so hour-of-day is irrelevant; delta = NaN.
    #
    # nws_fc is already sorted chronologically (ts strings sort correctly for
    # the 'YYYY-MM-DD HH:MM:SS' format produced by nws_auto_logger.py).
    nws_before_9 = [(ts, v) for ts, v in nws_fc if (_ts_hour(ts) or 25) < 9]
    if nws_before_9 and nws_fc:
        features["nws_post_9am_delta"] = features["nws_last"] - nws_before_9[-1][1]
    else:
        features["nws_post_9am_delta"] = np.nan

    accu_before_9 = [(ts, v) for ts, v in accu_fc if (_ts_hour(ts) or 25) < 9]
    if accu_before_9 and accu_fc:
        features["accu_post_9am_delta"] = features["accu_last"] - accu_before_9[-1][1]
    else:
        features["accu_post_9am_delta"] = np.nan

    # --- 9. Build DataFrame, fill NaN AccuWeather with NWS fallbacks ---
    # Use model's own feature list (feature_names_in_) when available so that
    # the inference feature set always matches what the model was trained on,
    # regardless of changes to FEATURE_COLS in model_config.py.
    X = pd.DataFrame([features])
    _v1_feature_cols = FEATURE_COLS
    if temp_model is not None and hasattr(temp_model, "feature_names_in_"):
        _v1_feature_cols = list(temp_model.feature_names_in_)
    for col in _v1_feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[_v1_feature_cols]
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

    # --- 11. Try v4 models first (v2 + observation features), fall back to v2 ---
    v4_regressor, v4_bucket_info, v4_classifier = _load_v4_models()
    v2_temp_model, v2_bucket_info, v2_classifier, v2_atm_predictor = _load_v2_models()

    # Select best available model
    use_v4 = v4_classifier is not None
    active_classifier = v4_classifier if use_v4 else v2_classifier
    active_regressor = v4_regressor if use_v4 else v2_temp_model
    active_bucket_info = v4_bucket_info if use_v4 else v2_bucket_info
    # Use the regressor's own feature list when available (self-consistent with training),
    # falling back to FEATURE_COLS_V4/V2 from model_config for new models without saved names.
    if active_regressor is not None and hasattr(active_regressor, "feature_names_in_"):
        active_feature_cols = list(active_regressor.feature_names_in_)
    elif use_v4:
        active_feature_cols = FEATURE_COLS_V4
    else:
        active_feature_cols = FEATURE_COLS_V2
    active_version = "v4_observation_features" if use_v4 else "v2_atm_classifier"

    if active_classifier is not None:
        try:
            # Fetch atmospheric features — use prefetched data when available
            # (caller already fetched for intraday shift comparison, avoid double call)
            if prefetched_atm is not None and len(prefetched_atm) > 0:
                atm_features = prefetched_atm
                print(f"🌤️ Using prefetched atmospheric features ({len(atm_features)} keys)")
            else:
                atm_features = _fetch_atmospheric_features(target_date_iso)

            # Merge atmospheric features into the feature dict
            v2_features = dict(features)
            v2_features.update(atm_features)

            # Fetch MOS forecast
            mos_temp = _fetch_mos_forecast(target_date_iso)
            v2_features["mos_max_temp"] = float(mos_temp) if mos_temp is not None else np.nan

            # Fetch observation features (real-time NWS station data)
            obs_features = _fetch_observation_features(
                target_date_iso,
                nws_last=features.get("nws_last"),
                atm_features=atm_features,
            )
            v2_features.update(obs_features)
            obs_populated = sum(1 for v in obs_features.values()
                                if v is not None and not (isinstance(v, float) and np.isnan(v)))
            print(f"🔭 Observation features: {obs_populated}/{len(obs_features)} populated")

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

            # Build feature DataFrame (v4 or v2 depending on available model)
            X_v2 = pd.DataFrame([v2_features])
            for col in active_feature_cols:
                if col not in X_v2.columns:
                    X_v2[col] = np.nan
            X_v2 = X_v2[active_feature_cols]
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

            if active_regressor is not None:
                v2_bias = float(active_regressor.predict(X_v2)[0])
                v2_temp = base + v2_bias
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"({active_version} regression: base={base:.0f} + bias={v2_bias:+.1f})")
            else:
                # No atm predictor and no regression — use forecast average
                all_forecasts = [features["nws_last"]]
                if has_accu and not np.isnan(features["accu_last"]):
                    all_forecasts.append(features["accu_last"])
                v2_temp = float(np.mean(all_forecasts))
                accu_note = f", AccuWx={features['accu_last']:.0f}" if has_accu else ""
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"(forecast avg: NWS={features['nws_last']:.0f}{accu_note})")

            # Exceedance adjustment DISABLED — it chases observed temps after
            # the market already sees them, producing fake WINs with no edge.
            # The v4 model with observation features will learn to make
            # legitimate early adjustments from training data over time.
            # obs_high, obs_hour = _fetch_observed_high_so_far(target_date_iso)
            # if obs_high is not None:
            #     v2_temp = _adjust_center_for_exceedance(v2_temp, obs_high, obs_hour)

            # v2 classifier bucket prediction
            # Use 15 candidates (±7) to cover full Kalshi range and handle
            # source disagreements (e.g., AccuWeather says 77, NWS says 88)
            bucket_probs = active_classifier.predict_bucket_probs(
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
                result["ml_version"] = active_version

                # Direct map: what bucket does the regression temp map to?
                from model_config import temp_to_bucket_label
                result["ml_direct_bucket"] = temp_to_bucket_label(v2_temp)

                print(f"🧠 ML {active_version} prediction for {target_date_iso}: {v2_temp:.1f}°F "
                      f"→ bucket={v2_best['bucket']} ({v2_best['probability']:.0%}) "
                      f"[direct: {result['ml_direct_bucket']}]")
                if len(bucket_probs) > 1:
                    runner = bucket_probs[1]
                    print(f"   Runner-up: {runner['bucket']} ({runner['probability']:.0%})")
        except Exception as e:
            print(f"⚠️ v2 prediction failed, using v1: {e}")

    return result


# ---------------------------------------------------------------------------
# NWS observation collection → Supabase
# ---------------------------------------------------------------------------

def _kmh_to_mph(kmh):
    """Convert km/h to mph, returning None if input is None."""
    return round(kmh * 0.621371, 1) if kmh is not None else None

def _pa_to_hpa(pa):
    """Convert Pascals to hectopascals, returning None if input is None."""
    return round(pa / 100.0, 1) if pa is not None else None

def _c_to_f(c):
    """Convert Celsius to Fahrenheit, returning None if input is None."""
    return round(c * 9.0 / 5.0 + 32.0, 1) if c is not None else None


import re
_METAR_6HR_MAX_RE = re.compile(r'\b1([01])(\d{3})\b')

def _parse_metar_6hr_max(raw_message: str) -> float | None:
    """Parse 6-hour maximum temperature from METAR remarks.

    METAR group format: 1snTTT
      - 1  = group identifier (6-hr max)
      - sn = sign (0=positive, 1=negative)
      - TTT = temperature in tenths of °C (e.g., 117 = 11.7°C)

    Example: '10117' → +11.7°C → 53.1°F
    """
    m = _METAR_6HR_MAX_RE.search(raw_message)
    if not m:
        return None
    sign = -1 if m.group(1) == '1' else 1
    temp_c = sign * int(m.group(2)) / 10.0
    return round(temp_c * 9.0 / 5.0 + 32.0, 1)


def collect_nws_observations(city_key: str = None) -> int:
    """
    Fetch recent NWS station observations and upsert into Supabase nws_observations table.

    Returns the number of rows upserted.
    """
    import nws_auto_logger as _nal
    cfg = _nal._CITY_CFG
    station = cfg.get("obs_station", "KNYC")
    tz_name = cfg.get("timezone", "America/New_York")
    city = city_key or _CITY_KEY

    # Fetch last 24 observations (~24 hours of hourly data)
    url = f"https://api.weather.gov/stations/{station}/observations?limit=24"
    req = urllib.request.Request(url, headers={
        "Accept": "application/geo+json",
        "User-Agent": "nws-forecast-logger/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ Failed to fetch NWS observations for {station}: {e}")
        return 0

    obs_features = data.get("features", [])
    if not obs_features:
        print(f"⚠️ No observations returned for {station}")
        return 0

    # Sort features oldest-first so our rate check is chronologically consistent.
    # The NWS API returns newest-first by default; sorting here makes the
    # prev → curr direction unambiguous.
    try:
        obs_features = sorted(
            obs_features,
            key=lambda f: f.get("properties", {}).get("timestamp", ""),
        )
    except Exception:
        pass  # keep original order if sort fails

    # Parse observations into rows
    rows = []
    _prev_temp_f: Optional[float] = None  # last ACCEPTED temp
    _prev_ts_str: Optional[str] = None    # last ACCEPTED timestamp
    _skipped_bad = 0
    for feat in obs_features:
        props = feat.get("properties", {})
        ts = props.get("timestamp")
        if not ts:
            continue

        temp_c = props.get("temperature", {}).get("value")
        if temp_c is None:
            continue  # skip obs with no temperature

        temp_f = _c_to_f(temp_c)

        # ── Time-aware observation sanity gate ───────────────────────────
        # The NWS station API occasionally mixes stale cached observations
        # from a previous warm day into the current overnight response.
        # This produces readings that oscillate ~9-15°F every 5 minutes —
        # physically impossible for real ambient temperature change.
        #
        # Root cause observed Apr 5 2026: readings of 64°F and 73°F
        # alternating at midnight when actual KNYC was 44°F. Each
        # individual jump was only 9°F (below a naive 15°F threshold)
        # but the RATE was 110°F/hour — clearly corrupt data.
        #
        # Gate logic:
        #  1. Hard bounds: discard readings outside [-20, 115]°F
        #  2. Time-aware rate: allow up to 15°F/hour for real changes
        #     (generous — actual cold fronts do ~5-10°F/hr). For any
        #     two readings <12 minutes apart, cap the allowable delta
        #     at 3°F (well above any real sub-hourly ambient change).
        if temp_f is not None:
            if temp_f < -20 or temp_f > 115:
                print(f"  ⚠️ obs sanity: {temp_f:.1f}°F out of physical bounds at {ts} — skipped")
                _skipped_bad += 1
                continue

            if _prev_temp_f is not None and _prev_ts_str is not None:
                try:
                    curr_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    prev_dt = datetime.fromisoformat(_prev_ts_str.replace("Z", "+00:00"))
                    hours = abs((curr_dt - prev_dt).total_seconds()) / 3600.0
                    delta_f = abs(temp_f - _prev_temp_f)
                    # max_allowable: 5°F minimum covers legitimate SPECI reports
                    # (special obs triggered by rapid change, ~5°F in 10-15 min).
                    # Scales up at 15°F/hour for longer intervals (generous —
                    # real cold fronts do ~5-10°F/hr; we allow up to 15).
                    max_allowable = max(5.0, hours * 15.0)
                    if delta_f > max_allowable:
                        rate = delta_f / max(hours, 1/60)  # °F/hr
                        print(f"  ⚠️ obs sanity: {temp_f:.1f}°F (prev {_prev_temp_f:.1f}°F, "
                              f"{delta_f:.1f}°F in {hours*60:.0f}min ≈ {rate:.0f}°F/hr) — skipped")
                        _skipped_bad += 1
                        continue
                except Exception:
                    pass  # if timestamp parse fails, accept the reading

        _prev_temp_f = temp_f
        _prev_ts_str = ts
        # ────────────────────────────────────────────────────────────────

        wind_kmh = props.get("windSpeed", {}).get("value")
        gust_kmh = props.get("windGust", {}).get("value")
        wdir = props.get("windDirection", {}).get("value")
        dewpoint_c = props.get("dewpoint", {}).get("value")
        pressure_pa = props.get("barometricPressure", {}).get("value")
        humidity = props.get("relativeHumidity", {}).get("value")
        sky = props.get("textDescription", "") or ""
        raw_msg = props.get("rawMessage", "") or ""

        # Parse 6-hr max: first try API field, then parse from METAR remarks
        max24_c = props.get("maxTemperatureLast24Hours", {}).get("value")
        six_hr_max = _c_to_f(max24_c)
        if six_hr_max is None and raw_msg:
            six_hr_max = _parse_metar_6hr_max(raw_msg)

        rows.append({
            "city": city,
            "station": station,
            "observed_at": ts,
            "temp_f": _c_to_f(temp_c),
            "wind_speed_mph": _kmh_to_mph(wind_kmh),
            "wind_gust_mph": _kmh_to_mph(gust_kmh),
            "wind_direction_deg": round(wdir, 0) if wdir is not None else None,
            "sky_condition": sky.strip() if sky else None,
            "dewpoint_f": _c_to_f(dewpoint_c),
            "pressure_hpa": _pa_to_hpa(pressure_pa),
            "humidity_pct": round(humidity, 1) if humidity is not None else None,
            "six_hr_max_f": six_hr_max,
            "raw_message": raw_msg.strip() if raw_msg else None,
        })

    if not rows:
        print(f"⚠️ No valid temperature observations for {station}")
        return 0

    # Upsert into Supabase nws_observations table
    sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not sb_url or not sb_key:
        print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE — skipping obs upsert")
        return 0

    endpoint = f"{sb_url}/rest/v1/nws_observations"
    upserted = 0
    for row in rows:
        body = json.dumps(row, ensure_ascii=False).encode("utf-8")
        r = urllib.request.Request(
            f"{endpoint}?on_conflict=city,station,observed_at",
            data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
                "apikey": sb_key,
                "Authorization": f"Bearer {sb_key}",
            },
        )
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                _ = resp.read()
            upserted += 1
        except Exception as e:
            err_body = ""
            if hasattr(e, "read"):
                try: err_body = e.read().decode("utf-8", "ignore")
                except: pass
            print(f"⚠️ obs upsert failed for {row['observed_at']}: {e} {err_body}")

    bad_note = f", {_skipped_bad} rejected by sanity gate" if _skipped_bad > 0 else ""
    print(f"✅ Collected {upserted}/{len(rows)} NWS observations for {station} ({city}){bad_note}")
    return upserted


# ---------------------------------------------------------------------------
# Observation-derived features for ML model
# ---------------------------------------------------------------------------

# Sky condition → numeric cloud cover mapping
_SKY_COVER_MAP = {
    "clear": 0.0, "fair": 0.1, "a few clouds": 0.15,
    "partly cloudy": 0.25, "mostly cloudy": 0.75,
    "overcast": 1.0, "fog": 0.9, "fog/mist": 0.9,
    "haze": 0.3, "mostly clear": 0.1,
}


def _sky_to_cloud_cover(text_desc: str) -> float:
    """Map NWS textDescription to numeric cloud cover [0.0, 1.0].

    Handles compound descriptions like 'Mostly Cloudy and Breezy' by
    checking for known sky keywords in order of specificity.
    """
    if not text_desc:
        return np.nan
    low = text_desc.lower().strip()
    # Direct match first
    if low in _SKY_COVER_MAP:
        return _SKY_COVER_MAP[low]
    # Check for keywords in compound descriptions (most specific first)
    for key in ("overcast", "mostly cloudy", "partly cloudy", "a few clouds",
                "fog", "haze", "mostly clear", "fair", "clear"):
        if key in low:
            return _SKY_COVER_MAP[key]
    # Rain/snow/thunderstorm → assume heavy cloud cover
    if any(w in low for w in ("rain", "snow", "thunder", "drizzle", "sleet", "ice")):
        return 0.95
    return 0.5  # unknown → middle


def _query_supabase_observations(target_date_iso: str, city: str = None) -> list[dict]:
    """Query Supabase nws_observations for a specific date. Returns list of obs dicts."""
    sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not sb_url or not sb_key:
        return []

    import nws_auto_logger as _nal
    cfg = _nal._CITY_CFG
    tz_name = cfg.get("timezone", "America/New_York")
    city = city or _CITY_KEY

    # Query observations for the target date (local time boundaries)
    # Use date range: target_date 00:00 to target_date+1 00:00 in local tz
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    start_local = datetime.fromisoformat(f"{target_date_iso}T00:00:00").replace(tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")

    endpoint = f"{sb_url}/rest/v1/nws_observations"
    params = (
        f"?city=eq.{city}"
        f"&observed_at=gte.{start_utc}"
        f"&observed_at=lt.{end_utc}"
        f"&order=observed_at.asc"
        f"&limit=50"
    )
    req = urllib.request.Request(
        endpoint + params,
        headers={
            "apikey": sb_key,
            "Authorization": f"Bearer {sb_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        return rows
    except Exception as e:
        print(f"⚠️ Failed to query observations for {target_date_iso}: {e}")
        return []


def _fetch_observation_features(
    target_date_iso: str,
    nws_last: float = None,
    atm_features: dict = None,
) -> dict:
    """
    Compute observation-derived features from NWS observations stored in Supabase.

    Args:
        target_date_iso: The date we're predicting for (YYYY-MM-DD)
        nws_last: Latest NWS forecast value (°F) for obs_temp_vs_forecast_max
        atm_features: Dict of atmospheric features (for obs_vs_intra_forecast)

    Returns dict with all OBSERVATION_COLS. NaN for any unavailable feature.
    """
    nan_result = {col: np.nan for col in OBSERVATION_COLS}

    # Only fetch observations for today (no observations exist for future dates)
    today = today_nyc().isoformat()
    if target_date_iso != today:
        # For D1 (tomorrow) predictions, we can still provide today's overshoot
        if nws_last is not None:
            today_obs = _query_supabase_observations(today)
            if today_obs:
                today_max = max(
                    (r["temp_f"] for r in today_obs if r.get("temp_f") is not None),
                    default=None,
                )
                if today_max is not None:
                    nan_result["obs_temp_vs_forecast_max"] = round(today_max - nws_last, 1)
        return nan_result

    obs_rows = _query_supabase_observations(target_date_iso)
    if not obs_rows:
        return nan_result

    # Filter to rows with valid temperature
    valid_obs = [r for r in obs_rows if r.get("temp_f") is not None]
    if not valid_obs:
        return nan_result

    # Second-line sanity gate: remove spike outliers already in Supabase.
    # Rows come in ascending observed_at order. Uses the same time-aware
    # rate check as collect_nws_observations() to handle bad data already
    # stored before this fix was deployed.
    if len(valid_obs) > 1:
        filtered = [valid_obs[0]]
        for obs in valid_obs[1:]:
            prev_f = filtered[-1]["temp_f"]
            curr_f = obs["temp_f"]
            delta_f = abs(curr_f - prev_f)
            try:
                curr_dt = datetime.fromisoformat(
                    obs["observed_at"].replace("Z", "+00:00"))
                prev_dt = datetime.fromisoformat(
                    filtered[-1]["observed_at"].replace("Z", "+00:00"))
                hours = abs((curr_dt - prev_dt).total_seconds()) / 3600.0
                max_allowable = max(5.0, hours * 15.0)
            except Exception:
                max_allowable = 15.0  # fallback if timestamp parse fails
            if delta_f > max_allowable:
                rate = delta_f / max(hours if 'hours' in dir() else 1, 1/60)
                print(f"  ⚠️ obs filter: {curr_f:.1f}°F from {prev_f:.1f}°F "
                      f"({delta_f:.1f}°F, rate too high) — excluded from ML features")
            else:
                filtered.append(obs)
        if len(filtered) < len(valid_obs):
            print(f"  ⚠️ Removed {len(valid_obs)-len(filtered)} spiked obs from ML features "
                  f"({len(filtered)} of {len(valid_obs)} kept)")
        valid_obs = filtered if filtered else valid_obs  # always keep at least one

    features = {}

    # Latest observation
    latest = valid_obs[-1]  # already sorted by observed_at asc
    features["obs_latest_temp"] = latest["temp_f"]

    # Parse hour from observed_at timestamp
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        tz_name = cfg.get("timezone", "America/New_York")
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
        obs_dt = datetime.fromisoformat(latest["observed_at"].replace("Z", "+00:00"))
        obs_local = obs_dt.astimezone(tz)
        features["obs_latest_hour"] = obs_local.hour
    except Exception:
        features["obs_latest_hour"] = np.nan

    # Running daily max
    features["obs_max_so_far"] = max(r["temp_f"] for r in valid_obs)

    # 6-hour max: max of last 6 hours of observations, or from six_hr_max_f column
    six_hr_vals = [r["six_hr_max_f"] for r in valid_obs if r.get("six_hr_max_f") is not None]
    if six_hr_vals:
        features["obs_6hr_max"] = max(six_hr_vals)
    else:
        # Compute from hourly obs: max of last 6 observations
        recent_6 = valid_obs[-6:] if len(valid_obs) >= 6 else valid_obs
        features["obs_6hr_max"] = max(r["temp_f"] for r in recent_6)

    # Delta vs Open-Meteo forecast at same hour
    obs_hour = features.get("obs_latest_hour")
    if obs_hour is not None and atm_features and not np.isnan(obs_hour):
        # Map hour to nearest intraday forecast feature
        intra_map = {
            9: "intra_temp_9am", 10: "intra_temp_9am",
            11: "intra_temp_noon", 12: "intra_temp_noon",
            13: "intra_temp_3pm", 14: "intra_temp_3pm", 15: "intra_temp_3pm",
            16: "intra_temp_5pm", 17: "intra_temp_5pm",
        }
        intra_key = intra_map.get(int(obs_hour))
        if intra_key and intra_key in atm_features:
            intra_val = atm_features[intra_key]
            if intra_val is not None and not (isinstance(intra_val, float) and np.isnan(intra_val)):
                features["obs_vs_intra_forecast"] = round(
                    features["obs_latest_temp"] - intra_val, 1
                )
            else:
                features["obs_vs_intra_forecast"] = np.nan
        else:
            features["obs_vs_intra_forecast"] = np.nan
    else:
        features["obs_vs_intra_forecast"] = np.nan

    # Wind speed and gust
    features["obs_wind_speed"] = latest.get("wind_speed_mph")
    if features["obs_wind_speed"] is None:
        features["obs_wind_speed"] = np.nan
    features["obs_wind_gust"] = latest.get("wind_gust_mph")
    if features["obs_wind_gust"] is None:
        features["obs_wind_gust"] = np.nan

    # Wind direction (circular encoding)
    wdir = latest.get("wind_direction_deg")
    if wdir is not None:
        features["obs_wind_dir_sin"] = round(math.sin(math.radians(wdir)), 4)
        features["obs_wind_dir_cos"] = round(math.cos(math.radians(wdir)), 4)
    else:
        features["obs_wind_dir_sin"] = np.nan
        features["obs_wind_dir_cos"] = np.nan

    # Cloud cover from sky condition text
    features["obs_cloud_cover"] = _sky_to_cloud_cover(latest.get("sky_condition", ""))

    # Heating rate: linear slope over last 3 hours of observations
    if len(valid_obs) >= 2:
        # Use last min(len, ~3 hours worth) observations
        recent = valid_obs[-4:] if len(valid_obs) >= 4 else valid_obs
        try:
            temps = [r["temp_f"] for r in recent]
            # Compute hours elapsed for each observation relative to first
            t0_str = recent[0]["observed_at"].replace("Z", "+00:00")
            t0 = datetime.fromisoformat(t0_str)
            hours = []
            for r in recent:
                ti = datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00"))
                hours.append((ti - t0).total_seconds() / 3600.0)
            if hours[-1] > 0:
                # Simple linear slope: (last - first) / hours_elapsed
                features["obs_heating_rate"] = round(
                    (temps[-1] - temps[0]) / hours[-1], 2
                )
            else:
                features["obs_heating_rate"] = np.nan
        except Exception:
            features["obs_heating_rate"] = np.nan
    else:
        features["obs_heating_rate"] = np.nan

    # Obs max vs NWS forecast
    if nws_last is not None and features["obs_max_so_far"] is not None:
        features["obs_temp_vs_forecast_max"] = round(
            features["obs_max_so_far"] - nws_last, 1
        )
    else:
        features["obs_temp_vs_forecast_max"] = np.nan

    # Fill any missing keys with NaN
    for col in OBSERVATION_COLS:
        if col not in features:
            features[col] = np.nan

    return features


def _sb_endpoint():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE")
    return f"{url}/rest/v1/prediction_logs", key

def _log_ml_revision(
    target_date_iso: str,
    lead_used: str,
    ml: dict,
    trigger_reason: str,
) -> None:
    """Append one row to prediction_revision_log — pure insert, never overwrites.
    Called on every ML compute: first write (trigger='first_write') and
    every intraday recompute (trigger = comma-joined reason strings).
    """
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not url or not key:
        return  # silently skip in non-Supabase environments
    endpoint = f"{url}/rest/v1/prediction_revision_log"
    row = {
        "city": _CITY_KEY,
        "target_date": target_date_iso,
        "lead_used": lead_used,
        "ml_bucket": ml.get("ml_bucket"),
        "ml_f": ml.get("ml_f"),
        "ml_confidence": ml.get("ml_confidence"),
        "trigger_reason": trigger_reason,
    }
    data = json.dumps(row, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        print(f"📝 Revision logged: {ml.get('ml_bucket')} / {ml.get('ml_f')}°F [{trigger_reason}]")
    except Exception as e:
        print(f"⚠️ revision log write failed: {e}")


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

# ---------------------------------------------------------------------------
# Observation feature backfill for training
# ---------------------------------------------------------------------------

def backfill_observation_features(city_key: str = None) -> str:
    """
    Query all real NWS observations from Supabase and compute observation
    features for each date. Saves to {prefix}observation_data.csv.

    This gives the training pipeline REAL obs_vs_intra_forecast deltas
    (non-zero) instead of proxy values (always 0). The model needs real
    deltas to learn what forecast-vs-observation divergence means.

    Returns path to the CSV file written.
    """
    import nws_auto_logger as _nal
    from model_config import OBSERVATION_COLS

    cfg = _nal._CITY_CFG
    prefix = cfg.get("model_prefix", "")
    city = city_key or _CITY_KEY
    tz_name = cfg.get("timezone", "America/New_York")

    sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not sb_url or not sb_key:
        print("⚠️ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE — cannot backfill observations")
        return ""

    # Fetch ALL observations for this city from Supabase (paginated)
    endpoint = f"{sb_url}/rest/v1/nws_observations"
    all_obs = []
    offset = 0
    page_size = 1000
    while True:
        params = (
            f"?city=eq.{city}"
            f"&order=observed_at.asc"
            f"&limit={page_size}"
            f"&offset={offset}"
        )
        req = urllib.request.Request(
            endpoint + params,
            headers={
                "apikey": sb_key,
                "Authorization": f"Bearer {sb_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            all_obs.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        except Exception as e:
            print(f"⚠️ Failed to fetch observations page at offset {offset}: {e}")
            break

    if not all_obs:
        print(f"⚠️ No observations found for {city} — nothing to backfill")
        return ""

    print(f"📊 Fetched {len(all_obs)} total observations for {city}")

    # Group observations by local date
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    obs_by_date = {}
    for obs in all_obs:
        ts = obs.get("observed_at")
        if not ts or obs.get("temp_f") is None:
            continue
        obs_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        obs_local = obs_dt.astimezone(tz)
        date_str = obs_local.strftime("%Y-%m-%d")
        if date_str not in obs_by_date:
            obs_by_date[date_str] = []
        obs_by_date[date_str].append(obs)

    print(f"📅 Observations span {len(obs_by_date)} unique dates")

    # Load atmospheric data to get intraday forecasts for obs_vs_intra_forecast
    atm_csv = f"{prefix}atmospheric_data.csv"
    atm_df = None
    if os.path.exists(atm_csv):
        atm_df = pd.read_csv(atm_csv)
        atm_df["target_date"] = atm_df["target_date"].astype(str)
        print(f"📂 Loaded {len(atm_df)} atmospheric feature rows for intraday forecast comparison")

    # Also load NWS forecast data for obs_temp_vs_forecast_max
    nws_csv = f"{prefix}nws_forecast_log.csv" if prefix else "nws_forecast_log.csv"
    nws_last_by_date = {}
    if os.path.exists(nws_csv):
        nws_df = pd.read_csv(nws_csv)
        for _, row in nws_df.iterrows():
            if row.get("forecast_or_actual") == "forecast" and row.get("target_date"):
                try:
                    nws_last_by_date[str(row["target_date"])] = float(row["predicted_high"])
                except (ValueError, TypeError):
                    pass

    # Compute observation features for each date
    feature_rows = []
    for date_str in sorted(obs_by_date.keys()):
        obs_list = obs_by_date[date_str]
        # Sort by timestamp
        obs_list.sort(key=lambda r: r.get("observed_at", ""))

        # Filter to valid temp observations
        valid_obs = [r for r in obs_list if r.get("temp_f") is not None]
        if not valid_obs:
            continue

        # Simulate as-of noon: only use observations up to ~noon local
        noon_obs = []
        for obs in valid_obs:
            obs_dt = datetime.fromisoformat(obs["observed_at"].replace("Z", "+00:00"))
            obs_local = obs_dt.astimezone(tz)
            if obs_local.hour <= 12:
                noon_obs.append(obs)

        # If no pre-noon obs, use first few obs of the day
        if not noon_obs:
            noon_obs = valid_obs[:6]

        if not noon_obs:
            continue

        features = {"target_date": date_str, "city": city}
        latest = noon_obs[-1]

        # obs_latest_temp
        features["obs_latest_temp"] = latest["temp_f"]

        # obs_latest_hour
        try:
            obs_dt = datetime.fromisoformat(latest["observed_at"].replace("Z", "+00:00"))
            obs_local = obs_dt.astimezone(tz)
            features["obs_latest_hour"] = float(obs_local.hour)
        except Exception:
            features["obs_latest_hour"] = np.nan

        # obs_max_so_far
        features["obs_max_so_far"] = max(r["temp_f"] for r in noon_obs)

        # obs_6hr_max
        six_hr_vals = [r["six_hr_max_f"] for r in noon_obs if r.get("six_hr_max_f") is not None]
        if six_hr_vals:
            features["obs_6hr_max"] = max(six_hr_vals)
        else:
            recent_6 = noon_obs[-6:] if len(noon_obs) >= 6 else noon_obs
            features["obs_6hr_max"] = max(r["temp_f"] for r in recent_6)

        # obs_vs_intra_forecast — THE KEY FEATURE
        # Compare real NWS observation against Open-Meteo forecast at the same hour
        features["obs_vs_intra_forecast"] = np.nan
        obs_hour = features.get("obs_latest_hour")
        if atm_df is not None and obs_hour is not None and not np.isnan(obs_hour):
            atm_row = atm_df[atm_df["target_date"] == date_str]
            if not atm_row.empty:
                intra_map = {
                    9: "intra_temp_9am", 10: "intra_temp_9am",
                    11: "intra_temp_noon", 12: "intra_temp_noon",
                    13: "intra_temp_3pm", 14: "intra_temp_3pm", 15: "intra_temp_3pm",
                    16: "intra_temp_5pm", 17: "intra_temp_5pm",
                }
                intra_key = intra_map.get(int(obs_hour))
                if intra_key and intra_key in atm_row.columns:
                    intra_val = atm_row.iloc[0][intra_key]
                    if pd.notna(intra_val):
                        features["obs_vs_intra_forecast"] = round(
                            features["obs_latest_temp"] - float(intra_val), 1
                        )

        # obs_wind_speed, obs_wind_gust
        features["obs_wind_speed"] = latest.get("wind_speed_mph") if latest.get("wind_speed_mph") is not None else np.nan
        features["obs_wind_gust"] = latest.get("wind_gust_mph") if latest.get("wind_gust_mph") is not None else np.nan

        # obs_wind_dir_sin, obs_wind_dir_cos
        wdir = latest.get("wind_direction_deg")
        if wdir is not None:
            features["obs_wind_dir_sin"] = round(math.sin(math.radians(wdir)), 4)
            features["obs_wind_dir_cos"] = round(math.cos(math.radians(wdir)), 4)
        else:
            features["obs_wind_dir_sin"] = np.nan
            features["obs_wind_dir_cos"] = np.nan

        # obs_cloud_cover
        features["obs_cloud_cover"] = _sky_to_cloud_cover(latest.get("sky_condition", ""))

        # obs_heating_rate
        if len(noon_obs) >= 2:
            try:
                t0_str = noon_obs[0]["observed_at"].replace("Z", "+00:00")
                t0 = datetime.fromisoformat(t0_str)
                t1_str = noon_obs[-1]["observed_at"].replace("Z", "+00:00")
                t1 = datetime.fromisoformat(t1_str)
                hours_span = (t1 - t0).total_seconds() / 3600.0
                if hours_span > 0:
                    features["obs_heating_rate"] = round(
                        (noon_obs[-1]["temp_f"] - noon_obs[0]["temp_f"]) / hours_span, 2
                    )
                else:
                    features["obs_heating_rate"] = np.nan
            except Exception:
                features["obs_heating_rate"] = np.nan
        else:
            features["obs_heating_rate"] = np.nan

        # obs_temp_vs_forecast_max
        nws_last = nws_last_by_date.get(date_str)
        if nws_last is not None:
            features["obs_temp_vs_forecast_max"] = round(features["obs_max_so_far"] - nws_last, 1)
        else:
            features["obs_temp_vs_forecast_max"] = np.nan

        feature_rows.append(features)

    if not feature_rows:
        print(f"⚠️ No observation features computed for {city}")
        return ""

    # Save to CSV
    csv_path = f"{prefix}observation_data.csv"
    df = pd.DataFrame(feature_rows)
    df.to_csv(csv_path, index=False)

    # Report non-zero obs_vs_intra_forecast stats
    non_zero = df["obs_vs_intra_forecast"].dropna()
    non_zero = non_zero[non_zero != 0]
    print(f"✅ Saved {len(df)} observation feature rows to {csv_path}")
    print(f"   obs_vs_intra_forecast: {len(non_zero)} non-zero values "
          f"(mean={non_zero.mean():.1f}°F, std={non_zero.std():.1f}°F)" if len(non_zero) > 0
          else f"   obs_vs_intra_forecast: all zero or NaN (need atmospheric_data.csv for comparison)")

    return csv_path


_LOCK_NOT_FOUND = "not_found"
_LOCK_ERROR = "error"

def _fetch_existing_prediction(target_date_iso: str):
    """
    Check if a prediction already exists in Supabase for this target date.

    Uses a lead-agnostic idempotency key so the first prediction for a given
    target date wins (tomorrow's prediction locks out today's for the same date).

    Also fetches nws_d0 and accuweather so write_today_for_today can compare
    the current agency forecasts against the values used in the last ML run,
    and only recompute when an agency has actually revised their forecast.

    Returns:
      dict  – existing prediction row (contains ml_f, ml_bucket, ml_confidence,
               ml_bucket_probs, ml_version, kalshi_market_snapshot, nws_d0, accuweather)
      _LOCK_NOT_FOUND – no row exists yet (compute fresh ML prediction)
      _LOCK_ERROR – network/auth error (skip ML to avoid overwriting)
    """
    try:
        endpoint, key = _sb_endpoint()
        idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{target_date_iso}"
        url = (f"{endpoint}?idempotency_key=eq.{idem_key}"
               f"&select=ml_f,ml_bucket,ml_confidence,ml_bucket_probs,ml_version,"
               f"kalshi_market_snapshot,nws_d0,accuweather,atm_snapshot")
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


def _check_atmospheric_shift(
    live_atm: dict,
    stored_snapshot_raw,
) -> tuple:
    """
    Compare live atmospheric features against the morning baseline snapshot.
    Returns (triggered: bool, reasons: list[str]).

    Thresholds are physically motivated starting points, not empirically calibrated:
      - BL height change > 500m  : spike risk changed materially
      - Ensemble spread change > 2°F : uncertainty widened/narrowed significantly
      - HRRR-ECMWF diff change > 3°F : model disagreement shifted (one is badly wrong)
      - Ensemble mean shift > 2°F    : consensus moved (independent of agencies)

    Only fires BEFORE the D0 cutoff (enforced by caller). Never fires when
    stored_snapshot_raw is None (no baseline = canonical hasn't been written yet,
    meaning this is the first run — ML fires anyway via LOCK_NOT_FOUND path).
    """
    if not live_atm or not stored_snapshot_raw:
        return False, []
    try:
        stored = (
            json.loads(stored_snapshot_raw)
            if isinstance(stored_snapshot_raw, str)
            else stored_snapshot_raw
        )
    except Exception:
        return False, []

    reasons = []

    def _valid(d: dict, k: str):
        v = d.get(k)
        return v if (v is not None and isinstance(v, float) and not math.isnan(v)) else None

    # BL height: >500m shift — spike risk changed
    bl_live    = _valid(live_atm, "atm_bl_height_max")
    bl_stored  = _valid(stored,   "atm_bl_height_max")
    if bl_live is not None and bl_stored is not None:
        delta = bl_live - bl_stored
        if abs(delta) > 500:
            reasons.append(
                f"BL height {bl_stored:.0f}→{bl_live:.0f}m (Δ{delta:+.0f}m, "
                f"{'spike risk ↑' if delta > 0 else 'spike risk ↓'})"
            )

    # Ensemble spread: >2°F change — uncertainty spiked or collapsed
    ens_spread_live   = _valid(live_atm, "ens_spread")
    ens_spread_stored = _valid(stored,   "ens_spread")
    if ens_spread_live is not None and ens_spread_stored is not None:
        delta = ens_spread_live - ens_spread_stored
        if abs(delta) > 2.0:
            reasons.append(
                f"Ensemble spread {ens_spread_stored:.1f}→{ens_spread_live:.1f}°F (Δ{delta:+.1f}°F)"
            )

    # HRRR-ECMWF diff: >3°F change — one model diverging significantly
    hrrr_live   = _valid(live_atm, "mm_hrrr_ecmwf_diff")
    hrrr_stored = _valid(stored,   "mm_hrrr_ecmwf_diff")
    if hrrr_live is not None and hrrr_stored is not None:
        delta = hrrr_live - hrrr_stored
        if abs(delta) > 3.0:
            reasons.append(
                f"HRRR-ECMWF diff {hrrr_stored:+.1f}→{hrrr_live:+.1f}°F (Δ{delta:+.1f}°F)"
            )

    # Ensemble mean: >2°F shift — consensus moved independently of agency forecasts
    ens_mean_live   = _valid(live_atm, "ens_mean")
    ens_mean_stored = _valid(stored,   "ens_mean")
    if ens_mean_live is not None and ens_mean_stored is not None:
        delta = ens_mean_live - ens_mean_stored
        if abs(delta) > 2.0:
            reasons.append(
                f"Ensemble mean {ens_mean_stored:.1f}→{ens_mean_live:.1f}°F (Δ{delta:+.1f}°F)"
            )

    triggered = len(reasons) > 0
    return triggered, reasons


def _add_obs_to_snap(snap: dict, live_obs: dict) -> None:
    """
    Add observation snapshot keys to an existing atm_snapshot dict in-place.
    These keys are prefixed obs_snap_* to avoid collision with model feature names.
    Called on both canonical write and post-recompute baseline advance.
    """
    if not live_obs:
        return

    def _safe(v):
        """Return v if valid float, else None (JSON serializable)."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 3)
        except (TypeError, ValueError):
            return None

    snap["obs_snap_temp"]        = _safe(live_obs.get("obs_latest_temp"))
    snap["obs_snap_heating_rate"]= _safe(live_obs.get("obs_heating_rate"))
    snap["obs_snap_vs_forecast"] = _safe(live_obs.get("obs_vs_intra_forecast"))
    snap["obs_snap_hour"]        = _safe(live_obs.get("obs_latest_hour"))
    obs_count = sum(
        1 for v in live_obs.values()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    )
    snap["obs_snap_populated"] = obs_count


def _check_obs_trigger(
    live_obs: dict,
    stored_snapshot: dict,
) -> tuple:
    """
    Check if ground-truth NWS station observations warrant an ML recompute.
    Returns (triggered: bool, reasons: list[str]).

    Three triggers — all require being in the morning heating window (before 2pm):

    1. obs_daytime_first: canonical was set before 9am (pre-sunrise obs only,
       no meaningful intraday signal). Now daytime obs are available for the
       first time. Fires once so the model re-runs with real ground-truth temps.

    2. obs_cold_vs_forecast: observed temp at this hour is 3°F+ colder than
       what Open-Meteo predicted for the same hour (obs_vs_intra_forecast).
       Only valid in the 9am-1pm window where intra_map has coverage.
       Indicates the intraday heating curve is running behind — reality is
       colder than the model assumed when it computed the canonical.

    3. obs_slow_heating: observed heating rate < 0.5°F/hr after 9am. With a
       freeze-warning start and barely-rising temps, the model needs to know
       the actual trajectory, not just the forecasted curve.
    """
    if not live_obs:
        return False, []

    reasons = []

    def _fval(d: dict, k: str):
        """Return float value or None if missing/NaN."""
        v = d.get(k)
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    live_hour   = _fval(live_obs,       "obs_latest_hour")
    live_temp   = _fval(live_obs,       "obs_latest_temp")
    live_rate   = _fval(live_obs,       "obs_heating_rate")
    live_vs_fc  = _fval(live_obs,       "obs_vs_intra_forecast")
    stored_hour = _fval(stored_snapshot,"obs_snap_hour")

    # Trigger 1: First daytime obs available after a pre-9am canonical write.
    # Canonical set at e.g. 7:32am has obs_latest_hour=6 with no intraday signal.
    # When obs_latest_hour crosses 9, we finally have a meaningful morning reading.
    if (
        live_hour is not None and live_hour >= 9
        and stored_hour is not None and stored_hour < 9
        and live_temp is not None
    ):
        reasons.append(
            f"obs_daytime_first: {live_temp:.1f}°F observed at {int(live_hour)}:00 "
            f"(canonical set at {int(stored_hour)}:00 — no daytime obs then)"
        )

    # Trigger 2: Observed temp running 3°F+ colder than intraday forecast.
    # obs_vs_intra_forecast = obs_latest_temp - Open-Meteo forecasted temp at same hour.
    # Negative = reality colder than model assumed. Only fire if this is NEW
    # (wasn't already cold at canonical write — avoid re-firing on same cold reading).
    stored_vs_fc = _fval(stored_snapshot, "obs_snap_vs_forecast")
    if (
        live_vs_fc is not None and live_vs_fc < -3.0
        and live_hour is not None and 9 <= live_hour <= 13
    ):
        # Only trigger if we're seeing a new cold signal vs what was stored
        if stored_vs_fc is None or live_vs_fc < stored_vs_fc - 1.0:
            reasons.append(
                f"obs_cold_vs_forecast: {live_vs_fc:+.1f}°F vs intraday forecast "
                f"at {int(live_hour)}:00 (reality running cold)"
            )

    # Trigger 3: Observed heating rate < 0.5°F/hr after 9am — slow/no heating.
    # Only trigger if the stored snapshot showed a different regime (faster heating
    # or no obs yet). Avoids re-firing every run on the same slow-heating reading.
    stored_rate = _fval(stored_snapshot, "obs_snap_heating_rate")
    if (
        live_rate is not None and live_hour is not None
        and live_hour >= 9 and live_rate < 0.5
    ):
        # Fire if: no stored rate, or stored rate was in normal range (>=0.5)
        if stored_rate is None or stored_rate >= 0.5:
            reasons.append(
                f"obs_slow_heating: {live_rate:+.2f}°F/hr at {int(live_hour)}:00 "
                f"(below 0.5°F/hr threshold — cold start confirmed)"
            )

    triggered = len(reasons) > 0
    return triggered, reasons


# Keys stored in atm_snapshot — enough to detect intraday shifts, small payload
_ATM_SNAPSHOT_KEYS = (
    "atm_bl_height_max", "atm_bl_height_mean",
    "ens_spread", "ens_std", "ens_mean", "ens_skew",
    "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff", "mm_spread",
    "atm_850mb_temp_max", "atm_925mb_temp_max",
    "atm_solar_radiation_peak",
)


def _score_bucket(ml_bucket: str, actual_int: int, kalshi_snapshot_raw) -> bool:
    """Return True (WIN) if actual_int falls in ml_bucket."""
    if kalshi_snapshot_raw:
        try:
            mkt = json.loads(kalshi_snapshot_raw) if isinstance(kalshi_snapshot_raw, str) else kalshi_snapshot_raw
            actual_kalshi = _find_kalshi_bucket_for_temp(float(actual_int), mkt)
            if "-" in ml_bucket:
                ml_f_ref = float(ml_bucket.split("-")[0])
            elif ml_bucket.startswith(">="):
                ml_f_ref = float(ml_bucket[2:])
            elif ml_bucket.startswith("<="):
                ml_f_ref = float(ml_bucket[2:])
            else:
                ml_f_ref = None
            ml_kalshi = _find_kalshi_bucket_for_temp(ml_f_ref, mkt) if ml_f_ref is not None else None
            ml_pick = ml_bucket if ml_bucket in mkt else ml_kalshi
            if actual_kalshi and ml_pick == actual_kalshi:
                return True
            return False
        except Exception:
            pass
    # Fallback: direct bucket check
    if ml_bucket.startswith("<="):
        try: return actual_int <= int(ml_bucket[2:])
        except ValueError: return False
    elif ml_bucket.startswith(">="):
        try: return actual_int >= int(ml_bucket[2:])
        except ValueError: return False
    elif "-" in ml_bucket:
        parts = ml_bucket.split("-")
        if len(parts) == 2:
            try:
                lo, hi = int(parts[0]), int(parts[1])
                return lo <= actual_int <= hi
            except ValueError: pass
    return False


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
               f"&select=ml_bucket,ml_f,ml_result,ml_actual_high,kalshi_market_snapshot,"
               f"ml_bucket_canonical,ml_f_canonical,ml_result_canonical")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            pred_rows = json.loads(resp.read().decode("utf-8"))

        if not pred_rows or not pred_rows[0].get("ml_bucket"):
            return
        pred = pred_rows[0]

        # Already scored with same actual? Skip.
        # But re-score if actual changed (e.g., CLI updated overnight)
        prev_actual = _float_or_none(pred.get("ml_actual_high"))
        if pred.get("ml_result") and prev_actual is not None and abs(prev_actual - actual_high) < 0.1:
            return  # same actual, already scored

        # Score against Kalshi's actual bucket structure (not ML's internal buckets)
        ml_bucket = pred["ml_bucket"]
        actual_int = int(round(actual_high))
        is_win = False

        # Try to use Kalshi market snapshot for accurate bucket comparison
        kalshi_snapshot = pred.get("kalshi_market_snapshot")
        if kalshi_snapshot:
            try:
                mkt = json.loads(kalshi_snapshot) if isinstance(kalshi_snapshot, str) else kalshi_snapshot
                # Find which Kalshi bucket the actual falls in
                actual_kalshi = _find_kalshi_bucket_for_temp(float(actual_int), mkt)
                # Find which Kalshi bucket the ML's center temp falls in
                ml_f_val = _float_or_none(pred.get("ml_f"))
                ml_kalshi = _find_kalshi_bucket_for_temp(ml_f_val, mkt) if ml_f_val is not None else None
                # Also check if the ML's picked bucket label matches a Kalshi bucket
                ml_pick_kalshi = ml_bucket if ml_bucket in mkt else _find_kalshi_bucket_for_temp(
                    float(ml_bucket.split("-")[0]) if "-" in ml_bucket else float(ml_bucket.replace("<=","").replace(">=","")),
                    mkt
                ) if ml_bucket else None

                if actual_kalshi and (ml_kalshi == actual_kalshi or ml_pick_kalshi == actual_kalshi):
                    is_win = True
                    print(f"📊 Kalshi-aware scoring: actual {actual_int}°F → {actual_kalshi}, "
                          f"ML pick → {ml_kalshi or ml_pick_kalshi} → WIN")
                elif actual_kalshi:
                    print(f"📊 Kalshi-aware scoring: actual {actual_int}°F → {actual_kalshi}, "
                          f"ML pick → {ml_kalshi or ml_pick_kalshi} → MISS")
                else:
                    # Couldn't map actual to a Kalshi bucket — fall back to direct comparison
                    kalshi_snapshot = None  # triggers fallback below
            except Exception as e:
                print(f"⚠️ Kalshi-aware scoring failed, using fallback: {e}")
                kalshi_snapshot = None  # triggers fallback below

        # Fallback: direct bucket check (when no Kalshi snapshot available)
        if not kalshi_snapshot:
            if ml_bucket.startswith("<="):
                try:
                    threshold = int(ml_bucket[2:])
                    is_win = actual_int <= threshold
                except ValueError:
                    pass
            elif ml_bucket.startswith(">="):
                try:
                    threshold = int(ml_bucket[2:])
                    is_win = actual_int >= threshold
                except ValueError:
                    pass
            elif "-" in ml_bucket:
                parts = ml_bucket.split("-")
                if len(parts) == 2:
                    try:
                        lo, hi = int(parts[0]), int(parts[1])
                        is_win = lo <= actual_int <= hi
                    except ValueError:
                        pass

        result = "WIN" if is_win else "MISS"

        # Score canonical (first-of-day) prediction separately for comparison research.
        patch_data: dict = {"ml_result": result, "ml_actual_high": actual_high}
        canonical_bucket = pred.get("ml_bucket_canonical")
        ks_raw = pred.get("kalshi_market_snapshot")
        if canonical_bucket and canonical_bucket != ml_bucket:
            # Canonical differs from latest — score both and log
            canon_win = _score_bucket(canonical_bucket, actual_int, ks_raw)
            canon_result = "WIN" if canon_win else "MISS"
            patch_data["ml_result_canonical"] = canon_result
            flip_icon = "🔄" if canon_win != is_win else ("✅" if canon_win else "❌")
            print(f"{flip_icon} Canonical vs Latest: '{canonical_bucket}' → {canon_result} | "
                  f"'{ml_bucket}' → {result} (actual={actual_high}°F)")
        elif canonical_bucket:
            # Canonical = latest — same result
            patch_data["ml_result_canonical"] = result

        # Update prediction_logs with result
        patch = json.dumps(patch_data).encode("utf-8")
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


def compare_canonical_vs_latest_accuracy() -> None:
    """
    Query all scored days from Supabase and compare first-of-day (canonical)
    vs latest prediction accuracy. Logs a summary table to stdout.
    Run periodically (e.g., weekly) to decide which prediction to use for scoring.
    """
    try:
        endpoint, key = _sb_endpoint()
        url = (f"{endpoint}?city=eq.{_CITY_KEY}"
               f"&ml_result=not.is.null"
               f"&ml_bucket_canonical=not.is.null"
               f"&ml_result_canonical=not.is.null"
               f"&select=target_date,ml_bucket,ml_bucket_canonical,"
               f"ml_result,ml_result_canonical,ml_actual_high"
               f"&order=target_date.desc&limit=90")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            rows = json.loads(resp.read().decode("utf-8"))

        if not rows:
            print("📊 No scored days with diverging canonical/latest predictions yet.")
            return

        diverged = [r for r in rows if r["ml_bucket"] != r["ml_bucket_canonical"]]
        if not diverged:
            print(f"📊 Canonical vs Latest: {len(rows)} scored days — buckets never diverged.")
            return

        canon_wins   = sum(1 for r in diverged if r["ml_result_canonical"] == "WIN")
        latest_wins  = sum(1 for r in diverged if r["ml_result"] == "WIN")
        n = len(diverged)
        print(f"\n📊 Canonical vs Latest accuracy on {n} days where prediction shifted:")
        print(f"   First-of-day (canonical): {canon_wins}/{n} = {canon_wins/n:.0%}")
        print(f"   Last-of-day  (latest):    {latest_wins}/{n} = {latest_wins/n:.0%}")

        both_win  = sum(1 for r in diverged if r["ml_result_canonical"] == "WIN" and r["ml_result"] == "WIN")
        flip_good = sum(1 for r in diverged if r["ml_result_canonical"] == "MISS" and r["ml_result"] == "WIN")
        flip_bad  = sum(1 for r in diverged if r["ml_result_canonical"] == "WIN"  and r["ml_result"] == "MISS")
        both_miss = sum(1 for r in diverged if r["ml_result_canonical"] == "MISS" and r["ml_result"] == "MISS")
        print(f"   Both WIN: {both_win}  |  Both MISS: {both_miss}  |  "
              f"Canon WIN→Latest MISS: {flip_bad}  |  Canon MISS→Latest WIN: {flip_good}")
        recommendation = "canonical (first)" if canon_wins >= latest_wins else "latest (last)"
        print(f"   → Recommend scoring by: {recommendation}\n")

    except Exception as e:
        print(f"⚠️ compare_canonical_vs_latest_accuracy failed: {e}")


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

    # Freeze once the actual high is recorded in the CSV.
    has_actual_today = any(
        r.get("forecast_or_actual") == "actual"
        and r.get("cli_date") == target_date_iso
        and _float_or_none(r.get("actual_high")) is not None
        for r in rows
    )

    existing = _fetch_existing_prediction(target_date_iso)

    # ml_recomputed tracks whether we actually ran the model this cycle.
    # When True  → include nws_d0/accuweather in the upsert so the stored
    #              baseline advances to the values the model just ran with.
    # When False → omit nws_d0/accuweather from the upsert so the stored
    #              baseline stays at the values used in the last real ML run,
    #              preserving the correct comparison point for future cycles.
    ml_recomputed = False

    # Determine whether we're before the D0 cutoff and the day isn't settled.
    # We only fetch live atmospheric features in this window — no point fetching
    # after the cutoff since the model is frozen, and not if the actual is in.
    past_cutoff = now_nyc().hour >= _D0_CUTOFF_HOUR_LOCAL.get(_CITY_KEY, 14)
    live_atm: Optional[dict] = None

    if has_actual_today and isinstance(existing, dict):
        # Day is over — actual recorded. Prediction is final.
        print(f"🔒 Actual high recorded — prediction frozen: {existing['ml_f']}°F → {existing.get('ml_bucket')}")
        ml = {
            "ml_f": existing["ml_f"],
            "ml_bucket": existing["ml_bucket"],
            "ml_confidence": existing["ml_confidence"],
            "ml_bucket_probs": existing.get("ml_bucket_probs"),
            "ml_version": existing.get("ml_version"),
        }

    elif isinstance(existing, dict) and existing.get("ml_f") is not None and past_cutoff:
        # Past the local D0 cutoff — freeze the canonical prediction.
        # After peak heating, agencies often echo the observed high rather than
        # forecasting; intraday curve features also reflect actual temps.
        # Holding the existing prediction prevents thermometer-chasing.
        cutoff_h = _D0_CUTOFF_HOUR_LOCAL.get(_CITY_KEY, 14)
        print(f"⏸️ Past D0 cutoff ({cutoff_h}:00 local) — holding canonical ML prediction "
              f"({existing['ml_f']}°F → {existing.get('ml_bucket')})")
        ml = {
            "ml_f": existing["ml_f"],
            "ml_bucket": existing["ml_bucket"],
            "ml_confidence": existing["ml_confidence"],
            "ml_bucket_probs": existing.get("ml_bucket_probs"),
            "ml_version": existing.get("ml_version"),
        }

    elif isinstance(existing, dict) and existing.get("ml_f") is not None:
        # A prediction exists and we're before the cutoff.
        # Check three independent triggers for recomputing ML:
        #   1. NWS revised their forecast by ≥1°F (agency new information)
        #   2. AccuWeather revised their forecast by ≥1°F (agency new information)
        #   3. Atmospheric conditions shifted materially (BL height, ensemble spread,
        #      HRRR-ECMWF divergence, ensemble mean) — independent of agencies,
        #      can fire BEFORE agencies update, giving us pre-market edge.
        stored_nws  = existing.get("nws_d0")
        stored_accu = existing.get("accuweather")

        nws_revised = (
            stored_nws is not None
            and nws_latest is not None
            and abs(float(nws_latest) - float(stored_nws)) >= 1.0
        )
        accu_revised = (
            stored_accu is not None
            and accu_latest is not None
            and abs(float(accu_latest) - float(stored_accu)) >= 1.0
        )

        # Fetch live atmospheric data and compare to morning snapshot.
        # This is the key intraday signal: GFS updates 4x/day, ECMWF 2x/day,
        # HRRR hourly. Fresh model runs appear here 1-3 hours before agencies
        # update public forecasts and before market prices shift.
        stored_atm_snapshot = existing.get("atm_snapshot")
        try:
            stored_snapshot_dict = (
                json.loads(stored_atm_snapshot)
                if isinstance(stored_atm_snapshot, str)
                else (stored_atm_snapshot or {})
            )
        except Exception:
            stored_snapshot_dict = {}

        try:
            live_atm = _fetch_atmospheric_features(target_date_iso)
        except Exception as _atm_e:
            print(f"⚠️ Live atmospheric fetch failed: {_atm_e}")
            live_atm = {}

        # Also fetch live NWS station observations for ground-truth trigger check.
        # This catches cold starts, slow heating, and reality diverging from the
        # Open-Meteo intraday forecast — BEFORE agencies update public forecasts.
        try:
            live_obs = _fetch_observation_features(
                target_date_iso,
                nws_last=nws_latest,
                atm_features=live_atm,
            )
        except Exception as _obs_e:
            print(f"⚠️ Live obs fetch failed: {_obs_e}")
            live_obs = {}

        atm_triggered, atm_reasons = _check_atmospheric_shift(live_atm, stored_atm_snapshot)
        obs_triggered, obs_reasons = _check_obs_trigger(live_obs, stored_snapshot_dict)

        if nws_revised or accu_revised or atm_triggered or obs_triggered:
            trigger_reasons = []
            if nws_revised:
                trigger_reasons.append(f"NWS {stored_nws:.0f}→{nws_latest:.0f}°F")
            if accu_revised:
                trigger_reasons.append(f"AccuWeather {stored_accu:.0f}→{accu_latest:.0f}°F")
            if atm_triggered:
                trigger_reasons.extend(atm_reasons)
            if obs_triggered:
                trigger_reasons.extend(obs_reasons)
            print(f"🔄 ML recompute triggered: {', '.join(trigger_reasons)}")
            # Pass prefetched atmospheric data — avoids redundant API calls (~15-20s)
            ml = _compute_ml_prediction(rows, target_date_iso, prefetched_atm=live_atm)
            ml_recomputed = True
            if ml:
                _log_ml_revision(target_date_iso, "today_for_today", ml, ", ".join(trigger_reasons))

        elif stored_nws is None and stored_accu is None:
            # No stored baseline (row predates this logic). Recompute once to
            # establish nws_d0/accuweather/atm_snapshot baseline going forward.
            print(f"🔄 No stored forecast baseline — recomputing to establish "
                  f"(previous: {existing['ml_f']}°F → {existing.get('ml_bucket')})")
            ml = _compute_ml_prediction(rows, target_date_iso, prefetched_atm=live_atm)
            ml_recomputed = True
            if ml:
                _log_ml_revision(target_date_iso, "today_for_today", ml, "baseline_reestablish")

        else:
            # No revision since last ML run — hold the existing prediction.
            nws_disp  = f"{nws_latest:.0f}°F"  if nws_latest  is not None else "N/A"
            accu_disp = f"{accu_latest:.0f}°F" if accu_latest is not None else "N/A"
            print(f"⏸️ No triggers fired (NWS={nws_disp}, AccuWeather={accu_disp} unchanged, "
                  f"atm stable) — ML held: {existing['ml_f']}°F → {existing.get('ml_bucket')}")
            ml = {
                "ml_f": existing["ml_f"],
                "ml_bucket": existing["ml_bucket"],
                "ml_confidence": existing["ml_confidence"],
                "ml_bucket_probs": existing.get("ml_bucket_probs"),
                "ml_version": existing.get("ml_version"),
            }

    else:
        # No existing prediction (LOCK_NOT_FOUND or LOCK_ERROR) — compute fresh.
        # Don't pre-fetch atmospheric here; _compute_ml_prediction will fetch inside.
        if isinstance(existing, dict):
            print(f"🔄 Recomputing ML prediction (previous: {existing.get('ml_f')}°F → {existing.get('ml_bucket')})")
        ml = _compute_ml_prediction(rows, target_date_iso)
        ml_recomputed = True
        # Fetch obs for snapshot storage on canonical write (LOCK_NOT_FOUND path).
        # This is a separate call from what _compute_ml_prediction fetched internally —
        # we need the obs dict here to store the baseline in atm_snapshot.
        if existing is _LOCK_NOT_FOUND:
            try:
                live_obs = _fetch_observation_features(
                    target_date_iso,
                    nws_last=nws_latest,
                    atm_features=live_atm or {},
                )
            except Exception:
                live_obs = {}
        # Log immediately for LOCK_ERROR case — canonical write logs separately below
        # for LOCK_NOT_FOUND. Avoids double-logging on canonical write.
        if ml and existing is not _LOCK_NOT_FOUND:
            _log_ml_revision(target_date_iso, "today_for_today", ml, "lock_error_recompute")

    # Canonical = the first non-null ML prediction for this date.
    # _LOCK_NOT_FOUND covers both "no row yet" and "row exists but ml_f is null".
    # merge-duplicates semantics mean omitting ml_bucket_canonical on subsequent
    # upserts preserves the value set here — it is never overwritten.
    is_canonical_write = (existing is _LOCK_NOT_FOUND) and ml is not None

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
        "rep_forecast": today_pre_mean,
        "bias_applied": avg_bias_excl_today,
        "version": MODEL_VERSION,
        "recommendation": "live — updates on agency forecast revision",
        "source_card": "nws_auto_logger",
        "city": _CITY_KEY,
    }
    # Only write nws_d0/accuweather/atm_snapshot when ML actually ran this cycle.
    # This preserves the stored baseline (the values the model last ran with)
    # so future cycles can detect genuine forecast revisions correctly.
    if ml_recomputed:
        payload["nws_d0"] = nws_latest
        payload["accuweather"] = accu_latest
        # Advance atmospheric + obs snapshot to the current live values so the next
        # comparison starts from the post-revision baseline, not the morning one.
        # (Only on non-canonical writes — canonical write handled separately below.)
        if not is_canonical_write and live_atm and any(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in live_atm.values()
        ):
            snap = {k: live_atm[k] for k in _ATM_SNAPSHOT_KEYS if k in live_atm}
            if snap:
                # Also advance obs snapshot keys so triggers don't re-fire on same reading
                _add_obs_to_snap(snap, live_obs)
                payload["atm_snapshot"] = json.dumps(snap)
                print(f"📸 Atmospheric baseline advanced after recompute ({len(snap)} keys)")
    if ml:
        payload["ml_f"] = ml["ml_f"]
        payload["ml_bucket"] = ml["ml_bucket"]
        payload["ml_confidence"] = ml["ml_confidence"]
        if ml.get("ml_bucket_probs"):
            payload["ml_bucket_probs"] = ml["ml_bucket_probs"]
        if ml.get("ml_version"):
            payload["ml_version"] = ml["ml_version"]
        if ml.get("ml_direct_bucket"):
            payload["ml_direct_bucket"] = ml["ml_direct_bucket"]
        # Canonical fields — written once on first non-null ML prediction.
        # Subsequent upserts omit these fields so merge-duplicates preserves them.
        payload["is_canonical"] = is_canonical_write
        if is_canonical_write:
            payload["ml_bucket_canonical"] = ml["ml_bucket"]
            payload["ml_f_canonical"] = ml["ml_f"]
            print(f"🏛️ Canonical prediction set: {ml['ml_f']}°F → {ml['ml_bucket']}")
            _log_ml_revision(target_date_iso, "today_for_today", ml, "first_write")
            # Store morning atmospheric + obs baseline on canonical write.
            # On subsequent runs, live_atm and live_obs are compared against this
            # snapshot to detect intraday shifts BEFORE agencies update forecasts.
            # Only store if we have valid atmospheric data.
            if live_atm and any(
                v is not None and not (isinstance(v, float) and math.isnan(v))
                for v in live_atm.values()
            ):
                snap = {k: live_atm[k] for k in _ATM_SNAPSHOT_KEYS if k in live_atm}
                if snap:
                    # Add obs snapshot keys — critical for cold-start trigger detection
                    _add_obs_to_snap(snap, live_obs)
                    payload["atm_snapshot"] = json.dumps(snap)
                    obs_populated = sum(
                        1 for v in (live_obs or {}).values()
                        if v is not None and not (isinstance(v, float) and math.isnan(v))
                    )
                    print(f"📸 Baseline stored: {len(snap)} atm keys, "
                          f"obs_hour={snap.get('obs_snap_hour')}, "
                          f"obs_temp={snap.get('obs_snap_temp')}, "
                          f"obs_populated={obs_populated}")

    # Fetch Kalshi market odds + compute bet signal
    market_probs = _fetch_kalshi_market_probs(target_date_iso)
    if market_probs:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    # Map ML prediction → Kalshi's actual bucket structure for today
    # Always re-map with fresh prediction (no lock — intraday re-prediction enabled)
    if ml and market_probs:
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

    # Intraday re-prediction for tomorrow: always recompute ML.
    # Today's observations (obs_temp_vs_forecast_max) inform tomorrow's prediction.
    # As today's high climbs, that signal flows into tomorrow's forecast.
    existing = _fetch_existing_prediction(tomorrow_iso)
    tomorrow_market_probs = _fetch_kalshi_market_probs(tomorrow_iso)
    if isinstance(existing, dict):
        print(f"🔄 Recomputing tomorrow's prediction (previous: {existing['ml_f']}°F → {existing.get('ml_bucket')})")
    ml = _compute_ml_prediction(rows, tomorrow_iso)
    if ml:
        trigger = "first_write" if existing is _LOCK_NOT_FOUND else "intraday_refresh"
        _log_ml_revision(tomorrow_iso, "today_for_tomorrow", ml, trigger)

    if bcp_tm is None and ml is None:
        print("⏭️ today_for_tomorrow: no BCP data and no ML prediction available."); return

    # Canonical = first non-null ML prediction for tomorrow's date (from today's D1 run).
    is_canonical_write = (existing is _LOCK_NOT_FOUND) and ml is not None

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
        if ml.get("ml_direct_bucket"):
            payload["ml_direct_bucket"] = ml["ml_direct_bucket"]
        # Canonical fields for D1 — set once on first non-null ML write for this date.
        payload["is_canonical"] = is_canonical_write
        if is_canonical_write:
            payload["ml_bucket_canonical"] = ml["ml_bucket"]
            payload["ml_f_canonical"] = ml["ml_f"]
            print(f"🏛️ Canonical D1 prediction set: {ml['ml_f']}°F → {ml['ml_bucket']}")

    # Use already-fetched Kalshi market odds (from lock check above)
    market_probs = tomorrow_market_probs
    if market_probs:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    # Map ML prediction → Kalshi's actual bucket structure for tomorrow
    # Always re-map with fresh prediction (no lock — intraday re-prediction enabled)
    if ml and market_probs:
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
    rows, _ = _read_all_rows()
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
    # Weekly: compare first-of-day vs last-of-day prediction accuracy
    try:
        if today_nyc().weekday() == 0:  # Monday only — keeps logs clean
            compare_canonical_vs_latest_accuracy()
    except Exception as e: print("⚠️ compare_canonical_vs_latest_accuracy failed:", e)


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
    s.add_parser("collect_obs")
    s.add_parser("backfill_obs")
    args = p.parse_args()

    set_city(args.city)
    global _CITY_KEY
    _CITY_KEY = args.city
    print(f"[prediction_writer] city={args.city}")

    if args.cmd == "collect_obs":        collect_nws_observations(args.city)
    elif args.cmd == "backfill_obs":     backfill_observation_features(args.city)
    elif args.cmd == "today_for_today":    write_today_for_today(args.date)
    elif args.cmd == "today_for_tomorrow": write_today_for_tomorrow(args.date)
    else: write_both_snapshots()

if __name__ == "__main__": _cli()
