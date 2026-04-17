# prediction_writer.py
from __future__ import annotations
import os, json, argparse, pickle, math, urllib.request
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Atmospheric snapshot caching (graceful fallback during API failures)
from atm_cache import (
    get_cached_snapshot, cache_snapshot, fill_missing_from_cache,
)

# reuse your existing helpers from nws_auto_logger.py (leave that file alone)
from nws_auto_logger import (
    now_nyc, today_nyc, _read_all_rows,
    _compute_avg_bias_excluding, _compute_today_pre_high_mean,
    _float_or_none, compute_today_gate_f,
)
from model_config import (
    FEATURE_COLS, FEATURE_COLS_V2, FEATURE_COLS_V4, ACCU_NWS_FALLBACK,
    ATM_PREDICTOR_INPUT_COLS, OBSERVATION_COLS, REGIONAL_OBS_COLS,
    NWS_SEQUENCE_COLS, AMBIENT_OBS_COLS, SYNOPTIC_OBS_COLS, NYSM_OBS_COLS,
    derive_bucket_probabilities,
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
# Dynamic agency cutoff: use seasonal cutoff from city_config
# Falls back to hardcoded defaults if not available
def _get_agency_cutoff_hour(city_key: str) -> int:
    """Get agency cutoff hour for current date and city (dynamic by season)."""
    try:
        from city_config import get_seasonal_agency_cutoff
        current_month = datetime.datetime.now().month
        return get_seasonal_agency_cutoff(current_month)
    except Exception:
        # Fallback to static defaults if import fails
        return {"nyc": 14, "lax": 14}.get(city_key, 14)

# Atmospheric-only cutoff — 1 hour later than agency cutoff
# BL height, 925mb winds, and GFS/HRRR/ECMWF ensemble data are MODEL-derived,
# not surface-observation-derived. A collapsing BL is independent evidence the
# mixing layer is done without looking at the thermometer — no contamination risk.
def _get_atm_cutoff_hour(city_key: str) -> int:
    """Get atmospheric cutoff hour (agency_cutoff + 1 hour)."""
    return _get_agency_cutoff_hour(city_key) + 1

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


def _load_v5_models():
    """Load v5 models: bcp_v5 regressor + classifier + feature cols (cached).
    v5 = v4 architecture + 3 HIGH_TIMING_COLS (126 total features).
    HIGH_TIMING_COLS (obs_high_peak_hour, obs_is_overnight_high, obs_temp_falling_hrs)
    are NaN for D+1 / overnight predictions — HistGBT handles NaN natively, so we
    never fall through to v4 just because they're missing.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v5_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v5_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v5_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v5_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v5 regression model (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v5 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v5_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v5_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v5_classifier.pkl"
                )
                print(f"✅ Loaded v5 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v5 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v5_feature_cols.pkl"):
                with open(f"{prefix}bcp_v5_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v5_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v5 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v5_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v5_feature_cols"),
    )


def _load_v8_models():
    """Load v8 models: bcp_v8 regressor + classifier + feature cols (cached).
    v8 = v7 (HRRR-anchored base, 138 features) + obs_heating_rate_delta (stall signal).
    obs_heating_rate_delta = recent_slope - early_slope (°F/hr): negative = cap holding.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v8_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v8_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v8_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v8_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v8 regression model [HRRR-anchored + stall signal] (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v8 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v8_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v8_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v8_classifier.pkl"
                )
                print(f"✅ Loaded v8 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v8 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v8_feature_cols.pkl"):
                with open(f"{prefix}bcp_v8_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v8_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v8 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v8_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v8_feature_cols"),
    )


def _load_v10_models():
    """Load v10 models: bcp_v10 regressor + classifier + feature cols (cached).
    v10 = v9 (149 features) + MANH Manhattan Mesonet 5-min fill-in (2 features = 151 total).
    obs_manh_temp / obs_manh_vs_knyc — only available after MANH goes live in Synoptic pull.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v10_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v10_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v10_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v10_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v10 regression model [MANH 5-min Mesonet] (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v10 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v10_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v10_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v10_classifier.pkl"
                )
                print(f"✅ Loaded v10 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v10 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v10_feature_cols.pkl"):
                with open(f"{prefix}bcp_v10_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v10_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v10 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v10_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v10_feature_cols"),
    )


def _load_v11_models():
    """Load v11 models: bcp_v11 regressor + classifier + feature cols (cached).
    v11 = v10 (151 features) + model-vs-NWS divergence (3 features = 154 total).
    mm_hrrr_vs_nws / mm_nbm_vs_nws / mm_mean_vs_nws — derived at inference time.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v11_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v11_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v11_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v11_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v11 regression model [model-vs-NWS divergence] (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v11 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v11_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v11_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v11_classifier.pkl"
                )
                print(f"✅ Loaded v11 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v11 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v11_feature_cols.pkl"):
                with open(f"{prefix}bcp_v11_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v11_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v11 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v11_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v11_feature_cols"),
    )


def _load_v9_models():
    """Load v9 models: bcp_v9 regressor + classifier + feature cols (cached).
    v9 = v8 (HRRR-anchored + stall signal, 139 features) + 10 named Synoptic station
    features (obs_kjfk_temp, obs_klga_temp, obs_kewr_temp, obs_kteb_temp, obs_knyc_temp,
    obs_kjfk_vs_knyc, obs_klga_vs_knyc, obs_kewr_vs_knyc, obs_coastal_vs_inland,
    obs_airport_spread) = 149 total features.
    Only available after enough Synoptic backfill rows exist (≥30 with obs_kjfk_temp).
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v9_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v9_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v9_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v9_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v9 regression model [HRRR+stall+Synoptic stations] (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v9 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v9_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v9_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v9_classifier.pkl"
                )
                print(f"✅ Loaded v9 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v9 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v9_feature_cols.pkl"):
                with open(f"{prefix}bcp_v9_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v9_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v9 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v9_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v9_feature_cols"),
    )


def _load_v6_models():
    """Load v6 models: bcp_v6 regressor + classifier + feature cols (cached).
    v6 = v5 + NBM, GEM HRDPS, HRRR-specific 925mb, OKX radiosonde (138 features).
    Same AccuWeather/NWS base as v5 — v7 is the HRRR-anchored upgrade.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v6_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v6_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v6_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v6_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v6 regression model (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v6 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v6_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v6_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v6_classifier.pkl"
                )
                print(f"✅ Loaded v6 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v6 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v6_feature_cols.pkl"):
                with open(f"{prefix}bcp_v6_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v6_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v6 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v6_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v6_feature_cols"),
    )


def _load_v7_models():
    """Load v7 models: bcp_v7 regressor + classifier + feature cols (cached).
    v7 = v6 features (138) + HRRR-anchored training base.
    Key: regressor predicts (actual - HRRR_max) bias, not (actual - AccuWeather) bias.
    At inference: center = HRRR_max + bias when HRRR available, NBM_max + bias second.
    """
    import nws_auto_logger as _nal
    prefix = _nal._CITY_CFG.get("model_prefix", "")
    cache_key = f"{prefix}v7_regressor"
    if cache_key not in _ML_MODEL_CACHE:
        _ML_MODEL_CACHE[cache_key] = None
        _ML_MODEL_CACHE[f"{prefix}v7_classifier"] = None
        _ML_MODEL_CACHE[f"{prefix}v7_feature_cols"] = None

        try:
            with open(f"{prefix}bcp_v7_regressor.pkl", "rb") as f:
                _ML_MODEL_CACHE[cache_key] = pickle.load(f)
            print(f"✅ Loaded v7 regression model [HRRR-anchored] (prefix='{prefix}')")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ v7 regression load error: {e}")

        try:
            from train_classifier import BucketClassifier
            if os.path.exists(f"{prefix}bcp_v7_classifier.pkl"):
                _ML_MODEL_CACHE[f"{prefix}v7_classifier"] = BucketClassifier.load(
                    f"{prefix}bcp_v7_classifier.pkl"
                )
                print(f"✅ Loaded v7 bucket classifier (prefix='{prefix}')")
        except Exception as e:
            print(f"⚠️ v7 classifier load error: {e}")

        try:
            if os.path.exists(f"{prefix}bcp_v7_feature_cols.pkl"):
                with open(f"{prefix}bcp_v7_feature_cols.pkl", "rb") as f:
                    _ML_MODEL_CACHE[f"{prefix}v7_feature_cols"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ v7 feature cols load error: {e}")

    return (
        _ML_MODEL_CACHE.get(cache_key),
        _ML_MODEL_CACHE.get(f"{prefix}v7_classifier"),
        _ML_MODEL_CACHE.get(f"{prefix}v7_feature_cols"),
    )


def _detect_high_locked(target_date_iso: str,
                        nws_forecast: float | None = None) -> dict:
    """
    Dynamically detect whether today's calendar-day high has already been recorded,
    regardless of clock time.  Handles three distinct meteorological regimes:

      1. Overnight / pre-dawn high  — warm front passes at 1–3am, then cold air floods in.
         Signature: obs_max occurred before 09:00 local AND current temp is well below it.
         Suppressed if NWS is still forecasting ≥2°F above the overnight max, which means
         the daytime high has not yet been reached (e.g. overnight spike of 59°F but NWS
         still calls for 62°F → do not lock at 8am).

      2. Late-afternoon / evening high — summer sea-breeze collapse, afternoon convection.
         The hard 3pm atm_cutoff already extends coverage here, but if the high has
         *clearly* peaked (temp has been falling for 2+ consecutive hours, down ≥3°F from
         the day's max), we lock early regardless of clock time.

      3. Classic mid-afternoon high — normal solar-forced heating; the existing 2pm/3pm
         clock cutoffs handle this fine.

    Args:
        target_date_iso: ISO date string for today.
        nws_forecast:    Latest NWS high forecast for today (°F).  When provided,
                         Regime 1 is suppressed if NWS is still calling for ≥2°F
                         above the overnight max — indicating the real high is still ahead.

    Returns a dict:
      {
        "locked":       bool,          # True → freeze the prediction now
        "reason":       str,           # human-readable explanation
        "obs_high_f":   float | None,  # calendar-day max observed so far
        "obs_high_hour":int | None,    # local hour when that max occurred
        "current_f":    float | None,  # most recent observed temp
        "falling_hrs":  int,           # consecutive hours the temp has been falling
      }
    """
    result = dict(locked=False, reason="", obs_high_f=None,
                  obs_high_hour=None, current_f=None, falling_hrs=0)
    try:
        import nws_auto_logger as _nal
        cfg   = _nal._CITY_CFG
        station  = cfg.get("obs_station", "KNYC")
        tz_name  = cfg.get("timezone", "America/New_York")

        today = today_nyc().isoformat()
        if target_date_iso != today:
            return result  # only makes sense for today

        url = (f"https://api.weather.gov/stations/{station}"
               f"/observations?limit=150")
        req = urllib.request.Request(url, headers={
            "Accept": "application/geo+json",
            "User-Agent": "nws-forecast-logger/1.0",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)

        # Build chronological list of (hour, temp_f, datetime_local) for today
        obs_today: list[tuple[datetime, int, float]] = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            ts    = props.get("timestamp", "")
            if not ts:
                continue
            obs_dt    = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            obs_local = obs_dt.astimezone(tz)
            if obs_local.strftime("%Y-%m-%d") != target_date_iso:
                continue
            temp_c = props.get("temperature", {}).get("value")
            if temp_c is None:
                continue
            temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)
            obs_today.append((obs_local, obs_local.hour, temp_f))

        if not obs_today:
            return result

        # Sort oldest→newest
        obs_today.sort(key=lambda x: x[0])

        # Day's running max
        max_f    = max(o[2] for o in obs_today)
        max_hour = next(o[1] for o in reversed(obs_today) if o[2] == max_f)
        # Most recent reading
        current_f    = obs_today[-1][2]
        current_hour = obs_today[-1][1]

        result["obs_high_f"]    = max_f
        result["obs_high_hour"] = max_hour
        result["current_f"]     = current_f

        # ── Consecutive falling hours ─────────────────────────────────────
        # Walk backwards through hourly obs to count how long the temp has
        # been trending down from the peak.
        falling = 0
        prev_t  = current_f
        for dt_obs, _hr, t in reversed(obs_today[:-1]):
            if t > prev_t:
                break   # found an obs that was warmer than the one after it → stop
            falling += 1
            prev_t = t
        result["falling_hrs"] = falling

        gap = max_f - current_f  # °F below day's max right now

        now_hour = now_nyc().hour

        # ── Regime 1: Overnight / pre-dawn high ──────────────────────────
        if max_hour < 9 and gap >= 2.0 and now_hour >= 8:
            # Suppress if NWS is still forecasting substantially above the overnight
            # high — the real daytime high hasn't arrived yet.
            # Example: overnight spike 59°F, NWS says 62°F → 3°F margin → don't lock.
            if nws_forecast is not None and nws_forecast >= max_f + 2.0:
                print(
                    f"🌡️ Overnight lock suppressed: NWS forecasting {nws_forecast:.0f}°F, "
                    f"{nws_forecast - max_f:.1f}°F above overnight peak "
                    f"({max_f}°F at {max_hour:02d}:00) — daytime high not yet set"
                )
                # Fall through to NOT locked
            else:
                result["locked"] = True
                result["reason"] = (
                    f"Overnight high: {max_f}°F at {max_hour:02d}:00, "
                    f"currently {current_f}°F ({gap:.1f}°F below peak)"
                )
                print(f"🌙 {result['reason']}")
                return result

        # ── Regime 2: Late high clearly past its peak ────────────────────
        # Require: at least noon, 3°F gap, 2 consecutive falling hours
        if now_hour >= 12 and gap >= 3.0 and falling >= 2:
            result["locked"] = True
            result["reason"] = (
                f"Dynamic lock: {max_f}°F at {max_hour:02d}:00, "
                f"{gap:.1f}°F below peak, falling {falling}h consecutive"
            )
            print(f"📉 {result['reason']}")
            return result

        print(
            f"🌡️ High detector: max={max_f}°F @{max_hour:02d}:00, "
            f"current={current_f}°F, gap={gap:.1f}°F, falling={falling}h"
            f" → NOT locked"
        )
    except Exception as e:
        print(f"⚠️ _detect_high_locked: {e}")
    return result


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


def _get_nws_d1_final(target_date_iso: str) -> Optional[float]:
    """
    Return the last NWS forecast for target_date that was issued the day BEFORE it.
    Captures the overnight jump signal: D-1 final vs D0 morning.
    April 10 2026: D-1 final=63°F, D0 3am=66°F → +3°F overnight jump → actual=63°F.
    """
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        nws_csv = cfg.get("nws_csv", "nws_forecast_log.csv")
        import pandas as pd
        df = pd.read_csv(nws_csv)
        df["forecast_time"] = pd.to_datetime(df["forecast_time"], errors="coerce")
        df["target_date_str"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date.astype(str)
        target = pd.Timestamp(target_date_iso).date()
        d1_date = target - pd.Timedelta(days=1)
        d1_rows = df[
            (df["target_date_str"] == target_date_iso) &
            (df["forecast_or_actual"] == "forecast") &
            (df["forecast_time"].dt.date == d1_date)
        ].dropna(subset=["predicted_high"])
        if d1_rows.empty:
            return None
        return float(d1_rows.sort_values("forecast_time").iloc[-1]["predicted_high"])
    except Exception as e:
        print(f"  ⚠️ _get_nws_d1_final: {e}")
        return None


def _fetch_mos_forecast(target_date_iso: str) -> Optional[float]:
    """
    Fetch GFS-MOS max temperature forecast via Iowa State University Mesonet API.

    Iowa State provides parsed MOS guidance as a CSV, which is far more reliable
    than scraping the NWS MEX HTML text product. GFS-MOS runs at 00Z and 12Z UTC.

    For NYC, uses KJFK (closest NWS upper-air/MOS station to KNYC).

    Returns the MOS max temp as a float in °F, or None on failure.
    """
    import csv
    import io
    import nws_auto_logger as _nal

    cfg = _nal._CITY_CFG
    # MOS station: use city-specific config or fall back to KJFK for NYC
    mos_station = cfg.get("mos_station", "KJFK")

    try:
        target_dt = datetime.strptime(target_date_iso, "%Y-%m-%d")
    except ValueError:
        return None

    # Try today's 12Z run first (8 AM EDT), then 00Z (midnight), then prior day 12Z
    candidate_runtimes = [
        target_dt.strftime("%Y%m%d") + "12",     # 12Z = 8 AM EDT same day
        target_dt.strftime("%Y%m%d") + "00",     # 00Z = midnight EDT same day
        (target_dt - timedelta(days=1)).strftime("%Y%m%d") + "12",  # prior day 12Z
    ]

    for runtime_str in candidate_runtimes:
        url = (
            f"https://mesonet.agron.iastate.edu/mos/csv.php"
            f"?station={mos_station}&runtime={runtime_str}&model=GFS"
        )
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "nws-forecast-logger/1.0",
                "Accept": "text/csv,text/plain",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")

            if not raw.strip() or "error" in raw.lower()[:200]:
                continue

            reader = csv.DictReader(io.StringIO(raw))
            rows = list(reader)
            if not rows:
                continue

            # Iowa State MOS CSV has columns including 'ftime' (forecast valid time)
            # and 'tmp' (temperature) and derived 'xmx' or 'maxt' for daily max.
            # The max temp for a forecast day is in the row where ftime matches
            # the afternoon of the target date.

            # Strategy 1: look for a 'maxt' or 'xmx' column (daily max directly)
            max_col = next((c for c in (rows[0].keys() if rows else [])
                           if c.strip().lower() in ("maxt", "xmx", "x", "max_t", "tmax")), None)

            # Strategy 2: look for forecast valid times near 2 PM on the target date
            target_afternoon = target_dt.replace(hour=18)  # 18Z = 2 PM EDT

            best_max = None
            for row in rows:
                ftime_raw = (row.get("ftime") or row.get("valid") or "").strip()
                if not ftime_raw:
                    continue
                try:
                    ftime = datetime.strptime(ftime_raw[:16], "%Y-%m-%d %H:%M")
                except ValueError:
                    try:
                        ftime = datetime.strptime(ftime_raw[:13], "%Y%m%d%H%M"[:len(ftime_raw)])
                    except ValueError:
                        continue

                # Only look at forecast times on the target date
                if ftime.date() != target_dt.date():
                    continue

                # Try direct max-temp column
                if max_col:
                    v = (row.get(max_col) or "").strip()
                    if v and v not in ("", "None", "N/A", "M"):
                        try:
                            best_max = float(v)
                            break
                        except ValueError:
                            pass

                # Fallback: temperature at 18Z (closest to max temp hour)
                if abs((ftime - target_afternoon).total_seconds()) <= 3 * 3600:
                    tmp_col = next((c for c in row.keys()
                                   if c.strip().lower() in ("tmp", "temp", "t")), None)
                    if tmp_col:
                        v = (row.get(tmp_col) or "").strip()
                        if v and v not in ("", "None", "N/A", "M"):
                            try:
                                best_max = float(v)
                            except ValueError:
                                pass

            if best_max is not None:
                print(f"🌡️ GFS-MOS ({mos_station}) {runtime_str}Z: {best_max:.0f}°F max for {target_date_iso}")
                return best_max

        except Exception as e:
            print(f"  MOS {runtime_str}: {e}")
            continue

    print(f"  MOS: no GFS-MOS data found for {target_date_iso} at {mos_station}")
    return None


def _fetch_atmospheric_features(target_date_iso: str) -> dict:
    """Fetch atmospheric features from Open-Meteo + radiosonde for today/tomorrow."""
    try:
        import nws_auto_logger as _nal
        cfg = _nal._CITY_CFG
        from open_meteo_client import get_atmospheric_features_live
        lat = cfg.get("open_meteo_lat", 40.7834)
        lon = cfg.get("open_meteo_lon", -73.965)
        tz = cfg.get("timezone", "America/New_York")
        features = get_atmospheric_features_live(lat, lon, target_date_iso, tz)

        # v6: fetch OKX radiosonde upper-air sounding (observed 925mb/850mb temps)
        # OKX station for NYC; other cities use their nearest upper-air station
        raob_station = cfg.get("raob_station", "OKX")
        try:
            from raob_client import get_raob_features
            gfs_925 = features.get("atm_925mb_temp_mean")
            hrrr_925 = features.get("atm_925mb_hrrr_mean")
            raob_features = get_raob_features(
                target_date_iso,
                station=raob_station,
                gfs_925mb_mean=gfs_925 if (gfs_925 and not math.isnan(gfs_925)) else None,
                hrrr_925mb_mean=hrrr_925 if (hrrr_925 and not math.isnan(hrrr_925)) else None,
            )
            features.update(raob_features)
        except Exception as raob_e:
            print(f"  ⚠️ Radiosonde fetch skipped: {raob_e}")
            for col in ["raob_925mb_temp", "raob_850mb_temp", "raob_700mb_temp",
                        "raob_925mb_gfs_diff", "raob_925mb_hrrr_diff", "raob_sounding_hour", "raob_valid"]:
                features.setdefault(col, float("nan"))

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
    bucket_just_changed: bool = False,
) -> tuple[str, float, float]:
    """
    Compute bet signal using Kelly criterion for principled sizing.

    The ML pick (ml_bucket) is always the signal — market price only
    determines how much to bet, not whether to bet.

    Half-Kelly fraction:
        kelly_f = 0.5 * (p_model - p_market) / (1 - p_market)

    bucket_just_changed: if True, downgrade signal one tier — a bucket that
        just shifted hasn't proven stability yet, so we shouldn't call it
        STRONG BET on the same cycle it changed.

    Returns (signal, edge, kelly_fraction) where:
      signal:        "STRONG_BET" / "BET" / "LEAN" / "SKIP"
      edge:          ml_confidence - market_prob (raw edge)
      kelly_fraction: half-Kelly fraction of bankroll to wager (0 if no edge)
    """
    market_prob = market_probs.get(ml_bucket, 0.0) if market_probs else 0.0
    edge = round(ml_confidence - market_prob, 4)

    # Half-Kelly: requires positive market price to compute odds
    if market_probs and 0.0 < market_prob < 1.0:
        kelly_full = (ml_confidence - market_prob) / (1.0 - market_prob)
        kelly_half = round(max(kelly_full * 0.5, 0.0), 4)
    else:
        # No live market price — can still pick but can't size
        kelly_half = 0.0

    # Signal derived from Kelly fraction (principled, not arbitrary thresholds)
    if kelly_half >= 0.15:
        signal = "STRONG_BET"
    elif kelly_half >= 0.08:
        signal = "BET"
    elif kelly_half >= 0.03:
        signal = "LEAN"
    else:
        signal = "SKIP"

    # Stability guard: if the bucket just shifted, downgrade one tier.
    # A fresh revision hasn't proven itself stable — don't show STRONG BET
    # on the same cycle the model changed its mind.
    if bucket_just_changed and signal != "SKIP":
        _downgrade = {"STRONG_BET": "BET", "BET": "LEAN", "LEAN": "SKIP"}
        _orig = signal
        signal = _downgrade.get(signal, signal)
        print(f"  ⬇️  Bet signal downgraded {_orig} → {signal} (bucket just shifted — waiting for next cycle to confirm)")

    return signal, edge, kelly_half


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

    # ─ DEBUG: Log forecast values and detect anomalies ─
    accu_anomaly_override = False
    if len(nws_fc) > 0 or len(accu_fc) > 0:
        print(f"\n  📊 Forecast debug for {target_date_iso}:")
        if nws_fc:
            print(f"    NWS ({len(nws_fc)} forecasts): {' → '.join([f'{v}°F@{ts[:19]}' for ts, v in nws_fc])}")
            print(f"    NWS last={nws_vals[-1]:.1f}°F, trend={nws_vals[-1]-nws_vals[0]:+.1f}°F")
        if accu_fc:
            print(f"    AccuWeather ({len(accu_fc)} forecasts): {' → '.join([f'{v}°F@{ts[:19]}' for ts, v in accu_fc])}")
            print(f"    AccuWeather last={accu_vals[-1]:.1f}°F, trend={accu_vals[-1]-accu_vals[0]:+.1f}°F")
        # Anomaly detection: flag if one source is >5°F different from the other (recent issue)
        if has_accu and len(nws_vals) > 0:
            delta = abs(accu_vals[-1] - nws_vals[-1])
            if delta > 5.0:
                print(f"    ⚠️  ANOMALY: AccuWeather vs NWS delta={delta:.1f}°F (AccuWeather may be stale)")
                # Check if AccuWeather has made a sudden jump
                if len(accu_vals) > 1:
                    accu_jump = abs(accu_vals[-1] - accu_vals[-2])
                    nws_jump = abs(nws_vals[-1] - nws_vals[-2]) if len(nws_vals) > 1 else 0.0
                    if accu_jump > 5.0:
                        print(f"    ⚠️  AccuWeather jumped {accu_jump:.1f}°F in last update (likely API glitch/cutover)")
                    # CONSENSUS CHECK: if AccuWeather made sudden jump but NWS is stable → override with NWS
                    if accu_jump > 8.0 and nws_jump < 2.0:
                        print(f"    🔧 CORRECTION: AccuWeather anomaly ({accu_jump:.1f}°F jump) while NWS stable → using NWS value {nws_vals[-1]:.1f}°F")
                        accu_anomaly_override = True

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
        accu_last_val = float(accu_vals[-1])
        # Apply anomaly override if AccuWeather jumped but NWS didn't
        if accu_anomaly_override:
            accu_last_val = float(nws_vals[-1])
        features.update({
            "accu_first": float(accu_vals[0]),
            "accu_last": accu_last_val,
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
    # stale data.  AccuWeather lags in two windows:
    #  • Early morning (< 10am local): mirrors yesterday's airmass before the morning
    #    briefing publishes — a 3°F NWS gap is enough to flag it as stale.
    #  • Anytime: >8°F divergence = clearly outdated (yesterday's warm air, etc).
    # Fall back to NWS in both cases — a small bias mismatch beats a wrong anchor.
    if has_accu:
        spread = abs(features["nws_last"] - features["accu_last"])
        # City-aware local hour for early-morning threshold
        try:
            import nws_auto_logger as _nal_am
            _am_tz = _nal_am._CITY_CFG.get("timezone", "America/New_York")
            from zoneinfo import ZoneInfo as _ZI
            _local_hour_now = datetime.now(_ZI(_am_tz)).hour
        except Exception:
            _local_hour_now = now_nyc().hour
        _accu_stale_threshold = 3.0 if _local_hour_now < 10 else 8.0
        _accu_stale_label = (
            f"early morning (< 10am, threshold={_accu_stale_threshold:.0f}°F)"
            if _accu_stale_threshold < 8 else f"threshold={_accu_stale_threshold:.0f}°F"
        )
        if spread > _accu_stale_threshold:
            print(f"⚠️ NWS-AccuWeather spread={spread:.0f}°F > {_accu_stale_threshold:.0f}°F "
                  f"[{_accu_stale_label}] — "
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
        try:
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
        except Exception as _v1_e:
            print(f"⚠️ v1 regression predict failed (skipping to v4/v2): {_v1_e}")

    # --- 11. Try v8 > v7 > v6 > v5 > v4 > v2 ---
    # v10: v9 + MANH Manhattan Mesonet 5-min (151 features, needs ≥30 MANH rows)
    # v9: v8 + 10 Synoptic named-station features (149 features, needs ≥30 Synoptic rows)
    # v8: v7 + obs_heating_rate_delta stall signal (139 features)
    # v7: HRRR-anchored base (HRRR > NBM > AccuWeather > NWS), 138 features
    # v6: AccuWeather/NWS base + NBM, GEM HRDPS, HRRR 925mb, OKX radiosonde (138 features)
    # v5: AccuWeather/NWS base + high-timing features (122 features)
    # v4+: older architectures (backward compat)
    v11_regressor, v11_classifier, v11_feature_cols = _load_v11_models()
    v10_regressor, v10_classifier, v10_feature_cols = _load_v10_models()
    v9_regressor, v9_classifier, v9_feature_cols = _load_v9_models()
    v8_regressor, v8_classifier, v8_feature_cols = _load_v8_models()
    v7_regressor, v7_classifier, v7_feature_cols = _load_v7_models()
    v6_regressor, v6_classifier, v6_feature_cols = _load_v6_models()
    v5_regressor, v5_classifier, v5_feature_cols = _load_v5_models()
    v4_regressor, v4_bucket_info, v4_classifier = _load_v4_models()
    v2_temp_model, v2_bucket_info, v2_classifier, v2_atm_predictor = _load_v2_models()

    # Select best available model (prefer newest)
    use_v11 = v11_classifier is not None and v11_feature_cols is not None
    use_v10 = not use_v11 and v10_classifier is not None and v10_feature_cols is not None
    use_v9 = not use_v11 and not use_v10 and v9_classifier is not None and v9_feature_cols is not None
    use_v8 = not use_v11 and not use_v10 and not use_v9 and v8_classifier is not None and v8_feature_cols is not None
    use_v7 = not use_v11 and not use_v10 and not use_v9 and not use_v8 and v7_classifier is not None and v7_feature_cols is not None
    use_v6 = not use_v11 and not use_v10 and not use_v9 and not use_v8 and not use_v7 and v6_classifier is not None and v6_feature_cols is not None
    use_v5 = not use_v11 and not use_v10 and not use_v9 and not use_v8 and not use_v7 and not use_v6 and v5_classifier is not None and v5_feature_cols is not None
    use_v4 = not use_v11 and not use_v10 and not use_v9 and not use_v8 and not use_v7 and not use_v6 and not use_v5 and v4_classifier is not None

    if use_v11:
        active_classifier = v11_classifier
        active_regressor  = v11_regressor
        active_feature_cols = v11_feature_cols
        active_version = "v11_model_vs_nws_divergence"
    elif use_v10:
        active_classifier = v10_classifier
        active_regressor  = v10_regressor
        active_feature_cols = v10_feature_cols
        active_version = "v10_manh_mesonet"
    elif use_v9:
        active_classifier = v9_classifier
        active_regressor  = v9_regressor
        active_feature_cols = v9_feature_cols
        active_version = "v9_synoptic_stations"
    elif use_v8:
        active_classifier = v8_classifier
        active_regressor  = v8_regressor
        active_feature_cols = v8_feature_cols
        active_version = "v8_hrrr_stall_signal"
    elif use_v7:
        active_classifier = v7_classifier
        active_regressor  = v7_regressor
        active_feature_cols = v7_feature_cols
        active_version = "v7_hrrr_anchored"
    elif use_v6:
        active_classifier = v6_classifier
        active_regressor  = v6_regressor
        active_feature_cols = v6_feature_cols
        active_version = "v6_nbm_hrdps_hrrr925_radiosonde"
    elif use_v5:
        active_classifier = v5_classifier
        active_regressor  = v5_regressor
        active_feature_cols = v5_feature_cols
        active_version = "v5_high_timing"
    elif use_v4:
        active_classifier = v4_classifier
        active_regressor  = v4_regressor
        active_feature_cols = (
            list(v4_regressor.feature_names_in_)
            if (v4_regressor is not None and hasattr(v4_regressor, "feature_names_in_"))
            else FEATURE_COLS_V4
        )
        active_version = "v4_observation_features"
    else:
        active_classifier = v2_classifier
        active_regressor  = v2_temp_model
        active_feature_cols = FEATURE_COLS_V2
        active_version = "v2_atm_classifier"

    active_bucket_info = v4_bucket_info if (use_v10 or use_v9 or use_v8 or use_v7 or use_v6 or use_v5 or use_v4) else v2_bucket_info

    if active_classifier is not None:
        try:
            # Fetch atmospheric features — use prefetched data when available
            # (caller already fetched for intraday shift comparison, avoid double call)
            if prefetched_atm is not None and len(prefetched_atm) > 0:
                atm_features = prefetched_atm
                print(f"🌤️ Using prefetched atmospheric features ({len(atm_features)} keys)")
            else:
                atm_features = _fetch_atmospheric_features(target_date_iso)

                # ── Cache fallback: Fill missing values from cached snapshot ──
                # If Open-Meteo or Synoptic APIs are rate-limited/down, fill gaps with cache
                atm_features = fill_missing_from_cache(_CITY_KEY, atm_features)

            # Merge atmospheric features into the feature dict
            v2_features = dict(features)
            v2_features.update(atm_features)

            # Fetch MOS forecast
            mos_temp = _fetch_mos_forecast(target_date_iso)
            v2_features["mos_max_temp"] = float(mos_temp) if mos_temp is not None else np.nan

            # For D+1 predictions, compute today's NWS forecast so we can measure
            # whether NWS is currently biased warm/cold in the actual airmass today.
            # This is used as a pattern-stability-gated signal in _fetch_observation_features.
            _today_nws_last = None
            _today_iso = today_nyc().isoformat()
            if target_date_iso != _today_iso:
                _today_nws_fc = []
                for _r in rows:
                    if _r.get("forecast_or_actual") != "forecast":
                        continue
                    if _r.get("target_date") != _today_iso:
                        continue
                    if (_r.get("source") or "").lower() == "accuweather":
                        continue
                    _ph = _float_or_none(_r.get("predicted_high"))
                    if _ph is None:
                        continue
                    _ts = _r.get("timestamp") or _r.get("forecast_time") or ""
                    _today_nws_fc.append((_ts, _ph))
                if _today_nws_fc:
                    _today_nws_last = sorted(_today_nws_fc)[-1][1]

            # Fetch observation features (real-time NWS station data)
            obs_features = _fetch_observation_features(
                target_date_iso,
                nws_last=features.get("nws_last"),
                atm_features=atm_features,
                today_nws_last=_today_nws_last,
            )
            v2_features.update(obs_features)
            obs_populated = sum(1 for v in obs_features.values()
                                if v is not None and not (isinstance(v, float) and np.isnan(v)))
            print(f"🔭 Observation features: {obs_populated}/{len(obs_features)} populated")

            # Fetch Weather Underground PWS features (hyper-local Central Park area stations)
            # Requires WU_API_KEY secret. Auto-discovers nearby stations or uses WU_STATION_IDS.
            try:
                from wunderground_client import get_wu_obs_features
                import nws_auto_logger as _nal_wu
                _wu_cfg = _nal_wu._CITY_CFG
                ambient_feats = get_wu_obs_features(
                    lat=_wu_cfg.get("open_meteo_lat", 40.7834),
                    lon=_wu_cfg.get("open_meteo_lon", -73.965),
                    nws_last=features.get("nws_last"),
                )
                v2_features.update(ambient_feats)
                amb_populated = sum(1 for v in ambient_feats.values()
                                    if v is not None and not (isinstance(v, float) and np.isnan(v)))
                if amb_populated > 0:
                    print(f"🌤️ WU PWS features: {amb_populated}/{len(ambient_feats)} populated")
            except Exception as _amb_e:
                print(f"  ⚠️ WU PWS features skipped: {_amb_e}")
                for col in ["obs_ambient_temp", "obs_ambient_vs_nws",
                            "obs_ambient_spread", "obs_ambient_count"]:
                    v2_features.setdefault(col, np.nan)

            # NWS overnight jump: nws_first_d0 - nws_d1_final
            # Captures when NWS made a suspicious overnight revision (like today:
            # D-1 final=63°F, D0 3am=66°F → 3°F overnight warm jump → actual=63°F).
            try:
                nws_d1_final = _get_nws_d1_final(target_date_iso)
                nws_first_d0 = features.get("nws_first")
                v2_features["nws_d1_final"] = nws_d1_final if nws_d1_final is not None else np.nan
                if nws_d1_final is not None and nws_first_d0 is not None:
                    jump = round(float(nws_first_d0) - float(nws_d1_final), 1)
                    v2_features["nws_overnight_jump"] = jump
                    if abs(jump) >= 1.5:
                        direction = "↑" if jump > 0 else "↓"
                        print(f"  🌙 NWS overnight jump: D-1 final={nws_d1_final}°F → "
                              f"D0 first={nws_first_d0}°F ({direction}{abs(jump):.1f}°F)")
                else:
                    v2_features["nws_overnight_jump"] = np.nan
            except Exception as _seq_e:
                print(f"  ⚠️ NWS sequence features skipped: {_seq_e}")
                v2_features.setdefault("nws_d1_final", np.nan)
                v2_features.setdefault("nws_overnight_jump", np.nan)

            # ── Synoptic Data (MesoWest) — 100+ stations within 5mi of Central Park/KCQT ──
            try:
                from synoptic_client import get_synoptic_obs_features
                import nws_auto_logger as _nal_syn
                _syn_cfg = _nal_syn._CITY_CFG
                syn_feats = get_synoptic_obs_features(
                    lat=_syn_cfg.get("synoptic_lat", _syn_cfg.get("open_meteo_lat", 40.7834)),
                    lon=_syn_cfg.get("synoptic_lon", _syn_cfg.get("open_meteo_lon", -73.965)),
                    nws_last=features.get("nws_last"),
                    radius_miles=_syn_cfg.get("synoptic_radius_miles", 5.0),
                    city=_CITY_KEY,
                )
                v2_features.update(syn_feats)
                syn_pop = sum(1 for v in syn_feats.values()
                              if v is not None and not (isinstance(v, float) and np.isnan(v)))
                if syn_pop > 0:
                    print(f"📡 Synoptic features: {syn_pop}/{len(syn_feats)} populated")
                # Write-back to prefetched_atm so the caller can store them in atm_snapshot.
                # Include ALL obs_* keys from syn_feats — aggregate AND named station features.
                if prefetched_atm is not None and syn_pop > 0:
                    for _sk, _sv in syn_feats.items():
                        if _sk.startswith("obs_"):
                            prefetched_atm[_sk] = _sv
            except Exception as _syn_e:
                print(f"  ⚠️ Synoptic features skipped: {_syn_e}")
                for col in SYNOPTIC_OBS_COLS:
                    v2_features.setdefault(col, np.nan)

            # ── NY State Mesonet — borough stations, no API key needed ──
            try:
                from nysmesonet_client import get_nysm_obs_features
                nysm_feats = get_nysm_obs_features(nws_last=features.get("nws_last"))
                v2_features.update(nysm_feats)
                nysm_pop = sum(1 for v in nysm_feats.values()
                               if v is not None and not (isinstance(v, float) and np.isnan(v)))
                if nysm_pop > 0:
                    print(f"🏙️ NYSM features: {nysm_pop}/{len(nysm_feats)} populated")
                # Write-back to prefetched_atm so the caller can store them in atm_snapshot
                if prefetched_atm is not None and nysm_pop > 0:
                    for _nk, _nv in nysm_feats.items():
                        if _nk.startswith("obs_nysm_"):
                            prefetched_atm[_nk] = _nv
            except Exception as _nysm_e:
                print(f"  ⚠️ NYSM features skipped: {_nysm_e}")
                for col in NYSM_OBS_COLS:
                    v2_features.setdefault(col, np.nan)

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

            # v11: derive model-vs-NWS divergence features BEFORE building X_v2,
            # so they are populated in the DataFrame (not left as NaN fill-ins).
            # mm_hrrr_vs_nws = HRRR - nws_last: how far the #1-accuracy fast model
            # sits above/below the (often stale) NWS point forecast.
            # HistGradientBoosting handles NaN natively so partial availability
            # (e.g. HRRR available, NBM not) is fine.
            _nws_ref = v2_features.get("nws_last")
            if _nws_ref is not None and not (isinstance(_nws_ref, float) and math.isnan(_nws_ref)):
                _mm_hrrr = v2_features.get("mm_hrrr_max")
                _mm_nbm  = v2_features.get("mm_nbm_max")
                _mm_mean = v2_features.get("mm_mean")
                if _mm_hrrr is not None and not (isinstance(_mm_hrrr, float) and math.isnan(_mm_hrrr)):
                    v2_features["mm_hrrr_vs_nws"] = round(float(_mm_hrrr) - float(_nws_ref), 1)
                if _mm_nbm is not None and not (isinstance(_mm_nbm, float) and math.isnan(_mm_nbm)):
                    v2_features["mm_nbm_vs_nws"]  = round(float(_mm_nbm)  - float(_nws_ref), 1)
                if _mm_mean is not None and not (isinstance(_mm_mean, float) and math.isnan(_mm_mean)):
                    v2_features["mm_mean_vs_nws"] = round(float(_mm_mean) - float(_nws_ref), 1)

            # v13: derive BL safeguard features BEFORE building X_v2.
            # These guard against BL-triggered reductions when atmospheric state doesn't align.
            # entrainment_temp_diff = 925mb - obs_latest_temp (negative = cool aloft)
            # marine_containment = JFK_vs_KNYC / BL_max (ratio of gradient to depth)
            # inland_strength = mean(TEB,CDW,SMQ) - mm_mean (inland actual vs forecast)
            _atm_925mb = v2_features.get("atm_925mb_temp_mean")
            _obs_latest = v2_features.get("obs_latest_temp")
            if _atm_925mb is not None and _obs_latest is not None:
                if not (isinstance(_atm_925mb, float) and math.isnan(_atm_925mb)) and \
                   not (isinstance(_obs_latest, float) and math.isnan(_obs_latest)):
                    v2_features["entrainment_temp_diff"] = round(float(_atm_925mb) - float(_obs_latest), 1)

            _jfk_knyc = v2_features.get("obs_kjfk_vs_knyc")
            _bl_max = v2_features.get("atm_bl_height_max")
            if _jfk_knyc is not None and _bl_max is not None:
                if not (isinstance(_jfk_knyc, float) and math.isnan(_jfk_knyc)) and \
                   not (isinstance(_bl_max, float) and math.isnan(_bl_max)) and \
                   float(_bl_max) > 0:
                    v2_features["marine_containment"] = round(float(_jfk_knyc) / float(_bl_max), 6)

            _mm_mean = v2_features.get("mm_mean")
            if _mm_mean is not None and not (isinstance(_mm_mean, float) and math.isnan(_mm_mean)):
                inland_temps = []
                for col in ["obs_kteb_temp", "obs_kcdw_temp", "obs_ksmq_temp"]:
                    val = v2_features.get(col)
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        inland_temps.append(float(val))
                if inland_temps:
                    inland_mean = sum(inland_temps) / len(inland_temps)
                    v2_features["inland_strength"] = round(inland_mean - float(_mm_mean), 1)

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

            # ── Center temperature: v7 priority cascade ───────────────────────
            #
            # v7 architecture (HRRR-anchored):
            #   1. HRRR_max + v7_bias   HRRR is #1 accuracy model (wethr.net).
            #                           v7 regressor trained on (actual - HRRR), so
            #                           bias is the systematic HRRR correction.
            #   2. NBM_max + v7_bias    #2-3 accuracy, blends 50+ models.
            #   3. atm_predicted_high   Physics model — BL height, solar, 925mb.
            #                           Fallback when HRRR/NBM unavailable.
            #   4. accu/nws + v7_bias   Historical compatibility base.
            #   5. Forecast average     No models at all.
            #
            # Pre-v7 (v6 and below): atm_predicted_high was the primary anchor.
            # v7 promotes HRRR/NBM to tier-1 because they are highest-accuracy
            # SHORT-RANGE models and explicitly correct for GFS cap errors.
            # On a cap day: HRRR shows 54°F, atm_predicted_high (using GFS 925mb)
            # shows 58°F — v7 anchors at 54, avoids the NWS/AccuWeather cap miss.

            atm_pred_val = v2_features.get("atm_predicted_high")
            has_atm = (atm_pred_val is not None
                       and not (isinstance(atm_pred_val, float) and math.isnan(atm_pred_val))
                       and 0.0 < float(atm_pred_val) < 130.0)

            def _valid_temp(v):
                """Return float if v is a finite temperature in [0, 130], else None."""
                if v is None:
                    return None
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return None
                return fv if (math.isfinite(fv) and 0.0 < fv < 130.0) else None

            _hrrr_base = _valid_temp(v2_features.get("mm_hrrr_max"))
            _nbm_base  = _valid_temp(v2_features.get("mm_nbm_max"))

            _use_hrrr_anchor = (use_v8 or use_v7) and active_regressor is not None
            if _use_hrrr_anchor and _hrrr_base is not None:
                # TIER 1 (v7/v8): HRRR anchor + regressor bias correction
                # v7/v8 regressor trained on y_bias = actual - HRRR_max.
                _bias = float(active_regressor.predict(X_v2)[0])
                v2_temp = _hrrr_base + _bias
                _atm_note = (f" | atm={float(atm_pred_val):.1f}°F (delta={v2_temp - float(atm_pred_val):+.1f}°F)"
                             if has_atm else "")
                _stall = v2_features.get("obs_heating_rate_delta")
                _stall_note = (f" | stall={_stall:+.1f}°F/hr" if _stall is not None
                               and not (isinstance(_stall, float) and math.isnan(_stall)) else "")
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"({active_version}: HRRR={_hrrr_base:.0f} + bias={_bias:+.1f})"
                      f"{_stall_note}{_atm_note}")

            elif _use_hrrr_anchor and _nbm_base is not None:
                # TIER 2 (v7/v8): NBM anchor + regressor bias correction
                _bias = float(active_regressor.predict(X_v2)[0])
                v2_temp = _nbm_base + _bias
                _atm_note = (f" | atm={float(atm_pred_val):.1f}°F (delta={v2_temp - float(atm_pred_val):+.1f}°F)"
                             if has_atm else "")
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"({active_version}: NBM={_nbm_base:.0f} + bias={_bias:+.1f}){_atm_note}")

            elif has_atm:
                # TIER 3: atmospheric predictor (physics model, agency-independent)
                # Used for v6/v5/v4 when v7 is not active, or v7 when HRRR/NBM unavailable.
                v2_temp = float(atm_pred_val)
                _regression_note = ""
                if active_regressor is not None:
                    # Run regressor for monitoring/logging only — do NOT shift center.
                    try:
                        _v2_bias_ref = float(active_regressor.predict(X_v2)[0])
                        _v2_temp_ref = base + _v2_bias_ref
                        _regression_note = (f" | regressor would give {_v2_temp_ref:.1f}°F "
                                            f"(base={base:.0f} + bias={_v2_bias_ref:+.1f}) — "
                                            f"delta={v2_temp - _v2_temp_ref:+.1f}°F from atm")
                    except Exception:
                        pass
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"(atm_predicted_high — physics fallback [{active_version}]){_regression_note}")

            elif active_regressor is not None:
                # TIER 4: AccuWeather/NWS base + regressor bias (historical compat)
                v2_bias = float(active_regressor.predict(X_v2)[0])
                v2_temp = base + v2_bias
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"({active_version}: base={base:.0f} + bias={v2_bias:+.1f}) "
                      f"[HRRR/NBM/atm unavailable]")

            else:
                # TIER 5: no models — use forecast average
                all_forecasts = [features["nws_last"]]
                if has_accu and not np.isnan(features["accu_last"]):
                    all_forecasts.append(features["accu_last"])
                v2_temp = float(np.mean(all_forecasts))
                accu_note = f", AccuWx={features['accu_last']:.0f}" if has_accu else ""
                print(f"   Center temp: {v2_temp:.1f}°F "
                      f"(forecast avg fallback: NWS={features['nws_last']:.0f}{accu_note}) "
                      f"[no models available]")

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

                # ── v3 shadow: log v3's bucket pick alongside v4/v2 for backtesting ──
                # v3 is a unified regression on all multi-year days — never used at
                # inference, but we want to compare its bucket to v4's on live Kalshi
                # days before deciding whether to promote it.
                try:
                    v3_model, v3_atm = _load_v3_model()
                    if v3_model is not None:
                        from model_config import FEATURE_COLS_V3, temp_to_bucket_label as _t2b
                        v3_feat_vec = [features.get(c, np.nan) for c in FEATURE_COLS_V3]
                        v3_pred = float(v3_model.predict([v3_feat_vec])[0])
                        v3_bucket = _t2b(v3_pred)
                        # Store in result for callers that want it; NOT written to Supabase
                        # payload (no schema column yet). Shows in GH Actions logs only.
                        result["v3_shadow_f"] = round(v3_pred, 1)
                        result["v3_shadow_bucket"] = v3_bucket
                        agreement = "✅" if v3_bucket == v2_best["bucket"] else "⚡"
                        print(f"   {agreement} v3 shadow: {v3_pred:.1f}°F → {v3_bucket} "
                              f"(v4/v2: {v2_best['bucket']})")
                except Exception as _v3e:
                    pass  # shadow — never block production path

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
    Fetch recent NWS observations for primary + regional stations and upsert to Supabase.
    For NYC: fetches KNYC (primary) + KJFK + KLGA (regional).
    Returns total rows upserted across all stations.
    """
    import nws_auto_logger as _nal
    cfg = _nal._CITY_CFG
    primary_station = cfg.get("obs_station", "KNYC")
    tz_name = cfg.get("timezone", "America/New_York")
    city = city_key or _CITY_KEY
    all_stations = [primary_station] + cfg.get("regional_obs_stations", [])
    total = 0
    for stn in all_stations:
        total += _collect_obs_single_station(stn, city, tz_name)
    return total


def _collect_obs_single_station(station: str, city: str, tz_name: str) -> int:
    """Fetch and upsert NWS observations for a single station."""
    # Fetch last 24 observations
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
    today_nws_last: float = None,
) -> dict:
    """
    Compute observation-derived features from NWS observations stored in Supabase.

    Args:
        target_date_iso: The date we're predicting for (YYYY-MM-DD)
        nws_last: Latest NWS forecast value (°F) for target date (used for D0 obs signal)
        atm_features: Dict of atmospheric features (for obs_vs_intra_forecast)
        today_nws_last: Latest NWS forecast for TODAY (used for D1 pattern-stability signal).
            When predicting tomorrow, we compare today's running max vs today's NWS forecast
            (not tomorrow's) to measure whether NWS is currently biased warm/cold.
            Gated on pattern stability: if a big temp swing is expected overnight (front
            passage), the signal is noise and is set to NaN so 925mb/solar/HRRR carry it.

    Returns dict with all OBSERVATION_COLS. NaN for any unavailable feature.
    """
    nan_result = {col: np.nan for col in OBSERVATION_COLS}

    # Only fetch observations for today (no observations exist for future dates)
    today = today_nyc().isoformat()
    if target_date_iso != today:
        # For D1 (tomorrow) predictions: measure today's NWS bias and gate on pattern stability.
        #
        # Signal: today_max - today_nws_last (how wrong is NWS about TODAY's airmass?)
        # Gate: if today_max vs tomorrow_nws swing is large (>7°F), a front is passing
        #       overnight — today's NWS bias tells us nothing about tomorrow's pattern.
        #       Leave NaN and let 925mb/solar/HRRR features carry the D+1 prediction.
        if today_nws_last is not None:
            today_obs = _query_supabase_observations(today)
            today_max = max(
                (r["temp_f"] for r in today_obs if r.get("temp_f") is not None),
                default=None,
            ) if today_obs else None
            # Cross-check with the NWS API live reading (_detect_high_locked uses
            # api.weather.gov directly, which has the latest obs including readings
            # that may not yet be in Supabase due to ingestion lag).  Use the higher
            # of the two sources so an overnight spike never depresses the bias signal
            # when the afternoon high has already been observed.
            try:
                _dlock_today = _detect_high_locked(today)
                _api_max = _dlock_today.get("obs_high_f")
                if _api_max is not None:
                    today_max = max(today_max, _api_max) if today_max is not None else _api_max
            except Exception:
                pass
            if today_max is not None:
                today_nws_bias = today_max - today_nws_last  # how biased is NWS today?
                # Pattern stability check: how different is today's temp from tomorrow's forecast?
                pattern_swing = abs(today_max - nws_last) if nws_last is not None else 999.0
                if pattern_swing < 7.0:
                    # Stable pattern — NWS bias in today's airmass likely carries forward
                    nan_result["obs_temp_vs_forecast_max"] = round(today_nws_bias, 1)
                    print(f"  🌡️ D+1 NWS bias signal: today_max={today_max:.1f}°F, "
                          f"today_nws={today_nws_last:.1f}°F → bias={today_nws_bias:+.1f}°F "
                          f"(pattern stable, swing={pattern_swing:.1f}°F)")
                else:
                    # Front passage / big pattern change — signal is noise, leave NaN
                    print(f"  ⚡ D+1 pattern change detected (swing={pattern_swing:.1f}°F ≥ 7°F) "
                          f"— suppressing NWS bias signal, 925mb/solar/HRRR carry the prediction")
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

    # 6-hour max: max of last 6 calendar-day observations (temp_f only).
    # We intentionally ignore the METAR six_hr_max_f column here because it is a
    # rolling 6-hour window recorded by the weather station and can cross midnight,
    # pulling yesterday's warm readings into today's feature at 1–3 AM and causing
    # false "LIVE DATA EXCEEDS PREDICTION" alerts.  valid_obs is already filtered to
    # the current calendar day by _query_supabase_observations, so using temp_f is safe.
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

    # Heating rate DELTA (stall signal) — was warming, now plateaued?
    #
    # Compares the heating rate over the last 3 hours vs the preceding 3 hours.
    # This is exactly what a human does: look at the last 3 hours of the chart
    # and compare to the 3 hours before that. If recent rate << earlier rate,
    # the temperature has stalled — a cap is holding.
    #
    # Why not split all obs in half? Because the "early half" includes overnight
    # cooling, making the split contaminated. Fixed 3-hour windows isolate the
    # comparison to the afternoon prediction window.
    #
    # Example (April 12 cap day, model runs at 1 PM):
    #   Early window  (7 AM–10 AM): +1.7°F/hr  (normal morning ramp)
    #   Recent window (10 AM–1 PM): ≈ 0°F/hr   (cap holding at 52°F)
    #   delta = 0.0 – 1.7 = –1.7°F/hr  ← strong cap-hold fingerprint
    #
    # A random human with a browser tab spotted this on April 12 at 11 AM–1 PM.
    # This feature gives the model the same early-cycle signal.
    _RECENT_WINDOW_HRS = 3.0  # compare this many trailing hours
    _EARLY_WINDOW_HRS  = 3.0  # against the preceding window of same length
    features["obs_heating_rate_delta"] = np.nan  # default
    if len(valid_obs) >= 4:
        try:
            now_obs_dt = datetime.fromisoformat(
                valid_obs[-1]["observed_at"].replace("Z", "+00:00"))

            def _obs_dt(o):
                return datetime.fromisoformat(o["observed_at"].replace("Z", "+00:00"))

            recent_cutoff = now_obs_dt - timedelta(hours=_RECENT_WINDOW_HRS)
            early_cutoff  = now_obs_dt - timedelta(hours=_RECENT_WINDOW_HRS + _EARLY_WINDOW_HRS)

            recent_obs = [o for o in valid_obs if _obs_dt(o) >= recent_cutoff]
            early_obs  = [o for o in valid_obs
                          if early_cutoff <= _obs_dt(o) < recent_cutoff]

            def _slope(obs_slice):
                if len(obs_slice) < 2:
                    return None
                t0 = _obs_dt(obs_slice[0])
                t1 = _obs_dt(obs_slice[-1])
                hours_span = (t1 - t0).total_seconds() / 3600.0
                if hours_span < 0.25:  # need at least 15 minutes of span
                    return None
                return (obs_slice[-1]["temp_f"] - obs_slice[0]["temp_f"]) / hours_span

            early_slope  = _slope(early_obs)
            recent_slope = _slope(recent_obs)

            if early_slope is not None and recent_slope is not None:
                delta = recent_slope - early_slope
                features["obs_heating_rate_delta"] = round(delta, 2)
                if delta < -1.5:
                    print(f"  🧊🧊 STRONG stall: early={early_slope:+.1f}°F/hr → "
                          f"recent={recent_slope:+.1f}°F/hr (Δ={delta:+.1f}) — cap confirmed")
                elif delta < -0.8:
                    print(f"  🧊 Cap stall: early={early_slope:+.1f}°F/hr → "
                          f"recent={recent_slope:+.1f}°F/hr (Δ={delta:+.1f}) — cap likely holding")
        except Exception:
            pass

    # Obs max vs NWS forecast
    if nws_last is not None and features["obs_max_so_far"] is not None:
        features["obs_temp_vs_forecast_max"] = round(
            features["obs_max_so_far"] - nws_last, 1
        )
    else:
        features["obs_temp_vs_forecast_max"] = np.nan

    # ── High-timing features (HIGH_TIMING_COLS) ──────────────────────────
    # Populated from the same observation set — no extra API call.
    # Find the hour when obs_max_so_far was recorded and detect overnight pattern.
    try:
        max_val = features.get("obs_max_so_far")
        if max_val is not None and not (isinstance(max_val, float) and np.isnan(max_val)):
            # Find which observation had the max temp
            peak_hour = None
            for r in reversed(valid_obs):
                if abs(r["temp_f"] - max_val) < 0.15:
                    try:
                        dt = datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00"))
                        peak_hour = dt.astimezone(tz).hour
                    except Exception:
                        pass
                    break
            features["obs_high_peak_hour"] = float(peak_hour) if peak_hour is not None else np.nan
            # Only flag as overnight high if the NWS daytime forecast is NOT
            # substantially above the observed overnight peak. If NWS still
            # forecasts 3+ °F above the overnight max, the day's actual high
            # hasn't been set yet — don't lock the prediction prematurely.
            if peak_hour is not None and peak_hour < 9:
                nws_last_val = features.get("nws_last")
                nws_ok = (nws_last_val is not None
                          and not (isinstance(nws_last_val, float) and np.isnan(nws_last_val)))
                if nws_ok and float(nws_last_val) >= float(max_val) + 3.0:
                    # NWS still expects a significantly higher daytime high — suppress
                    features["obs_is_overnight_high"] = 0.0
                    print(f"  ℹ️  Overnight peak {max_val:.1f}°F @{peak_hour}h suppressed "
                          f"(NWS still forecasts {nws_last_val:.0f}°F → daytime high not yet set)")
                else:
                    features["obs_is_overnight_high"] = 1.0
            else:
                features["obs_is_overnight_high"] = 0.0
            # Consecutive falling hours from the peak
            falling = 0
            prev_t  = valid_obs[-1]["temp_f"] if valid_obs else None
            for r in reversed(valid_obs[:-1]):
                if prev_t is None or r["temp_f"] > prev_t:
                    break
                falling += 1
                prev_t = r["temp_f"]
            features["obs_temp_falling_hrs"] = float(falling)
        else:
            features["obs_high_peak_hour"]    = np.nan
            features["obs_is_overnight_high"] = np.nan
            features["obs_temp_falling_hrs"]  = np.nan
    except Exception as _ht_e:
        features["obs_high_peak_hour"]    = np.nan
        features["obs_is_overnight_high"] = np.nan
        features["obs_temp_falling_hrs"]  = np.nan

    # Fill any missing OBSERVATION_COLS keys with NaN
    for col in OBSERVATION_COLS:
        if col not in features:
            features[col] = np.nan

    # ── Regional multi-station features (JFK + LGA) ──────────────────────
    try:
        import nws_auto_logger as _nal
        cfg_r = _nal._CITY_CFG
        regional_stations = cfg_r.get("regional_obs_stations", [])
        station_col_map = {"KJFK": "obs_jfk_temp", "KLGA": "obs_lga_temp"}

        all_temps = []
        primary_max = features.get("obs_max_so_far")
        if primary_max is not None and not (isinstance(primary_max, float) and np.isnan(primary_max)):
            all_temps.append(primary_max)

        for stn in regional_stations:
            col_key = station_col_map.get(stn)
            stn_rows = _query_supabase_obs_by_station(target_date_iso, stn)
            valid = [r for r in stn_rows if r.get("temp_f") is not None]
            if valid and col_key:
                t = valid[-1]["temp_f"]
                features[col_key] = round(t, 1)
                all_temps.append(t)
                print(f"  🌡️ {stn}: {t:.1f}°F")
            elif col_key:
                features[col_key] = np.nan

        if len(all_temps) >= 2:
            features["obs_regional_spread"] = round(max(all_temps) - min(all_temps), 1)
            features["obs_regional_mean"] = round(sum(all_temps) / len(all_temps), 1)
            features["obs_regional_vs_nws"] = (
                round(features["obs_regional_mean"] - nws_last, 1) if nws_last is not None else np.nan
            )
            print(f"  🗺️ Regional: spread={features['obs_regional_spread']:.1f}°F  "
                  f"mean={features['obs_regional_mean']:.1f}°F  "
                  f"vs NWS={features['obs_regional_vs_nws']:+.1f}°F" if nws_last else
                  f"  🗺️ Regional: spread={features['obs_regional_spread']:.1f}°F  "
                  f"mean={features['obs_regional_mean']:.1f}°F")
        else:
            for col in ["obs_regional_spread", "obs_regional_mean", "obs_regional_vs_nws"]:
                features[col] = np.nan
    except Exception as _reg_e:
        print(f"  ⚠️ Regional obs failed: {_reg_e}")
        for col in ["obs_jfk_temp", "obs_lga_temp", "obs_regional_spread",
                    "obs_regional_mean", "obs_regional_vs_nws"]:
            features.setdefault(col, np.nan)

    return features


def _query_supabase_obs_by_station(target_date_iso: str, station: str, city: str = None) -> list[dict]:
    """Query Supabase nws_observations for a specific date + station."""
    sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not sb_url or not sb_key:
        return []
    import nws_auto_logger as _nal
    cfg = _nal._CITY_CFG
    tz_name = cfg.get("timezone", "America/New_York")
    city = city or _CITY_KEY
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    start_local = datetime.fromisoformat(f"{target_date_iso}T00:00:00").replace(tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end_local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    endpoint = f"{sb_url}/rest/v1/nws_observations"
    params = (f"?city=eq.{city}&station=eq.{station}"
              f"&observed_at=gte.{start_utc}&observed_at=lt.{end_utc}"
              f"&order=observed_at.asc&limit=50")
    req = urllib.request.Request(
        endpoint + params,
        headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ⚠️ Supabase obs query {station}: {e}")
        return []


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
        err_body = ""
        if hasattr(e, "read"):
            try:
                err_body = e.read().decode("utf-8", "ignore")
                print(f"❌ supabase {getattr(e,'code','?')}: {err_body}")
            except Exception:
                print(f"❌ supabase: {e}")
        else:
            print(f"❌ supabase: {e}")
        # Re-raise so callers know the upsert failed. write_both_snapshots wraps
        # each write in try/except and will log the failure without crashing.
        raise RuntimeError(f"supabase upsert failed: {err_body or e}") from e

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


def backfill_obs_historical(city_key: str = None) -> str:
    """
    Pull 4+ years of KJFK, KLGA, and KNYC hourly observations from the
    Iowa State Mesonet (IEM) ASOS archive — no API key required.

    Computes the same observation features as backfill_observation_features()
    but for every date from 2022-01-01 through today, giving the model
    ~1500 training rows of real obs features instead of ~14.

    Features produced match OBSERVATION_COLS + REGIONAL_OBS_COLS:
        obs_latest_temp, obs_latest_hour, obs_max_so_far, obs_6hr_max,
        obs_vs_intra_forecast, obs_wind_speed, obs_wind_gust,
        obs_wind_dir_sin, obs_wind_dir_cos, obs_cloud_cover,
        obs_heating_rate, obs_temp_vs_forecast_max,
        obs_jfk_temp, obs_lga_temp, obs_regional_spread,
        obs_regional_mean, obs_regional_vs_nws

    Saves (or merges with) {prefix}observation_data.csv.
    Returns path to saved CSV.
    """
    import io, csv as _csv
    import nws_auto_logger as _nal
    cfg     = _nal._CITY_CFG
    prefix  = cfg.get("model_prefix", "")
    city    = city_key or _CITY_KEY
    tz_name = cfg.get("timezone", "America/New_York")

    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)

    # ── IEM station mapping ────────────────────────────────────────────────
    # KNYC (Central Park) is NOT in the ASOS network — IEM uses "NYC" for it.
    # KJFK / KLGA are standard ICAO ASOS codes and work with IEM directly.
    IEM_STATIONS = {
        "NYC":  "nyc_cp",   # Central Park NYC → IEM station code "NYC"
        "KJFK": "jfk",
        "KLGA": "lga",
    }
    # NYC city configs only — extend here for LAX etc.
    if city != "nyc":
        print(f"⚠️  backfill_obs_historical only configured for nyc (got {city})")
        return ""

    # ── Date range ─────────────────────────────────────────────────────────
    from datetime import date as _date
    start_date = _date(2022, 1, 1)
    end_date   = _date.today()

    def _fetch_iem(station: str) -> list[dict]:
        """Fetch all hourly obs for one ASOS station from IEM."""
        url = (
            "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
            f"?station={station}"
            "&data=tmpf&data=sknt&data=gust&data=drct&data=skyc1"
            f"&year1={start_date.year}&month1={start_date.month}&day1={start_date.day}"
            f"&year2={end_date.year}&month2={end_date.month}&day2={end_date.day}"
            "&tz=America%2FNew_York"
            "&format=comma&latlon=no&direct=no"
        )
        print(f"  Fetching {station} from IEM ({start_date} → {end_date})…")
        req = urllib.request.Request(url, headers={"User-Agent": "nws-forecast-logger/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  ⚠️  Failed to fetch {station}: {exc}")
            return []

        # IEM prepends comment lines starting with '#' — strip them before parsing CSV
        clean_lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
        clean_csv   = "\n".join(clean_lines)

        if not clean_csv.strip():
            print(f"  ⚠️  Empty response for {station}")
            return []

        rows = []
        reader = _csv.DictReader(io.StringIO(clean_csv))
        for row in reader:
            try:
                tmpf = row.get("tmpf", "").strip()
                if not tmpf or tmpf in ("M", ""):
                    continue
                t_f = float(tmpf)
                valid_str = row.get("valid", "").strip()   # "2022-01-01 01:00"
                if not valid_str or valid_str == "valid":
                    continue
                # Parse as local time (IEM already converted via tz=America/New_York)
                obs_local = datetime.strptime(valid_str, "%Y-%m-%d %H:%M")
                obs_local = obs_local.replace(tzinfo=tz)

                # Wind (knots → mph)
                sknt = row.get("sknt", "").strip()
                gust = row.get("gust", "").strip()
                drct = row.get("drct", "").strip()
                skyc = row.get("skyc1", "").strip()

                rows.append({
                    "station":    station,
                    "obs_local":  obs_local,
                    "date_str":   obs_local.strftime("%Y-%m-%d"),
                    "hour":       obs_local.hour,
                    "temp_f":     t_f,
                    "wind_kts":   float(sknt) if sknt and sknt not in ("M", "") else None,
                    "gust_kts":   float(gust) if gust and gust not in ("M", "") else None,
                    "wind_dir":   float(drct) if drct and drct not in ("M", "") else None,
                    "sky":        skyc,
                })
            except Exception:
                continue
        print(f"    → {len(rows)} valid hourly rows for {station}")
        return rows

    # ── Fetch all three stations ───────────────────────────────────────────
    all_station_data: dict[str, list[dict]] = {}
    for station in IEM_STATIONS:
        all_station_data[station] = _fetch_iem(station)

    if not any(all_station_data.values()):
        print("⚠️  No IEM data fetched — check internet connection")
        return ""

    # Group each station's rows by date
    def _by_date(rows: list[dict]) -> dict[str, list[dict]]:
        d: dict[str, list[dict]] = {}
        for r in rows:
            d.setdefault(r["date_str"], []).append(r)
        return d

    by_date: dict[str, dict[str, list[dict]]] = {
        st: _by_date(rows) for st, rows in all_station_data.items()
    }

    # ── Sky condition → cloud cover ────────────────────────────────────────
    _SKY_MAP = {"CLR": 0.0, "SKC": 0.0, "FEW": 0.2, "SCT": 0.5, "BKN": 0.75, "OVC": 1.0}
    def _sky2cov(sky: str) -> float:
        return _SKY_MAP.get(sky[:3].upper(), float("nan")) if sky else float("nan")

    # ── Load NWS CSV for obs_temp_vs_forecast_max / obs_regional_vs_nws ───
    nws_csv  = f"{prefix}nws_forecast_log.csv"
    atm_csv  = f"{prefix}atmospheric_data.csv"
    nws_last_by_date: dict[str, float] = {}
    if os.path.exists(nws_csv):
        _ndf = pd.read_csv(nws_csv)
        for _, row in _ndf.iterrows():
            if row.get("forecast_or_actual") == "forecast" and row.get("target_date"):
                try:
                    nws_last_by_date[str(row["target_date"])] = float(row["predicted_high"])
                except (ValueError, TypeError):
                    pass

    atm_df = pd.read_csv(atm_csv) if os.path.exists(atm_csv) else None
    if atm_df is not None:
        atm_df["target_date"] = atm_df["target_date"].astype(str)

    # ── Compute features for every date that has KJFK or KLGA data ────────
    all_dates = sorted(
        set(by_date.get("KJFK", {})) | set(by_date.get("KLGA", {})) | set(by_date.get("NYC", {}))
    )
    print(f"📅 Computing obs features for {len(all_dates)} dates…")

    NOON_CUTOFF = 13   # use obs up to and including 1pm local
    INTRA_MAP   = {9: "intra_temp_9am", 10: "intra_temp_9am",
                   11: "intra_temp_noon", 12: "intra_temp_noon",
                   13: "intra_temp_3pm",  14: "intra_temp_3pm", 15: "intra_temp_3pm",
                   16: "intra_temp_5pm",  17: "intra_temp_5pm"}

    feature_rows = []
    for date_str in all_dates:
        def _snap(station: str) -> list[dict]:
            """Rows for this station up to NOON_CUTOFF."""
            return sorted(
                [r for r in by_date.get(station, {}).get(date_str, [])
                 if r["hour"] <= NOON_CUTOFF],
                key=lambda r: r["obs_local"],
            )

        nyc_rows = _snap("NYC")
        jfk_rows = _snap("KJFK")
        lga_rows = _snap("KLGA")

        # Need at least one of JFK or LGA
        if not jfk_rows and not lga_rows:
            continue

        # Use KNYC as primary; fall back to mean(JFK, LGA)
        primary_rows = nyc_rows if nyc_rows else (jfk_rows or lga_rows)
        latest       = primary_rows[-1]

        row: dict = {"target_date": date_str, "city": city}

        # ── Basic KNYC-equivalent obs ──────────────────────────────────
        row["obs_latest_temp"] = latest["temp_f"]
        row["obs_latest_hour"] = float(latest["hour"])
        row["obs_max_so_far"]  = max(r["temp_f"] for r in primary_rows)

        # 6hr max: last 6 hourly readings
        last6 = primary_rows[-6:] if len(primary_rows) >= 6 else primary_rows
        row["obs_6hr_max"] = max(r["temp_f"] for r in last6)

        # Wind from latest primary
        kts = latest["wind_kts"]
        gst = latest["gust_kts"]
        wdr = latest["wind_dir"]
        row["obs_wind_speed"] = round(kts * 1.15078, 1) if kts is not None else float("nan")
        row["obs_wind_gust"]  = round(gst * 1.15078, 1) if gst is not None else float("nan")
        if wdr is not None:
            row["obs_wind_dir_sin"] = round(math.sin(math.radians(wdr)), 4)
            row["obs_wind_dir_cos"] = round(math.cos(math.radians(wdr)), 4)
        else:
            row["obs_wind_dir_sin"] = float("nan")
            row["obs_wind_dir_cos"] = float("nan")
        row["obs_cloud_cover"] = _sky2cov(latest["sky"])

        # Heating rate (°F/hr from first to last reading)
        if len(primary_rows) >= 2:
            hrs = (primary_rows[-1]["obs_local"] - primary_rows[0]["obs_local"]).total_seconds() / 3600.0
            row["obs_heating_rate"] = round(
                (primary_rows[-1]["temp_f"] - primary_rows[0]["temp_f"]) / hrs, 2
            ) if hrs > 0 else float("nan")
        else:
            row["obs_heating_rate"] = float("nan")

        # obs_vs_intra_forecast
        row["obs_vs_intra_forecast"] = float("nan")
        if atm_df is not None:
            atm_row = atm_df[atm_df["target_date"] == date_str]
            if not atm_row.empty:
                ikey = INTRA_MAP.get(int(row["obs_latest_hour"]))
                if ikey and ikey in atm_row.columns:
                    ival = atm_row.iloc[0][ikey]
                    if pd.notna(ival):
                        row["obs_vs_intra_forecast"] = round(row["obs_latest_temp"] - float(ival), 1)

        # obs_temp_vs_forecast_max
        nws_last = nws_last_by_date.get(date_str)
        row["obs_temp_vs_forecast_max"] = (
            round(row["obs_max_so_far"] - nws_last, 1) if nws_last is not None else float("nan")
        )

        # ── Regional JFK / LGA features ───────────────────────────────
        jfk_t = jfk_rows[-1]["temp_f"] if jfk_rows else None
        lga_t = lga_rows[-1]["temp_f"] if lga_rows else None
        row["obs_jfk_temp"] = jfk_t if jfk_t is not None else float("nan")
        row["obs_lga_temp"] = lga_t if lga_t is not None else float("nan")

        avail = [v for v in [row["obs_latest_temp"], jfk_t, lga_t] if v is not None]
        if len(avail) >= 2:
            row["obs_regional_spread"] = round(max(avail) - min(avail), 1)
            row["obs_regional_mean"]   = round(sum(avail) / len(avail), 1)
        else:
            row["obs_regional_spread"] = float("nan")
            row["obs_regional_mean"]   = avail[0] if avail else float("nan")

        row["obs_regional_vs_nws"] = (
            round(row["obs_regional_mean"] - nws_last, 1)
            if nws_last is not None and pd.notna(row["obs_regional_mean"])
            else float("nan")
        )

        feature_rows.append(row)

    if not feature_rows:
        print("⚠️  No feature rows computed — check IEM data")
        return ""

    new_df = pd.DataFrame(feature_rows)

    # ── Merge with existing observation_data.csv ───────────────────────────
    csv_path = f"{prefix}observation_data.csv"
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing["target_date"] = existing["target_date"].astype(str)
        new_df["target_date"]   = new_df["target_date"].astype(str)
        # IEM historical rows take precedence for any date both cover
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="target_date", keep="last")
        combined = combined.sort_values("target_date")
        print(f"  Merged {len(existing)} existing rows + {len(new_df)} IEM rows "
              f"→ {len(combined)} total")
        combined.to_csv(csv_path, index=False)
    else:
        new_df = new_df.sort_values("target_date")
        new_df.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(new_df)} IEM observation rows to {csv_path}")

    # Summary
    valid_jfk = new_df["obs_jfk_temp"].notna().sum()
    valid_lga = new_df["obs_lga_temp"].notna().sum()
    print(f"   JFK data: {valid_jfk}/{len(new_df)} dates  |  LGA: {valid_lga}/{len(new_df)} dates")
    non_zero_vs = new_df["obs_vs_intra_forecast"].dropna()
    non_zero_vs = non_zero_vs[non_zero_vs != 0]
    if len(non_zero_vs):
        print(f"   obs_vs_intra_forecast: {len(non_zero_vs)} non-zero "
              f"(mean={non_zero_vs.mean():.1f}°F, std={non_zero_vs.std():.1f}°F)")

    return csv_path


def backfill_high_timing_features(city_key: str = None) -> str:
    """
    Compute high-timing features for all historical dates using the
    pre-dawn observation window (midnight–07:59 local), simulating what
    the model sees at its ~6am prediction run.

    Features computed:
      obs_high_peak_hour    — hour (0-23) of the running max in the midnight-8am window
      obs_is_overnight_high — 1 if the overnight max is the calendar-day high AND
                              temp was already falling by 8am (warm-front passage)
      obs_temp_falling_hrs  — consecutive falling hours from the running max by 8am

    Overnight-high detection logic (same as _detect_high_locked):
      peak_hour < 9 AND gap (peak - 8am temp) >= 2°F

    Saves to {prefix}high_timing_data.csv.
    Returns the CSV path.
    """
    import nws_auto_logger as _nal
    cfg    = _nal._CITY_CFG
    prefix = cfg.get("model_prefix", "")
    city   = city_key or _CITY_KEY
    tz_name = cfg.get("timezone", "America/New_York")

    sb_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("SUPABASE_SERVICE_ROLE", "")
    if not sb_url or not sb_key:
        print("⚠️ Missing SUPABASE_URL/SUPABASE_SERVICE_ROLE — cannot backfill high-timing")
        return ""

    # ── Fetch all obs from Supabase (paginated) ───────────────────────────
    endpoint = f"{sb_url}/rest/v1/nws_observations"
    all_obs: list[dict] = []
    offset, page_size = 0, 1000
    while True:
        params = (f"?city=eq.{city}&order=observed_at.asc"
                  f"&limit={page_size}&offset={offset}")
        req = urllib.request.Request(
            endpoint + params,
            headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}",
                     "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            all_obs.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        except Exception as e:
            print(f"⚠️ Obs fetch failed at offset {offset}: {e}")
            break

    if not all_obs:
        print(f"⚠️ No observations found for {city}")
        return ""
    print(f"📊 Fetched {len(all_obs)} obs for {city} across all dates")

    # ── Group by local date ───────────────────────────────────────────────
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(tz_name)
    obs_by_date: dict[str, list] = {}
    for obs in all_obs:
        ts = obs.get("observed_at")
        if not ts or obs.get("temp_f") is None:
            continue
        local_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
        d = local_dt.strftime("%Y-%m-%d")
        obs_by_date.setdefault(d, []).append((local_dt, obs["temp_f"]))

    print(f"📅 Spans {len(obs_by_date)} unique dates")

    # ── Compute features per date ─────────────────────────────────────────
    rows_out = []
    overnight_count = 0
    for date_str in sorted(obs_by_date.keys()):
        day_obs = sorted(obs_by_date[date_str], key=lambda x: x[0])  # chrono

        # Pre-dawn window: midnight–07:59 local (what model sees at ~6am run)
        pre_dawn = [(dt, t) for dt, t in day_obs if dt.hour < 8]
        if not pre_dawn:
            # No pre-dawn obs — use first available obs of the day
            pre_dawn = day_obs[:3] if day_obs else []
        if not pre_dawn:
            continue

        temps_pd = [t for _, t in pre_dawn]
        max_t    = max(temps_pd)
        # Hour of running max in pre-dawn window
        peak_hr  = next(dt.hour for dt, t in pre_dawn if t == max_t)
        # Current temp at end of pre-dawn window (closest to 8am)
        curr_t   = pre_dawn[-1][1]
        gap      = max_t - curr_t  # °F the temp has fallen from peak

        # Consecutive falling hours from peak (in pre-dawn window)
        falling = 0
        prev_t  = curr_t
        for _, t in reversed(pre_dawn[:-1]):
            if t > prev_t:
                break
            falling += 1
            prev_t = t

        # Overnight-high flag: peak was pre-9am AND temp already falling ≥2°F
        is_overnight = int(peak_hr < 9 and gap >= 2.0)
        if is_overnight:
            overnight_count += 1

        rows_out.append({
            "target_date":          date_str,
            "city":                 city,
            "obs_high_peak_hour":   float(peak_hr),
            "obs_is_overnight_high": float(is_overnight),
            "obs_temp_falling_hrs": float(falling),
        })

    if not rows_out:
        print("⚠️ No high-timing features computed")
        return ""

    csv_path = f"{prefix}high_timing_data.csv"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)
    print(f"✅ Saved {len(rows_out)} high-timing rows → {csv_path}")
    print(f"   Overnight highs detected: {overnight_count} "
          f"({overnight_count/len(rows_out)*100:.1f}% of days)")
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
               f"kalshi_market_snapshot,nws_d0,accuweather,atm_snapshot,ml_bucket_canonical")
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


def _add_obs_to_snap(snap: dict, live_obs: dict, live_atm: dict = None, city_key: str = None) -> None:
    """
    Add observation snapshot keys to an existing atm_snapshot dict in-place.
    These keys are prefixed obs_snap_* to avoid collision with model feature names.
    Called on both canonical write and post-recompute baseline advance.

    live_atm (optional): atmospheric dict that may carry Synoptic/NYSM keys written
        back by _compute_ml_prediction() via prefetched_atm. Used as fallback when
        live_obs doesn't have them (since _fetch_observation_features() skips them).
    city_key (optional): city for cache fallback when live Synoptic fetch fails.
    """
    if not live_obs and not live_atm:
        # Try to load cached observations if live data is empty
        if city_key:
            cached = get_cached_snapshot(city_key)
            if cached:
                obs = cached
            else:
                return
        else:
            return
    else:
        obs = live_obs or {}

    def _safe(v):
        """Return v if valid float, else None (JSON serializable)."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else round(f, 3)
        except (TypeError, ValueError):
            return None

    def _atm_fallback(obs_key: str, atm_key: str = None):
        """Read from live_obs; fall back to live_atm if null/missing."""
        v = obs.get(obs_key)
        if v is not None:
            return v
        if live_atm and atm_key:
            return live_atm.get(atm_key)
        return None

    snap["obs_snap_temp"]              = _safe(obs.get("obs_latest_temp"))
    snap["obs_snap_heating_rate"]      = _safe(obs.get("obs_heating_rate"))
    snap["obs_snap_heating_rate_delta"]= _safe(obs.get("obs_heating_rate_delta"))  # stall signal
    snap["obs_snap_vs_forecast"]       = _safe(obs.get("obs_vs_intra_forecast"))
    snap["obs_snap_hour"]        = _safe(obs.get("obs_latest_hour"))
    snap["obs_snap_max_so_far"]  = _safe(obs.get("obs_max_so_far"))
    # High-timing signals — for dashboard overnight-high warning + future model training
    snap["obs_snap_high_peak_hour"]    = _safe(obs.get("obs_high_peak_hour"))
    snap["obs_snap_is_overnight_high"] = _safe(obs.get("obs_is_overnight_high"))
    snap["obs_snap_temp_falling_hrs"]  = _safe(obs.get("obs_temp_falling_hrs"))
    snap["obs_snap_wind_speed"]  = _safe(obs.get("obs_wind_speed"))
    snap["obs_snap_cloud_cover"] = _safe(obs.get("obs_cloud_cover"))
    # Regional NWS stations (JFK / LGA)
    snap["obs_snap_jfk"]         = _safe(obs.get("obs_jfk_temp"))
    snap["obs_snap_lga"]         = _safe(obs.get("obs_lga_temp"))
    snap["obs_snap_regional_spread"] = _safe(obs.get("obs_regional_spread"))
    snap["obs_snap_regional_mean"]   = _safe(obs.get("obs_regional_mean"))
    snap["obs_snap_regional_vs_nws"] = _safe(obs.get("obs_regional_vs_nws"))
    # Synoptic Data (MesoWest) — 5mi radius.
    # _fetch_observation_features() doesn't fetch Synoptic; it's fetched inside
    # _compute_ml_prediction() and written back to prefetched_atm (live_atm here).
    snap["obs_snap_syn_mean"]    = _safe(_atm_fallback("obs_synoptic_mean",   "obs_synoptic_mean"))
    snap["obs_snap_syn_min"]     = _safe(_atm_fallback("obs_synoptic_min",    "obs_synoptic_min"))
    snap["obs_snap_syn_max"]     = _safe(_atm_fallback("obs_synoptic_max",    "obs_synoptic_max"))
    snap["obs_snap_syn_spread"]  = _safe(_atm_fallback("obs_synoptic_spread", "obs_synoptic_spread"))
    snap["obs_snap_syn_vs_nws"]  = _safe(_atm_fallback("obs_synoptic_vs_nws", "obs_synoptic_vs_nws"))
    _syn_count = _atm_fallback("obs_synoptic_count", "obs_synoptic_count")
    snap["obs_snap_syn_count"]   = _syn_count  # int OK
    # v9: named ASOS stations individually
    snap["obs_snap_kjfk"]        = _safe(_atm_fallback("obs_kjfk_temp", "obs_kjfk_temp"))
    snap["obs_snap_klga"]        = _safe(_atm_fallback("obs_klga_temp", "obs_klga_temp"))
    snap["obs_snap_kewr"]        = _safe(_atm_fallback("obs_kewr_temp", "obs_kewr_temp"))
    snap["obs_snap_kteb"]        = _safe(_atm_fallback("obs_kteb_temp", "obs_kteb_temp"))
    # KCDW (Caldwell NJ, ~25mi) and KSMQ (Somerville NJ, ~35mi) — deeper inland probes
    snap["obs_snap_kcdw"]        = _safe(_atm_fallback("obs_kcdw_temp", "obs_kcdw_temp"))
    snap["obs_snap_ksmq"]        = _safe(_atm_fallback("obs_ksmq_temp", "obs_ksmq_temp"))
    snap["obs_snap_knyc_syn"]    = _safe(_atm_fallback("obs_knyc_temp", "obs_knyc_temp"))
    snap["obs_snap_kjfk_vs_knyc"]        = _safe(_atm_fallback("obs_kjfk_vs_knyc", "obs_kjfk_vs_knyc"))
    snap["obs_snap_klga_vs_knyc"]        = _safe(_atm_fallback("obs_klga_vs_knyc", "obs_klga_vs_knyc"))
    snap["obs_snap_kewr_vs_knyc"]        = _safe(_atm_fallback("obs_kewr_vs_knyc", "obs_kewr_vs_knyc"))
    snap["obs_snap_airport_spread"]      = _safe(_atm_fallback("obs_airport_spread", "obs_airport_spread"))
    snap["obs_snap_coastal_vs_inland"]   = _safe(_atm_fallback("obs_coastal_vs_inland", "obs_coastal_vs_inland"))
    # Full coast→inland gradient: JFK to deepest available inland (SMQ > CDW > TEB > EWR)
    snap["obs_snap_inland_gradient"]     = _safe(_atm_fallback("obs_inland_gradient", "obs_inland_gradient"))
    # Manhattan Mesonet — 5-min updates, near-Central Park fill-in between KNYC hourly drops
    snap["obs_snap_manh_temp"]     = _safe(_atm_fallback("obs_manh_temp",     "obs_manh_temp"))
    snap["obs_snap_manh_vs_knyc"]  = _safe(_atm_fallback("obs_manh_vs_knyc",  "obs_manh_vs_knyc"))
    # Observation timestamps — ISO strings, not floats (skip _safe())
    # Lets dashboard show staleness: "KNYC: 52°F (47 min ago)" · "MANH: 53°F (3 min ago)"
    for _stid, _snap_key, _atm_key in [
        ("KNYC", "obs_snap_knyc_obs_at", "obs_knyc_obs_at"),
        ("KJFK", "obs_snap_kjfk_obs_at", "obs_kjfk_obs_at"),
        ("KLGA", "obs_snap_klga_obs_at", "obs_klga_obs_at"),
        ("KEWR", "obs_snap_kewr_obs_at", "obs_kewr_obs_at"),
        ("KTEB", "obs_snap_kteb_obs_at", "obs_kteb_obs_at"),
        ("KCDW", "obs_snap_kcdw_obs_at", "obs_kcdw_obs_at"),
        ("KSMQ", "obs_snap_ksmq_obs_at", "obs_ksmq_obs_at"),
        ("MANH", "obs_snap_manh_obs_at", "obs_manh_obs_at"),
    ]:
        _ts = _atm_fallback(_atm_key, _atm_key)
        snap[_snap_key] = _ts if isinstance(_ts, str) else None
    # NY State Mesonet (borough stations) — same pattern as Synoptic
    snap["obs_snap_nysm_mean"]   = _safe(_atm_fallback("obs_nysm_mean",   "obs_nysm_mean"))
    snap["obs_snap_nysm_min"]    = _safe(_atm_fallback("obs_nysm_min",    "obs_nysm_min"))
    snap["obs_snap_nysm_max"]    = _safe(_atm_fallback("obs_nysm_max",    "obs_nysm_max"))
    snap["obs_snap_nysm_spread"] = _safe(_atm_fallback("obs_nysm_spread", "obs_nysm_spread"))
    snap["obs_snap_nysm_vs_nws"] = _safe(_atm_fallback("obs_nysm_vs_nws", "obs_nysm_vs_nws"))
    _nysm_count = _atm_fallback("obs_nysm_count", "obs_nysm_count")
    snap["obs_snap_nysm_count"]  = _nysm_count  # int OK
    # WU PWS (ambient stations near Central Park)
    snap["obs_snap_wu_mean"]     = _safe(obs.get("obs_ambient_temp"))
    snap["obs_snap_wu_vs_nws"]   = _safe(obs.get("obs_ambient_vs_nws"))
    snap["obs_snap_wu_spread"]   = _safe(obs.get("obs_ambient_spread"))
    snap["obs_snap_wu_count"]    = obs.get("obs_ambient_count")  # int OK

    # ── Cirrostratus plume monitoring (non-model, for data collection) ──
    # Detects conditions suggesting high-altitude reflective clouds (cirrostratus)
    # that reduce solar radiation and suppress afternoon heating.
    # Use LIVE current values from Open-Meteo (not forecast averages) for real-time detection
    # (this is monitoring/tracking only, not fed to model)
    _sol_rad_live = live_atm.get("solar_now_wm2") if live_atm else None
    _cloud_pct_live = live_atm.get("cloud_cover_now") if live_atm else None
    # Heuristic: plume signal if solar rad < 400 W/m² AND cloud cover > 60% (during meaningful daytime)
    # Skip detection at night (solar < 50 W/m² indicates sunset/night, not plume)
    _plume_detected = None
    if _sol_rad_live is not None and _cloud_pct_live is not None:
        # Only flag plume during daytime (50 < solar < 400) with high clouds
        if 50 <= _sol_rad_live < 400 and _cloud_pct_live > 60:
            _plume_detected = 1  # flag: plume likely overhead
        else:
            _plume_detected = 0
    snap["atm_plume_monitoring"] = _plume_detected  # 1=plume, 0=clear, None=unknown
    if _plume_detected == 1:
        print(f"  ⚠️  Cirrostratus plume signal: solar_rad_now={_sol_rad_live:.0f} W/m² (suppressed), "
              f"cloud_cover_now={_cloud_pct_live:.0f}% (high)")

    obs_count = sum(
        1 for v in obs.values()
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

    Four triggers — all require being in the morning heating window (before 2pm):

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

    4. obs_warm_vs_forecast: observed temp is 2°F+ WARMER than intraday
       forecast at the same hour (9am-2pm window). Mirror of trigger 2 —
       reality is running hot, signalling a potential flip to a higher Kalshi
       bucket or a blow-past scenario. Threshold is 2°F (tighter than cold
       trigger) because upside surprises tend to move fast and have larger
       Kalshi implications. Only fires when this is a new warm reading vs
       what was stored at canonical time, to avoid re-triggering on the same
       hot observation every 30-minute cycle.
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

    # Trigger 4: Observed temp running 2°F+ WARMER than intraday forecast.
    # This is the mirror of trigger 2 — reality is ahead of the model's heating
    # curve. Signals a potential flip to a higher bucket or blow-past scenario.
    # Only valid in the 9am-2pm window (intra_map coverage), and only fires if
    # this is a NEW warm signal vs what was stored at canonical write time.
    # Threshold is 2°F (vs 3°F for cold) because upside surprises tend to be
    # faster and have bigger Kalshi implications when they occur.
    if (
        live_vs_fc is not None and live_vs_fc > 2.0
        and live_hour is not None and 9 <= live_hour <= 14
    ):
        if stored_vs_fc is None or live_vs_fc > stored_vs_fc + 1.0:
            reasons.append(
                f"obs_warm_vs_forecast: {live_vs_fc:+.1f}°F vs intraday forecast "
                f"at {int(live_hour)}:00 — reality running HOT, flip to higher bucket possible"
            )

    triggered = len(reasons) > 0
    return triggered, reasons


# Keys stored in atm_snapshot — full feature set at prediction time.
# Stored as JSONB in Supabase so the training pipeline can read them back
# directly (features-at-prediction-time → actual outcome = gold-standard rows).
def _inject_nws_sequence_to_snap(
    snap: dict,
    nws_latest: float | None,
    target_date_iso: str,
    rows: list[dict],
) -> None:
    """
    Inject NWS sequence features (nws_last, nws_d1_final, nws_overnight_jump)
    into an existing atm_snapshot dict in-place.

    These are computed inside _compute_ml_prediction() as model features but
    are not returned via live_atm (Open-Meteo only). Without this injection
    the dashboard "NWS Overnight Jump" card always shows "Awaiting next cycle".
    """
    # nws_last: the most recent NWS forecast used by the model
    if nws_latest is not None and "nws_last" not in snap:
        snap["nws_last"] = float(nws_latest)

    # nws_d1_final and nws_overnight_jump — require reading the D-1 final NWS value
    if "nws_d1_final" not in snap or "nws_overnight_jump" not in snap:
        try:
            d1f = _get_nws_d1_final(target_date_iso)
            if d1f is not None:
                snap.setdefault("nws_d1_final", float(d1f))
                # First D0 forecast = earliest NWS row for today
                d0_fc = sorted(
                    [
                        (str(r.get("timestamp", "") or r.get("forecast_time", "")),
                         float(r["predicted_high"]))
                        for r in rows
                        if r.get("forecast_or_actual") == "forecast"
                        and str(r.get("target_date", "")) == target_date_iso
                        and r.get("predicted_high") is not None
                        and (r.get("source") or "").lower() != "accuweather"
                    ],
                    key=lambda x: x[0],
                )
                if d0_fc:
                    jump = round(d0_fc[0][1] - float(d1f), 1)
                    snap.setdefault("nws_overnight_jump", jump)
        except Exception:
            pass


_ATM_SNAPSHOT_KEYS = (
    # Boundary layer / mixing
    "atm_bl_height_max", "atm_bl_height_mean",
    # Cloud cover & solar
    "atm_cloud_cover_mean", "atm_cloud_cover_max",
    "atm_solar_radiation_peak", "atm_solar_radiation_mean",
    # Surface wind (speed + circular encoding)
    "atm_wind_max", "atm_wind_mean",
    "atm_wind_dir_sin", "atm_wind_dir_cos",
    # Humidity / dewpoint / pressure
    "atm_humidity_mean", "atm_humidity_min",
    "atm_dewpoint_mean",
    "atm_pressure_mean", "atm_pressure_change",
    # Temperature structure
    "atm_temp_range", "atm_overnight_min", "atm_morning_temp_6am",
    "atm_850mb_temp_max", "atm_850mb_temp_mean",
    "atm_925mb_temp_max", "atm_925mb_temp_mean",
    "atm_precip_total",
    # Ensemble uncertainty
    "ens_spread", "ens_std", "ens_mean", "ens_skew", "ens_iqr",
    # Multi-model spread (original 5 models)
    "mm_spread", "mm_std", "mm_mean",
    "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff",
    "mm_hrrr_max", "mm_ecmwf_max", "mm_icon_max", "mm_gem_max",
    "mm_icon_gfs_diff", "mm_gem_ecmwf_diff",
    # v6: high-accuracy models (NBM #3 accuracy, GEM HRDPS #5 accuracy per wethr.net)
    "mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
    "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff",
    # v6: HRRR-specific 925mb (3km BL resolution vs GFS 13km) + GFS-HRRR cap diff
    "atm_925mb_hrrr_max", "atm_925mb_hrrr_mean",
    "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
    "atm_925mb_gfs_hrrr_diff",   # GFS - HRRR: positive = GFS too warm, missing cap
    # v6: OKX radiosonde observed upper-air soundings (actual 925mb/850mb, not model)
    "raob_925mb_temp", "raob_850mb_temp", "raob_700mb_temp",
    "raob_sounding_hour", "raob_valid",
    "raob_925mb_gfs_diff",   # GFS forecast - observed: positive = GFS missed the cap
    "raob_925mb_hrrr_diff",  # HRRR forecast - observed: positive = HRRR missed the cap
    # NWS sequence
    "nws_d1_final", "nws_overnight_jump",
    "nws_last", "nws_first", "nws_mean", "nws_spread",
    "accu_last", "accu_mean",
    # Intraday revision deltas
    "nws_post_9am_delta", "accu_post_9am_delta",
    # Carryover
    "prev_day_high", "prev_day_temp_drop", "midnight_temp",
    "atm_predicted_high", "atm_vs_forecast_diff",
    # v8: stall signal — was warming, now plateau? Negative = cap holding.
    "obs_heating_rate_delta",
    # Synoptic multi-station network (15-25 stations within 10mi: KNYC+airports+Mesonet)
    # IMPORTANT: these were fetched at inference but not persisted — fixed here.
    # On a cap day, obs_synoptic_spread ≈ 0 (all stations capped), obs_synoptic_vs_nws ≈ -5.
    "obs_synoptic_mean", "obs_synoptic_min", "obs_synoptic_max",
    "obs_synoptic_spread", "obs_synoptic_vs_nws", "obs_synoptic_count",
    # NY State Mesonet borough stations (Brooklyn, Queens, Bronx, Manhattan, Staten Island)
    "obs_nysm_mean", "obs_nysm_min", "obs_nysm_max",
    "obs_nysm_spread", "obs_nysm_vs_nws", "obs_nysm_count",
    # WU PWS citizen stations near Central Park
    "obs_ambient_temp", "obs_ambient_vs_nws", "obs_ambient_spread", "obs_ambient_count",
    # v9: named ASOS stations individually (marine cap signal)
    # KJFK (coastal) < KLGA < KNYC < KEWR < KTEB (inland) on cap days
    "obs_kjfk_temp", "obs_klga_temp", "obs_kewr_temp", "obs_kteb_temp", "obs_knyc_temp",
    "obs_kjfk_vs_knyc",      # negative = sea breeze reaching Central Park
    "obs_klga_vs_knyc",
    "obs_kewr_vs_knyc",      # positive = NJ warmer = no marine cap
    "obs_airport_spread",    # near-zero = all airports uniformly capped
    "obs_coastal_vs_inland", # mean(KJFK,KLGA) - mean(KEWR,KTEB): negative = marine
    # obs_heating_rate_delta already listed above under v8
    # v10: NNJ inland gradient (KCDW + KSMQ, up to 35mi inland)
    "obs_snap_kcdw", "obs_snap_ksmq",
    "obs_snap_inland_gradient",     # KSMQ - JFK (positive = warm air advancing from west)
    "obs_snap_inland_warming_rate", # rate of intraday gradient change
    # v10: scored outcome features (written at scoring time, not prediction time)
    # These are the ground-truth labels for training the blow-past and oscillation models.
    "scored_blow_past_occurred",    # bool: actual_high > bucket_ceiling (did it blow past?)
    "scored_actual_vs_ceiling",     # float: actual_high - bucket_ceiling (magnitude of blow-past or miss)
    "scored_intraday_flip_count",   # int: confirmed model flip count for the scored day
    "scored_flip_history",          # list[str]: sequence of confirmed non-canonical buckets
    "scored_obs_inland_gradient",   # float: NNJ gradient at time of last obs update
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
    # Fallback: direct bucket check (no Kalshi snapshot).
    # Kalshi "68-69" covers both 68°F and 69°F — inclusive on both ends.
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
               f"ml_bucket_canonical,ml_f_canonical,ml_result_canonical,"
               f"ml_bucket_2,ml_bucket_2_prob,bucket_rank_hit,atm_snapshot")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            pred_rows = json.loads(resp.read().decode("utf-8"))

        if not pred_rows or not pred_rows[0].get("ml_bucket"):
            return
        pred = pred_rows[0]
        bucket_2 = pred.get("ml_bucket_2")  # second-best Kalshi bucket (may be None for old rows)

        # Already fully scored with same actual? Skip.
        # Re-score if actual changed (e.g., CLI updated overnight), OR if
        # ml_result_canonical is still null (canonical was written after scoring ran).
        prev_actual = _float_or_none(pred.get("ml_actual_high"))
        fully_scored = (
            pred.get("ml_result") is not None
            and pred.get("ml_result_canonical") is not None
            and prev_actual is not None
            and abs(prev_actual - actual_high) < 0.1
        )
        if fully_scored:
            return  # both latest and canonical scored with same actual — nothing to do

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

        # Fallback: direct bucket check (when no Kalshi snapshot available).
        # Kalshi "68-69" covers both 68°F and 69°F — inclusive on both ends.
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

        # ── Compute blow-past and oscillation features for ML training ───────────
        # Blow-past: did the actual high exceed the TOP of the predicted bucket?
        # For "lo-hi" buckets the ceiling is `hi`.  ">=" buckets have no ceiling.
        _bucket_ceiling: "float | None" = None
        if "-" in ml_bucket:
            _bk_parts = ml_bucket.split("-")
            if len(_bk_parts) == 2:
                try: _bucket_ceiling = float(_bk_parts[1])
                except ValueError: pass
        elif ml_bucket.startswith("<="):
            try: _bucket_ceiling = float(ml_bucket[2:])
            except ValueError: pass
        # ">=" is unbounded above — no ceiling, blow-past not applicable.
        _blow_past_occurred = bool(
            _bucket_ceiling is not None and actual_high > _bucket_ceiling
        )
        _actual_vs_ceiling = (
            round(actual_high - _bucket_ceiling, 1)
            if _bucket_ceiling is not None else None
        )

        # Read flip state from yesterday's atm_snapshot (already fetched above).
        _atm_raw = pred.get("atm_snapshot")
        try:
            _atm_snap: dict = (
                json.loads(_atm_raw) if isinstance(_atm_raw, str)
                else (_atm_raw if isinstance(_atm_raw, dict) else {})
            ) if _atm_raw else {}
        except Exception:
            _atm_snap = {}
        _intraday_flip_count = int(_atm_snap.get("ml_flip_count") or 0)
        _flip_history        = _atm_snap.get("ml_flip_history") or []

        # Persist scored features back into atm_snapshot so the training pipeline
        # can use them as ground-truth labels / contextual features next season.
        _atm_snap["scored_blow_past_occurred"]  = _blow_past_occurred
        _atm_snap["scored_intraday_flip_count"] = _intraday_flip_count
        if _actual_vs_ceiling is not None:
            _atm_snap["scored_actual_vs_ceiling"] = _actual_vs_ceiling
        if isinstance(_flip_history, list) and _flip_history:
            _atm_snap["scored_flip_history"] = _flip_history
        # inland gradient at scoring time (already in snap from obs pipeline)
        _atm_snap.setdefault("scored_obs_inland_gradient",
                             _atm_snap.get("obs_snap_inland_gradient"))

        _ceil_str = (f"ceiling={_bucket_ceiling}°F | delta={_actual_vs_ceiling:+.1f}°F"
                     if _actual_vs_ceiling is not None else "no ceiling (≥ bucket)")
        print(f"  📈 Scored ML features: blow_past={_blow_past_occurred} "
              f"(actual={actual_high}°F vs {_ceil_str}), "
              f"flip_count={_intraday_flip_count}")
        # ────────────────────────────────────────────────────────────────────────

        # Score bucket rank: 1=bucket1 hit, 2=bucket2 hit (bucket1 missed), 0=both missed
        bucket_rank_hit: Optional[int] = None
        if is_win:
            bucket_rank_hit = 1
        elif bucket_2:
            # Check if bucket 2 was correct
            b2_win = _score_bucket(bucket_2, actual_int, pred.get("kalshi_market_snapshot"))
            bucket_rank_hit = 2 if b2_win else 0
            if b2_win:
                print(f"🥈 Bucket 2 was right: '{bucket_2}' matched actual {actual_high}°F")
            else:
                print(f"❌ Both buckets missed: '{ml_bucket}' and '{bucket_2}' vs actual {actual_high}°F")
        else:
            bucket_rank_hit = 0  # no bucket 2 stored (old row) — mark as miss

        # Score canonical (first-of-day) prediction separately for comparison research.
        # Include the updated atm_snapshot so scored ML features are persisted.
        patch_data: dict = {"ml_result": result, "ml_actual_high": actual_high,
                            "bucket_rank_hit": bucket_rank_hit,
                            "atm_snapshot": json.dumps(_atm_snap)}
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


def backfill_canonical_results() -> None:
    """
    One-time (and ongoing) cleanup: score any historical rows that have
    ml_bucket_canonical + ml_actual_high but are missing ml_result_canonical.

    Runs on every write_both_snapshots call — idempotent, skips already-scored rows.
    Fixes the gap left by the original settlement early-return bug.
    """
    try:
        endpoint, key = _sb_endpoint()
        # Fetch rows with canonical set + actual known but result missing
        url = (f"{endpoint}?city=eq.{_CITY_KEY}"
               f"&ml_bucket_canonical=not.is.null"
               f"&ml_actual_high=not.is.null"
               f"&ml_result_canonical=is.null"
               f"&select=idempotency_key,ml_bucket_canonical,ml_f_canonical,"
               f"ml_actual_high,kalshi_market_snapshot")
        req = urllib.request.Request(url, headers={
            "apikey": key, "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            rows_to_score = json.loads(resp.read().decode("utf-8"))

        if not rows_to_score:
            return

        print(f"🔁 Backfilling ml_result_canonical for {len(rows_to_score)} row(s)...")
        for row in rows_to_score:
            idem_key  = row.get("idempotency_key")
            canonical = row.get("ml_bucket_canonical")
            actual    = _float_or_none(row.get("ml_actual_high"))
            ks_raw    = row.get("kalshi_market_snapshot")
            if not idem_key or not canonical or actual is None:
                continue

            actual_int = int(round(actual))
            canon_win  = _score_bucket(canonical, actual_int, ks_raw)
            canon_result = "WIN" if canon_win else "MISS"

            patch = json.dumps({"ml_result_canonical": canon_result}).encode("utf-8")
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
            icon = "✅" if canon_win else "❌"
            print(f"{icon} Backfilled canonical result: {canonical} vs {actual}°F → {canon_result} "
                  f"({idem_key})")

    except Exception as e:
        print(f"⚠️ backfill_canonical_results failed: {e}")


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


# ── Flip-field carry-forward (module scope — used by both write_today_for_today
# and write_today_for_tomorrow so it cannot be a local function) ──────────────
_FLIP_FIELDS = (
    "ml_flip_occurred", "ml_flip_count", "ml_flip_bucket",
    "ml_flip_history", "ml_flip_pending_bucket",
)

def _carry_flip_fields(snap: dict, src_existing: dict) -> None:
    """Copy flip tracking fields from the previous snapshot into snap (in-place).

    CRITICAL: atm_snapshot is replaced wholesale by Supabase merge-duplicates,
    not merged key-by-key.  So any path that rebuilds the snapshot from live ATM
    data (recompute, canonical write, D+1 refresh) would silently erase all flip
    history unless we explicitly carry these fields forward here.

    Only copies fields that are absent from snap — never overwrites a value that
    was freshly set by the flip state machine later in the same cycle."""
    _src_raw = src_existing.get("atm_snapshot") if isinstance(src_existing, dict) else None
    if not _src_raw:
        return
    try:
        _src = json.loads(_src_raw) if isinstance(_src_raw, str) else _src_raw
    except Exception:
        return
    if not isinstance(_src, dict):
        return
    copied = []
    for fk in _FLIP_FIELDS:
        if fk in _src and fk not in snap:
            snap[fk] = _src[fk]
            copied.append(fk)
    if copied:
        print(f"  🔒 Carried flip fields into snapshot: {copied}")


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

    # Two-tier D0 freeze (seasonal):
    #   agency_cutoff: freeze NWS/AccuWeather/obs triggers — these use observed
    #     surface data and agency echoes, which contaminate the prediction after peak heating.
    #     Time varies by season: 2 PM (winter), 3 PM (spring/fall), 4 PM (summer).
    #   atm_cutoff: freeze atmospheric triggers — BL height, 925mb temps, GFS/HRRR
    #     ensemble data are MODEL-derived (not surface-observation-derived), so they can
    #     safely fire until atm_cutoff. A collapsing BL is independent evidence the
    #     mixing layer is done without looking at the thermometer.
    # The canonical write guard uses atm_cutoff (full freeze) so is_canonical_write is
    # only True in the morning before any cutoff fires.
    agency_cutoff_hour = _get_agency_cutoff_hour(_CITY_KEY)
    atm_cutoff_hour = _get_atm_cutoff_hour(_CITY_KEY)
    agency_cutoff = now_nyc().hour >= agency_cutoff_hour
    atm_cutoff    = now_nyc().hour >= atm_cutoff_hour
    past_cutoff   = atm_cutoff  # full freeze = atm cutoff (used for canonical write gate)
    live_atm: Optional[dict] = None

    # Initialize live_atm / live_obs here so they're always defined regardless
    # of which branch below executes (full-freeze, cutoff, normal, first-write).
    # The stable-day snapshot refresh block references both unconditionally.
    live_atm: dict = {}
    live_obs: dict = {}

    # ── mm_* carry-key registry and morning-anchor helper ─────────────────────
    # Defined at function scope so they're available in ALL branches:
    #   - the normal intraday branch (carry-forward, restoration loop)
    #   - the LOCK_NOT_FOUND branch (canonical write for first write of the day)
    #   - the stable-cycle branch (self-heal and obs-panel refresh)
    _MM_CARRY_KEYS = (
        "mm_spread", "mm_std", "mm_mean",
        "mm_hrrr_max", "mm_nbm_max", "mm_gem_max", "mm_icon_max",
        "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff",
        "mm_icon_gfs_diff", "mm_gem_ecmwf_diff",
        "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
        "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff",
        "atm_925mb_hrrr_max", "atm_925mb_hrrr_mean",
        "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
        "atm_925mb_gfs_hrrr_diff",
    )

    def _morning_anchor_key(ck: str) -> str:
        """Map a carry-key to its frozen daily morning-anchor key.

        Naming convention (keeps all anchors under the mm_morning_ namespace):
          mm_spread            → mm_morning_spread
          mm_hrrr_max          → mm_morning_hrrr_max
          atm_925mb_hrrr_max   → mm_morning_atm_925mb_hrrr_max

        Written once at canonical write, NEVER overwritten intraday.
        Acts as the ultimate fallback when consecutive Open-Meteo ensemble
        failures deplete the stored_snapshot_dict of all mm_* values.
        """
        return "mm_morning_" + (ck[3:] if ck.startswith("mm_") else ck)

    # ── Dynamic high-lock: detects overnight highs and clearly-peaked late highs
    # regardless of clock time.  Runs BEFORE the hard clock cutoffs so it can
    # lock a prediction at 9am if the high occurred at 1am, or at 4pm if the
    # high peaked at 2pm and has been falling for 2+ hours.
    _dlock = _detect_high_locked(target_date_iso,
                                 nws_forecast=float(nws_latest) if nws_latest is not None else None)
    if _dlock["locked"] and not past_cutoff:
        # Override both cutoffs — treat as if atm_cutoff already fired
        agency_cutoff = True
        atm_cutoff    = True
        past_cutoff   = True
        print(f"🔒 Dynamic high-lock overrides clock cutoffs: {_dlock['reason']}")

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
        # Past the full freeze point (atm cutoff OR dynamic lock).
        # All signal types (agency, atmospheric, obs) are now stale or contaminated.
        lock_label = _dlock["reason"] if _dlock["locked"] else f"atm cutoff ({_D0_ATM_CUTOFF_HOUR_LOCAL.get(_CITY_KEY, 15)}:00 local)"
        print(f"⏸️ Full freeze [{lock_label}] — ML held: "
              f"{existing['ml_f']}°F → {existing.get('ml_bucket')}")
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
        # Atmospheric fetch runs until the atm_cutoff (3pm); agency/obs checks
        # only run until the agency_cutoff (2pm).
        stored_atm_snapshot = existing.get("atm_snapshot")
        try:
            _parsed_snap = (
                json.loads(stored_atm_snapshot)
                if isinstance(stored_atm_snapshot, str)
                else stored_atm_snapshot
            )
            # Guard: JSONB column must be a dict.  If it was ever written as a JSON
            # array (e.g. from an early buggy serialisation) json.loads returns a list,
            # causing 'list' object has no attribute 'get' on subsequent .get() calls.
            stored_snapshot_dict = _parsed_snap if isinstance(_parsed_snap, dict) else {}
            if _parsed_snap and not isinstance(_parsed_snap, dict):
                print(f"  ⚠️  atm_snapshot is {type(_parsed_snap).__name__}, not dict — ignoring")
        except Exception:
            stored_snapshot_dict = {}

        try:
            live_atm = _fetch_atmospheric_features(target_date_iso)
        except Exception as _atm_e:
            print(f"⚠️ Live atmospheric fetch failed: {_atm_e}")
            live_atm = {}

        # ── Carry-forward: seed live_atm with previous snapshot's mm_* / ATM values
        # if the fresh Open-Meteo ensemble fetch missed them (partial or timeout).
        #
        # WHY THIS MATTERS: The Open-Meteo ensemble endpoint sometimes returns null
        # for individual members (HRRR, NBM, GEM) even when the base forecast works.
        # Without this carry-forward:
        #   1. The ML model runs with NaN for mm_hrrr_max / mm_spread → prediction
        #      shifts toward the NWS baseline, flipping the displayed bucket.
        #   2. The snapshot is written without those keys → dashboard shows
        #      "Awaiting next cycle" for Multi-Model Spread / HRRR vs NWS.
        # The carry-forward only activates when the fresh fetch returns None/NaN for
        # a key; if HRRR genuinely updated, the fresh value takes precedence.
        if stored_snapshot_dict and live_atm:
            _carried         = []
            _morning_rescued = []
            for _ck in _MM_CARRY_KEYS:
                _fresh_v = live_atm.get(_ck)
                _prev_v  = stored_snapshot_dict.get(_ck)
                if (_fresh_v is None or (isinstance(_fresh_v, float) and math.isnan(_fresh_v))) \
                        and _prev_v is not None:
                    # Primary carry-forward: use the previous snapshot's value.
                    live_atm[_ck] = _prev_v
                    _carried.append(_ck)
                elif (_fresh_v is None or (isinstance(_fresh_v, float) and math.isnan(_fresh_v))) \
                        and _prev_v is None:
                    # Tertiary fallback: morning anchor (immune to cascading API failures).
                    # If consecutive fetch failures have also wiped the previous snapshot
                    # of mm_* values (cascading null), carry-forward has nothing to copy.
                    # The mm_morning_* anchor is written once at canonical time and never
                    # overwritten, so it's always available as the day-of baseline.
                    _ak  = _morning_anchor_key(_ck)
                    _mav = stored_snapshot_dict.get(_ak)
                    if _mav is not None:
                        live_atm[_ck] = _mav
                        _morning_rescued.append(_ck)
            if _carried:
                print(f"  🔁 Carried {len(_carried)} mm_/atm_ keys from prev snapshot "
                      f"(ensemble fetch partial): "
                      f"{', '.join(_carried[:5])}{'…' if len(_carried) > 5 else ''}")
            if _morning_rescued:
                print(f"  🌅 Morning anchor rescued {len(_morning_rescued)} mm_/atm_ key(s) "
                      f"(cascading-null prevention — both live + prev snapshot were null): "
                      f"{', '.join(_morning_rescued[:5])}{'…' if len(_morning_rescued) > 5 else ''}")

        # Fetch observation features for DISPLAY (slope, wind, cloud, etc.) throughout day.
        # But only use them for ML triggers before agency_cutoff (2pm). After 2pm, station
        # temps reflect observed reality — using them for triggers is thermometer-chasing.
        # Display features (slope, wind direction) continue to update for dashboard visibility.
        live_obs = {}
        try:
            live_obs = _fetch_observation_features(
                target_date_iso,
                nws_last=nws_latest,
                atm_features=live_atm,
            )
        except Exception as _obs_e:
            print(f"⚠️ Live obs fetch failed: {_obs_e}")

        atm_triggered, atm_reasons = _check_atmospheric_shift(live_atm, stored_atm_snapshot)
        obs_triggered, obs_reasons = (
            _check_obs_trigger(live_obs, stored_snapshot_dict)
            if not agency_cutoff else (False, [])
        )

        # After agency_cutoff: NWS/AccuWeather/obs triggers are silenced.
        # Atmospheric trigger (BL collapse, ensemble shift) stays live until atm_cutoff.
        if agency_cutoff:
            nws_revised = False   # agencies echo observed temps after 2pm
            accu_revised = False

        # AccuWeather revisions advance the stored baseline (so future comparisons are
        # correct) but do NOT trigger an ML recompute — AccuWeather often echoes observed
        # temps intraday, which would cause the ML to chase the thermometer.
        # NWS, atmospheric shifts, and obs ground-truth are the independent signals.
        if nws_revised or atm_triggered or obs_triggered:
            trigger_reasons = []
            if nws_revised:
                trigger_reasons.append(f"NWS {stored_nws:.0f}→{nws_latest:.0f}°F")
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
        elif accu_revised:
            # AccuWeather revised — advance baseline only, no ML recompute.
            accu_disp = f"{stored_accu:.0f}→{accu_latest:.0f}°F"
            print(f"📌 AccuWeather revised ({accu_disp}) — advancing baseline, ML held: "
                  f"{existing['ml_f']}°F → {existing.get('ml_bucket')}")
            ml_recomputed = True  # causes nws_d0/accuweather baseline to advance in payload
            ml = {
                "ml_f": existing["ml_f"],
                "ml_bucket": existing["ml_bucket"],
                "ml_confidence": existing["ml_confidence"],
                "ml_bucket_probs": existing.get("ml_bucket_probs"),
                "ml_version": existing.get("ml_version"),
            }

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
        if isinstance(existing, dict):
            print(f"🔄 Recomputing ML prediction (previous: {existing.get('ml_f')}°F → {existing.get('ml_bucket')})")
        ml = _compute_ml_prediction(rows, target_date_iso)
        ml_recomputed = True
        # Fetch atm + obs for canonical snapshot storage (LOCK_NOT_FOUND path).
        # _compute_ml_prediction fetches atm internally but doesn't return it;
        # we need it here to store the morning baseline for intraday shift detection.
        if existing is _LOCK_NOT_FOUND:
            try:
                live_atm = _fetch_atmospheric_features(target_date_iso)
            except Exception:
                live_atm = {}
            # Fallback: if fetch failed, use cached snapshot to ensure we always have
            # at least some atmospheric data for display cards on the first write.
            if not live_atm or not any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in live_atm.values()):
                try:
                    cached = get_cached_snapshot(_CITY_KEY)
                    if cached:
                        live_atm = fill_missing_from_cache(_CITY_KEY, live_atm)
                        print(f"📦 Filled live_atm from cache after fetch failure")
                except Exception:
                    pass
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

    # Canonical = the morning ML prediction for this date (set once, never overwritten).
    # Fires when: (a) no row exists yet, OR (b) a D1 row exists but lacks a canonical
    # (the common case — D1 runs the evening before, D0 must claim canonical next morning).
    # Guarded by `not past_cutoff` so we never retroactively set canonical after peak heating.
    existing_has_canonical = isinstance(existing, dict) and existing.get("ml_bucket_canonical") is not None
    is_canonical_write = (
        (existing is _LOCK_NOT_FOUND or not existing_has_canonical)
        and ml is not None
        and not past_cutoff
    )

    if bcp is None and ml is None:
        print("⏭️ today_for_today: no BCP data and no ML prediction available."); return

    ts = now_nyc().isoformat()
    now_hour = int(now_nyc().strftime("%H"))
    # Archive snapshots every 3 hours (9, 12, 15, 18 EST = 1, 4, 7, 10 PM + 12, 3, 6 AM for overnight)
    # Map any hour to its 3-hour bucket: 0-2→0, 3-5→3, 6-8→6, 9-11→9, 12-14→12, 15-17→15, 18-20→18, 21-23→21
    snapshot_hour = (now_hour // 3) * 3
    # Use hourly idempotency key for snapshot archival so each 3-hour bucket creates a new row
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{target_date_iso}:{snapshot_hour:02d}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": target_date_iso,
        "snapshot_hour": snapshot_hour,  # Archive this snapshot's 3-hour bucket
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
    # Always write nws_d0 so the dashboard fallback (ml.todayNwsD0) and the
    # NWS Overnight Jump card always have a current value — even on stable cycles
    # where ml_recomputed=False. The ATM trigger check already reads
    # existing.get("nws_d0") earlier in this function, so updating the payload
    # here does NOT affect trigger detection (it reads the OLD stored value).
    payload["nws_d0"] = nws_latest
    # accuweather baseline: only advance when ML ran (preserves the baseline
    # the model actually used so future ATM checks detect genuine revisions).
    if ml_recomputed:
        payload["accuweather"] = accu_latest
        # Advance atmospheric + obs snapshot to the current live values so the next
        # comparison starts from the post-revision baseline, not the morning one.
        # (Only on non-canonical writes — canonical write handled separately below.)
        if not is_canonical_write and live_atm and any(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in live_atm.values()
        ):
            # Read the existing snapshot so we can preserve obs_snap_* display keys
            # that were written by the stable-cycle refresh.  On AccuWeather-only
            # baseline advances (accu_revised), _compute_ml_prediction is never
            # called, so live_atm has no Synoptic data — _add_obs_to_snap would
            # write obs_snap_syn_mean=None and wipe the good value.  We restore
            # any obs_snap_* key that became None after _add_obs_to_snap if the
            # existing snapshot had a valid value for it.
            _prev_snap_raw = existing.get("atm_snapshot") if isinstance(existing, dict) else None
            _prev_snap = (
                json.loads(_prev_snap_raw) if isinstance(_prev_snap_raw, str)
                else (_prev_snap_raw if isinstance(_prev_snap_raw, dict) else {})
            )
            snap = {k: live_atm[k] for k in _ATM_SNAPSHOT_KEYS if k in live_atm}
            if snap:
                # Inject NWS sequence features — computed inside _compute_ml_prediction()
                # but not returned via live_atm (which is Open-Meteo only).
                # These power the "NWS Overnight Jump" dashboard card.
                _inject_nws_sequence_to_snap(snap, nws_latest, target_date_iso, rows)
                # Also advance obs snapshot keys so triggers don't re-fire on same reading
                # Pass live_atm so Synoptic/NYSM keys written back by _compute_ml_prediction
                # via prefetched_atm are also included (they're not in live_obs).
                _add_obs_to_snap(snap, live_obs, live_atm, city_key=_CITY_KEY)
                # Restore display keys from the previous snapshot that the fresh write
                # left as None — covers two cases:
                #   obs_snap_*: Synoptic/NYSM data absent on AccuWeather-only advance path
                #   mm_* / atm_925mb_hrrr_* etc: ensemble endpoint returned partial data
                #     (the carry-forward above handles this for the ML model; this is the
                #     safety-net for any residual gaps after _add_obs_to_snap runs).
                for _pk, _pv in _prev_snap.items():
                    if (_pk.startswith("obs_snap_") or _pk.startswith("mm_")
                            or _pk in _MM_CARRY_KEYS) \
                            and _pv is not None and snap.get(_pk) is None:
                        snap[_pk] = _pv
                # ── Last-resort mm_* re-fetch ────────────────────────────────
                # If mm_spread is STILL null after carry-forward + restoration
                # (meaning both live_atm and stored_snapshot_dict lacked it —
                # e.g. consecutive fetch failures or first run of the day), do
                # one targeted re-fetch.  This recovers the currently-stuck
                # "Awaiting next cycle" state without waiting for a stable cycle.
                if snap.get("mm_spread") is None:
                    try:
                        _rescue_atm = _fetch_atmospheric_features(target_date_iso)
                        _rescued = []
                        for _rk in _MM_CARRY_KEYS:
                            _rv = _rescue_atm.get(_rk)
                            if _rv is not None and not (isinstance(_rv, float) and math.isnan(_rv)):
                                snap[_rk] = _rv
                                _rescued.append(_rk)
                        if _rescued:
                            print(f"  🆘 Last-resort mm_* re-fetch recovered {len(_rescued)} keys "
                                  f"(mm_spread={snap.get('mm_spread')}, "
                                  f"mm_hrrr_max={snap.get('mm_hrrr_max')})")
                    except Exception as _rescue_e:
                        print(f"  ⚠️  Last-resort mm_* re-fetch failed: {_rescue_e}")
                # ── Propagate morning cloud anchor into recompute snap ─────────
                # The morning cloud was frozen at canonical write.  If the recompute
                # path rebuilt snap from scratch, restore it from stored_snapshot_dict
                # so the stratus-clearing signal isn't lost.
                if snap.get("obs_snap_morning_cloud") is None:
                    _recompute_mc = stored_snapshot_dict.get("obs_snap_morning_cloud")
                    if _recompute_mc is not None:
                        snap["obs_snap_morning_cloud"] = _recompute_mc
                # ── Stratus clearing on recompute path ────────────────────────
                _r_mc = snap.get("obs_snap_morning_cloud")
                _r_cc = snap.get("obs_snap_cloud_cover")
                _r_hr = snap.get("obs_snap_hour")
                if _r_mc is not None and _r_cc is not None and _r_hr is not None:
                    snap["obs_snap_stratus_clearing"] = bool(
                        float(_r_mc) >= 50
                        and float(_r_cc) <= 25
                        and 11 <= float(_r_hr) <= 19
                    )
                # ── Inland NJ warming deltas on recompute path ────────────────
                # Compare current snap KTEB/KEWR vs stored_snapshot_dict so the
                # inland warming rate is available immediately after ML reruns,
                # not 30 min later when the next stable cycle writes it.
                if _CITY_KEY != "lax":
                    def _rk_delta(key: str) -> "float | None":
                        """Compute inter-run delta for a named snap key, falling back to stored."""
                        _prev = stored_snapshot_dict.get(key)
                        _curr = snap.get(key)
                        if _prev is not None and _curr is not None:
                            _d = round(float(_curr) - float(_prev), 1)
                            snap[f"{key}_delta"] = _d
                            return _d
                        _stored_d = stored_snapshot_dict.get(f"{key}_delta")
                        if _stored_d is not None:
                            snap[f"{key}_delta"] = _stored_d
                        return _stored_d

                    _rk_kteb_d = _rk_delta("obs_snap_kteb")
                    _rk_kewr_d = _rk_delta("obs_snap_kewr")
                    _rk_kcdw_d = _rk_delta("obs_snap_kcdw")
                    _rk_ksmq_d = _rk_delta("obs_snap_ksmq")

                    _rk_inland_vals = [v for v in [_rk_kteb_d, _rk_kewr_d, _rk_kcdw_d, _rk_ksmq_d] if v is not None]
                    if _rk_inland_vals:
                        _rk_inland_rate = round(sum(_rk_inland_vals) / len(_rk_inland_vals), 1)
                        snap["obs_snap_inland_warming_rate"] = _rk_inland_rate
                        _rk_surf_r = snap.get("obs_snap_heating_rate")
                        snap["obs_snap_warming_accel"] = bool(
                            _rk_inland_rate >= 1.5
                            and _rk_surf_r is not None
                            and float(_rk_surf_r) >= 0.3
                        )
                    elif stored_snapshot_dict.get("obs_snap_inland_warming_rate") is not None:
                        # Preserve previous cycle's rate when deltas can't be computed
                        snap["obs_snap_inland_warming_rate"] = stored_snapshot_dict["obs_snap_inland_warming_rate"]
                        snap["obs_snap_warming_accel"] = stored_snapshot_dict.get("obs_snap_warming_accel", False)

                # ── Carry flip fields before overwriting snapshot (recompute path) ──
                # Supabase merge-duplicates replaces the whole JSONB column, so any
                # flip tracking state from prior cycles would be erased without this.
                if isinstance(existing, dict):
                    _carry_flip_fields(snap, existing)

                # ── Cache complete snapshot for fallback on next API failures ──
                cache_snapshot(_CITY_KEY, snap)

                payload["atm_snapshot"] = json.dumps(snap)
                print(f"📸 Atmospheric baseline advanced after recompute ({len(snap)} keys)")

    # ── Unconditional obs-panel refresh ──────────────────────────────────────
    # On stable days (ml_recomputed = False) the existing atm_snapshot is never
    # rewritten above, so Synoptic / NWS Overnight Jump stay stale forever.
    # We merge fresh obs-display keys into the stored snapshot every cycle without
    # touching the ATM trigger-baseline keys (those must stay at their morning
    # values for change-detection to work correctly).
    #
    # IMPORTANT: run even when existing atm_snapshot is null/empty.  If a prior
    # recompute wiped the snapshot (e.g. Synoptic failed mid-day, or the canonical
    # write happened on a Full Freeze day with live_atm={}), the stable cycle must
    # be able to self-heal by writing display keys from scratch.  The `if _ex_snap:`
    # guard was the bug: a null snapshot can never recover if we skip the block.
    if not is_canonical_write and not ml_recomputed and isinstance(existing, dict):
        try:
            # Supabase returns JSONB columns as already-parsed Python dicts,
            # not as strings.  json.loads(dict) raises TypeError, which the
            # outer except block silently catches — causing the entire refresh
            # to be skipped every cycle after the first.  Handle both cases:
            _ex_snap_raw = existing.get("atm_snapshot")
            try:
                _ex_snap = (
                    json.loads(_ex_snap_raw) if isinstance(_ex_snap_raw, str)
                    else (_ex_snap_raw if isinstance(_ex_snap_raw, dict) else {})
                )
            except Exception:
                _ex_snap = {}
            # ── Preserve pre-modification snapshot for derived signal deltas ──
            # stored_snapshot_dict is used by inland warming / stratus clearing
            # to compare current values against previous cycle.  On the recompute
            # path it's defined earlier; on the stable-cycle path we define it here.
            stored_snapshot_dict = dict(_ex_snap)
            if True:  # always run — see comment above re: self-healing null snapshots
                # ── Sanitize NaN → None across entire snapshot before any writes ──
                # Python json.dumps serializes float('nan') as the non-standard token
                # NaN, which Postgres JSONB rejects and stores as null.  On the next
                # read those keys come back as None and get written as None on every
                # subsequent stable cycle.  Sanitize once here so all values that
                # survived as Python nan are written as JSON null (not NaN).
                def _san(v):
                    if isinstance(v, float) and math.isnan(v):
                        return None
                    return v
                _ex_snap = {k: _san(v) for k, v in _ex_snap.items()}

                # ── Re-inject mm_spread and companion ATM keys if null ─────────
                # mm_spread is set during ML recompute via _fetch_atmospheric_features.
                # On Full Freeze days ML never reruns, so if the canonical write had
                # live_atm={} (lock fired before first run), mm_spread stays null.
                # Fetch fresh ATM features here so the panel has data.
                _MM_KEYS = (
                    "mm_spread", "mm_std", "mm_mean",
                    "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff",
                    "mm_hrrr_max", "mm_ecmwf_max", "mm_icon_max", "mm_gem_max",
                    "mm_icon_gfs_diff", "mm_gem_ecmwf_diff",
                    # v6 additions — high-accuracy models (NBM, GEM HRDPS)
                    "mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
                    "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff",
                    # v6 HRRR-specific 925mb and GFS-HRRR diff
                    "atm_925mb_hrrr_max", "atm_925mb_hrrr_mean",
                    "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
                    "atm_925mb_gfs_hrrr_diff",
                    # v6 radiosonde
                    "raob_925mb_temp", "raob_850mb_temp", "raob_700mb_temp",
                    "raob_925mb_gfs_diff", "raob_925mb_hrrr_diff",
                )
                # Trigger if ANY key display card uses is null — not just mm_spread.
                # mm_hrrr_max can be null independently (HRRR vs NWS gap blank) even
                # when mm_spread is populated, and would never be healed under the old
                # mm_spread-only guard.
                _mm_needs_heal = any(
                    _ex_snap.get(k) is None
                    for k in ("mm_spread", "mm_hrrr_max", "mm_nbm_max", "mm_gem_max")
                )
                if _mm_needs_heal:
                    try:
                        _fresh_atm = _fetch_atmospheric_features(target_date_iso)
                        _injected = []
                        for _k in _MM_KEYS:
                            if _k in _fresh_atm:
                                v = _fresh_atm[_k]
                                _ex_snap[_k] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
                                if _ex_snap[_k] is not None:
                                    _injected.append(_k)
                        if _injected:
                            print(f"  📊 Stable-cycle mm_* heal: injected {len(_injected)} keys "
                                  f"(mm_spread={_ex_snap.get('mm_spread')}, "
                                  f"mm_hrrr_max={_ex_snap.get('mm_hrrr_max')})")
                        else:
                            print(f"  ⚠️  Stable-cycle mm_* heal: Open-Meteo returned null for all ensemble keys")
                    except Exception as _mm_e:
                        print(f"  ⚠️  mm_spread refresh skipped: {_mm_e}")

                # Refresh NWS display values (always overwrite — not setdefault)
                if nws_latest is not None:
                    _ex_snap["nws_last"] = float(nws_latest)
                try:
                    _d1f = _get_nws_d1_final(target_date_iso)
                    if _d1f is not None:
                        _ex_snap["nws_d1_final"] = float(_d1f)
                        _d0_fc = sorted([
                            (str(r.get("timestamp", "")), float(r["predicted_high"]))
                            for r in rows
                            if r.get("forecast_or_actual") == "forecast"
                            and str(r.get("target_date", "")) == target_date_iso
                            and r.get("predicted_high") is not None
                            and (r.get("source") or "").lower() != "accuweather"
                        ], key=lambda x: x[0])
                        if _d0_fc:
                            _ex_snap["nws_overnight_jump"] = round(
                                _d0_fc[0][1] - float(_d1f), 1
                            )
                except Exception:
                    pass
                # Fetch fresh Synoptic / NYSM data directly — on stable days
                # _compute_ml_prediction() never runs so live_atm has no obs keys.
                # IMPORTANT: write obs_snap_* display keys directly (NOT obs_synoptic_*
                # ML feature keys) — the dashboard reads obs_snap_syn_mean etc., not
                # obs_synoptic_mean. _add_obs_to_snap is NOT called here because
                # live_obs is empty on frozen paths and would clobber KNYC obs keys.
                _nws_for_obs = float(nws_latest) if nws_latest is not None else None

                def _safe_snap(v):
                    if v is None: return None
                    try:
                        f = float(v)
                        return None if math.isnan(f) else round(f, 3)
                    except (TypeError, ValueError): return None

                try:
                    from synoptic_client import get_synoptic_obs_features
                    import nws_auto_logger as _nal_snap
                    _snap_cfg = _nal_snap._CITY_CFG
                    # Use city-specific synoptic coords if provided (LAX uses KCQT downtown
                    # rather than KLAX airport — KCQT is the Kalshi reference point and
                    # has far better station density). Falls back to open_meteo coords.
                    _syn_lat = _snap_cfg.get("synoptic_lat", _snap_cfg.get("open_meteo_lat", 40.7834))
                    _syn_lon = _snap_cfg.get("synoptic_lon", _snap_cfg.get("open_meteo_lon", -73.965))
                    _syn_radius = _snap_cfg.get("synoptic_radius_miles", 10.0)
                    _syn_fresh = get_synoptic_obs_features(
                        lat=_syn_lat,
                        lon=_syn_lon,
                        nws_last=_nws_for_obs,
                        radius_miles=_syn_radius,
                        city=_CITY_KEY,
                    )
                    # ── CRITICAL: Only overwrite if we got actual data ──
                    # When Synoptic API fails (403/rate-limit), it returns all NaN.
                    # _safe_snap(NaN) → None. If we unconditionally write None,
                    # we DESTROY existing valid data every 10-min cycle.
                    # Fix: Only update fields that have real (non-None) values.
                    _syn_pop = 0
                    for _snap_key, _fresh_key in [
                        ("obs_snap_syn_mean",   "obs_synoptic_mean"),
                        ("obs_snap_syn_min",    "obs_synoptic_min"),
                        ("obs_snap_syn_max",    "obs_synoptic_max"),
                        ("obs_snap_syn_spread", "obs_synoptic_spread"),
                        ("obs_snap_syn_vs_nws", "obs_synoptic_vs_nws"),
                    ]:
                        _v = _safe_snap(_syn_fresh.get(_fresh_key))
                        if _v is not None:
                            _ex_snap[_snap_key] = _v
                            _syn_pop += 1
                    _sc = _syn_fresh.get("obs_synoptic_count")
                    if _sc is not None and not (isinstance(_sc, float) and math.isnan(_sc)):
                        _ex_snap["obs_snap_syn_count"] = _sc
                        _syn_pop += 1
                    if _syn_pop == 0:
                        print(f"  ⚠️ Synoptic API returned no data — preserving existing snapshot values")

                    # v9: named station readings — city-specific
                    # Same guard: only overwrite if fresh data is non-None
                    if _CITY_KEY == "lax":
                        # LAX regional stations
                        for _snap_key, _fresh_key in [
                            ("obs_snap_lax", "obs_lax_temp"),
                            ("obs_snap_smo", "obs_smo_temp"),
                            ("obs_snap_bur", "obs_bur_temp"),
                            ("obs_snap_vny", "obs_vny_temp"),
                            ("obs_snap_cqt", "obs_cqt_temp"),
                            ("obs_snap_bur_vs_lax", "obs_bur_vs_lax"),
                            ("obs_snap_coastal_vs_inland_lax", "obs_coastal_vs_inland_lax"),
                            ("obs_snap_airport_spread_lax", "obs_airport_spread_lax"),
                        ]:
                            _v = _safe_snap(_syn_fresh.get(_fresh_key))
                            if _v is not None:
                                _ex_snap[_snap_key] = _v
                        # LAX observation timestamps (only overwrite if string)
                        for _sk, _fk in [
                            ("obs_snap_lax_obs_at", "obs_lax_obs_at"),
                            ("obs_snap_smo_obs_at", "obs_smo_obs_at"),
                            ("obs_snap_bur_obs_at", "obs_bur_obs_at"),
                            ("obs_snap_vny_obs_at", "obs_vny_obs_at"),
                            ("obs_snap_cqt_obs_at", "obs_cqt_obs_at"),
                        ]:
                            _ts = _syn_fresh.get(_fk)
                            if isinstance(_ts, str):
                                _ex_snap[_sk] = _ts
                        print(f"  📡 Stable-cycle Synoptic LAX: "
                              f"{_ex_snap.get('obs_snap_syn_mean')} "
                              f"({_ex_snap.get('obs_snap_syn_count')} stations)  "
                              f"KLAX={_ex_snap.get('obs_snap_lax')}  "
                              f"KBUR-KLAX={_ex_snap.get('obs_snap_bur_vs_lax')}  "
                              f"KCQT={_ex_snap.get('obs_snap_cqt')}")
                    else:
                        # NYC regional airports + Manhattan Mesonet
                        for _snap_key, _fresh_key in [
                            ("obs_snap_kjfk", "obs_kjfk_temp"),
                            ("obs_snap_klga", "obs_klga_temp"),
                            ("obs_snap_kewr", "obs_kewr_temp"),
                            ("obs_snap_kteb", "obs_kteb_temp"),
                            ("obs_snap_knyc_syn", "obs_knyc_temp"),
                            ("obs_snap_kjfk_vs_knyc", "obs_kjfk_vs_knyc"),
                            ("obs_snap_klga_vs_knyc", "obs_klga_vs_knyc"),
                            ("obs_snap_kewr_vs_knyc", "obs_kewr_vs_knyc"),
                            ("obs_snap_airport_spread", "obs_airport_spread"),
                            ("obs_snap_coastal_vs_inland", "obs_coastal_vs_inland"),
                            ("obs_snap_manh_temp", "obs_manh_temp"),
                            ("obs_snap_manh_vs_knyc", "obs_manh_vs_knyc"),
                        ]:
                            _v = _safe_snap(_syn_fresh.get(_fresh_key))
                            if _v is not None:
                                _ex_snap[_snap_key] = _v
                        # NYC observation timestamps (only overwrite if string)
                        for _sk, _fk in [
                            ("obs_snap_knyc_obs_at", "obs_knyc_obs_at"),
                            ("obs_snap_kjfk_obs_at", "obs_kjfk_obs_at"),
                            ("obs_snap_klga_obs_at", "obs_klga_obs_at"),
                            ("obs_snap_kewr_obs_at", "obs_kewr_obs_at"),
                            ("obs_snap_kteb_obs_at", "obs_kteb_obs_at"),
                            ("obs_snap_manh_obs_at", "obs_manh_obs_at"),
                        ]:
                            _ts = _syn_fresh.get(_fk)
                            if isinstance(_ts, str):
                                _ex_snap[_sk] = _ts
                        print(f"  📡 Stable-cycle Synoptic NYC: "
                              f"{_ex_snap.get('obs_snap_syn_mean')} "
                              f"({_ex_snap.get('obs_snap_syn_count')} stations)  "
                              f"KJFK={_ex_snap.get('obs_snap_kjfk')}  "
                              f"KJFK-KNYC={_ex_snap.get('obs_snap_kjfk_vs_knyc')}  "
                              f"MANH={_ex_snap.get('obs_snap_manh_temp')}")
                except Exception as _syn_e:
                    print(f"  ⚠️  Stable-cycle Synoptic skipped: {_syn_e}")

                # MANH (Manhattan Mesonet via Synoptic) — re-enabled: enterprise
                # Synoptic access includes NYSM STIDs.  Primary path: MANH shows up
                # in the 10-mile radius sweep (~2.5mi from Central Park) and
                # obs_snap_manh_temp is populated above from _syn_fresh.get("obs_manh_temp").
                # Fallback: if MANH was still null (e.g. station briefly offline, radius
                # sweep missed it), do a targeted direct STID fetch for just MANH.
                if _CITY_KEY != "lax" and _ex_snap.get("obs_snap_manh_temp") is None:
                    try:
                        from synoptic_client import _fetch_stids_direct
                        _manh_stns = _fetch_stids_direct(["MANH"])
                        if _manh_stns:
                            _m_obs = _manh_stns[0].get("OBSERVATIONS", {})
                            _m_tv  = _m_obs.get("air_temp_value_1", {})
                            _m_t   = _m_tv.get("value") if isinstance(_m_tv, dict) else _m_tv
                            _m_at  = _m_tv.get("date_time") if isinstance(_m_tv, dict) else None
                            if _m_t is not None:
                                try:
                                    _ex_snap["obs_snap_manh_temp"] = round(float(_m_t), 1)
                                    if _m_at:
                                        _ex_snap["obs_snap_manh_obs_at"] = _m_at
                                    print(f"  🏙️ MANH (direct fallback): {_ex_snap['obs_snap_manh_temp']}°F")
                                except (TypeError, ValueError):
                                    pass
                    except Exception as _nysm_e:
                        print(f"  ⚠️ MANH direct fallback skipped: {_nysm_e}")

                # Secondary fallback: if MANH still NULL, try NY State Mesonet API directly
                if _CITY_KEY != "lax" and _ex_snap.get("obs_snap_manh_temp") is None:
                    try:
                        from nysmesonet_client import get_nysm_obs_features
                        _nysm_feats = get_nysm_obs_features(nws_last=_nws_for_obs)
                        _manh_t = _nysm_feats.get("obs_nysm_manh")
                        if _manh_t is not None and not (isinstance(_manh_t, float) and math.isnan(_manh_t)):
                            _ex_snap["obs_snap_manh_temp"] = round(float(_manh_t), 1)
                            # Calculate MANH vs KNYC if we have KNYC
                            _knyc_t = _ex_snap.get("obs_snap_knyc_temp")
                            if _knyc_t is not None:
                                _ex_snap["obs_snap_manh_vs_knyc"] = round(float(_manh_t) - float(_knyc_t), 1)
                            print(f"  🏙️ MANH (NYSM fallback): {_ex_snap.get('obs_snap_manh_temp')}°F")
                    except Exception as _nysm_fallback_e:
                        print(f"  ⚠️ MANH NYSM fallback skipped: {_nysm_fallback_e}")

                # ── Refresh KNYC / obs display keys for dashboard ─────────────
                # agency_cutoff silences live_obs for ML trigger purposes (correct —
                # avoids thermometer-chasing), but the dashboard still needs fresh
                # obs_snap_max_so_far, obs_snap_is_overnight_high, obs_snap_temp,
                # obs_snap_jfk, obs_snap_lga etc. so KNYC Max and the overnight
                # note don't go stale after 2 PM.  Fetch obs purely for display.
                try:
                    _obs_disp = _fetch_observation_features(
                        target_date_iso,
                        nws_last=_nws_for_obs,
                        atm_features=live_atm,
                    )
                    _OBS_DISPLAY_KEYS = (
                        # ── Primary KNYC observation ──────────────────────────
                        "obs_snap_temp", "obs_snap_max_so_far",
                        "obs_snap_is_overnight_high", "obs_snap_high_peak_hour",
                        "obs_snap_temp_falling_hrs", "obs_snap_heating_rate",
                        "obs_snap_heating_rate_delta",   # stall signal — was missing, went stale
                        "obs_snap_vs_forecast", "obs_snap_hour",
                        "obs_snap_wind_speed", "obs_snap_cloud_cover",
                        # ── Regional NWS stations (JFK / LGA) ────────────────
                        "obs_snap_jfk", "obs_snap_lga",
                        "obs_snap_regional_spread", "obs_snap_regional_mean",
                        "obs_snap_regional_vs_nws",
                        # ── Named ASOS stations (v9 Synoptic) ────────────────
                        # These were missing — caused KEWR/KTEB/marine card to go
                        # stale and show "Awaiting next cycle" after 2 PM.
                        "obs_snap_kjfk", "obs_snap_klga",
                        "obs_snap_kewr", "obs_snap_kteb", "obs_snap_knyc_syn",
                        # KCDW (Caldwell NJ, ~25mi) + KSMQ (Somerville NJ, ~35mi) — deeper inland
                        "obs_snap_kcdw", "obs_snap_ksmq",
                        "obs_snap_kjfk_vs_knyc", "obs_snap_klga_vs_knyc",
                        "obs_snap_kewr_vs_knyc",
                        "obs_snap_airport_spread", "obs_snap_coastal_vs_inland",
                        "obs_snap_inland_gradient",   # JFK→SMQ full spread
                        # ── Named station observation timestamps ──────────────
                        # Staleness badges (e.g. "47 min ago") depend on these;
                        # without them the badge freezes at the canonical write time.
                        "obs_snap_knyc_obs_at", "obs_snap_kjfk_obs_at",
                        "obs_snap_klga_obs_at", "obs_snap_kewr_obs_at",
                        "obs_snap_kteb_obs_at", "obs_snap_kcdw_obs_at",
                        "obs_snap_ksmq_obs_at", "obs_snap_manh_obs_at",
                        # ── Manhattan Mesonet (5-min fill-in) ────────────────
                        "obs_snap_manh_temp", "obs_snap_manh_vs_knyc",
                        # ── Synoptic network (5mi radius) ────────────────────
                        "obs_snap_syn_mean", "obs_snap_syn_min", "obs_snap_syn_max",
                        "obs_snap_syn_spread", "obs_snap_syn_vs_nws", "obs_snap_syn_count",
                        # ── WU PWS (citizen weather stations) ────────────────
                        "obs_snap_wu_mean", "obs_snap_wu_vs_nws",
                        "obs_snap_wu_spread", "obs_snap_wu_count",
                        # ── Derived nowcast signals ───────────────────────────
                        # These were missing — stratus clearing and inland warming
                        # signals froze after 2pm just when they're most relevant.
                        "obs_snap_morning_cloud",
                        "obs_snap_stratus_clearing",
                        "obs_snap_kteb_delta", "obs_snap_kewr_delta",
                        "obs_snap_kcdw_delta", "obs_snap_ksmq_delta",
                        "obs_snap_inland_warming_rate", "obs_snap_warming_accel",
                        # ── Metadata ─────────────────────────────────────────
                        "obs_snap_populated",
                    )
                    # _fetch_observation_features returns raw feature keys (obs_*),
                    # not the snap keys (obs_snap_*).  Map via _add_obs_to_snap
                    # into a temp dict, then cherry-pick display keys.
                    _tmp_snap: dict = {}
                    _add_obs_to_snap(_tmp_snap, _obs_disp, live_atm)
                    for _dk in _OBS_DISPLAY_KEYS:
                        if _dk in _tmp_snap and _tmp_snap[_dk] is not None:
                            _ex_snap[_dk] = _tmp_snap[_dk]
                    print(f"  🌡️ Stable-cycle obs refresh: "
                          f"temp={_ex_snap.get('obs_snap_temp')} "
                          f"max={_ex_snap.get('obs_snap_max_so_far')} "
                          f"overnight_flag={_ex_snap.get('obs_snap_is_overnight_high')}")
                except Exception as _obs_disp_e:
                    print(f"  ⚠️  Stable-cycle obs display refresh skipped: {_obs_disp_e}")

                # ── Derived signals: stratus clearing + inland NJ warming ────────
                # These use the current _ex_snap values (freshly merged above) vs
                # stored_snapshot_dict (the previous snapshot from Supabase) to
                # detect intraday changes the community nowcasters see before NWS.
                try:
                    # Preserve morning cloud anchor from stored snapshot if not yet
                    # in _ex_snap (canonical write may have been on a different run)
                    _stored_mc = stored_snapshot_dict.get("obs_snap_morning_cloud")
                    if _stored_mc is not None and _ex_snap.get("obs_snap_morning_cloud") is None:
                        _ex_snap["obs_snap_morning_cloud"] = _stored_mc

                    # ── Stratus clearing: cap dissolving mid-afternoon ─────────
                    # If cloud cover was ≥50% at market open but is now ≤25%,
                    # the stratus is clearing → late-session warmth risk.
                    _mc  = _ex_snap.get("obs_snap_morning_cloud")
                    _cc  = _ex_snap.get("obs_snap_cloud_cover")
                    _hr  = _ex_snap.get("obs_snap_hour")
                    if _mc is not None and _cc is not None and _hr is not None:
                        _ex_snap["obs_snap_stratus_clearing"] = bool(
                            float(_mc) >= 50
                            and float(_cc) <= 25
                            and 11 <= float(_hr) <= 19
                        )
                        if _ex_snap["obs_snap_stratus_clearing"]:
                            print(f"  🌤️  Stratus clearing detected: "
                                  f"morning {_mc}% → current {_cc}% at hour {_hr}")

                    # ── Inter-run NNJ inland delta (4-station warming rate) ──────
                    # Compare current named-station temps against the PREVIOUS
                    # snapshot to detect rapid NNJ surface warming — the early
                    # warm-advection signal that nowcasters see before NWS updates.
                    # KEWR: Newark NJ (~15mi), KTEB: Teterboro (~20mi),
                    # KCDW: Caldwell (~25mi), KSMQ: Somerville (~35mi — deepest probe)
                    if _CITY_KEY != "lax":
                        _prev_kteb = stored_snapshot_dict.get("obs_snap_kteb")
                        _curr_kteb = _ex_snap.get("obs_snap_kteb")
                        _kteb_d: float | None = None
                        if _prev_kteb is not None and _curr_kteb is not None:
                            _kteb_d = round(float(_curr_kteb) - float(_prev_kteb), 1)
                            _ex_snap["obs_snap_kteb_delta"] = _kteb_d
                        else:
                            _kteb_d = _ex_snap.get("obs_snap_kteb_delta")

                        _prev_kewr = stored_snapshot_dict.get("obs_snap_kewr")
                        _curr_kewr = _ex_snap.get("obs_snap_kewr")
                        _kewr_d: float | None = None
                        if _prev_kewr is not None and _curr_kewr is not None:
                            _kewr_d = round(float(_curr_kewr) - float(_prev_kewr), 1)
                            _ex_snap["obs_snap_kewr_delta"] = _kewr_d
                        else:
                            _kewr_d = _ex_snap.get("obs_snap_kewr_delta")

                        _prev_kcdw = stored_snapshot_dict.get("obs_snap_kcdw")
                        _curr_kcdw = _ex_snap.get("obs_snap_kcdw")
                        _kcdw_d: float | None = None
                        if _prev_kcdw is not None and _curr_kcdw is not None:
                            _kcdw_d = round(float(_curr_kcdw) - float(_prev_kcdw), 1)
                            _ex_snap["obs_snap_kcdw_delta"] = _kcdw_d
                        else:
                            _kcdw_d = _ex_snap.get("obs_snap_kcdw_delta")

                        _prev_ksmq = stored_snapshot_dict.get("obs_snap_ksmq")
                        _curr_ksmq = _ex_snap.get("obs_snap_ksmq")
                        _ksmq_d: float | None = None
                        if _prev_ksmq is not None and _curr_ksmq is not None:
                            _ksmq_d = round(float(_curr_ksmq) - float(_prev_ksmq), 1)
                            _ex_snap["obs_snap_ksmq_delta"] = _ksmq_d
                        else:
                            _ksmq_d = _ex_snap.get("obs_snap_ksmq_delta")

                        # Inland NNJ composite warming rate (avg across all available stations)
                        _inland_vals = [v for v in [_kteb_d, _kewr_d, _kcdw_d, _ksmq_d] if v is not None]
                        if _inland_vals:
                            _inland_rate = round(sum(_inland_vals) / len(_inland_vals), 1)
                            _ex_snap["obs_snap_inland_warming_rate"] = _inland_rate
                            # Warming acceleration: rapid NNJ warming + positive surface rate
                            _surf_r = _ex_snap.get("obs_snap_heating_rate")
                            _ex_snap["obs_snap_warming_accel"] = bool(
                                _inland_rate >= 1.5
                                and _surf_r is not None
                                and float(_surf_r) >= 0.3
                            )
                            if _inland_rate >= 1.5:
                                print(f"  🔥 Inland NNJ warming: TEB Δ={_kteb_d} "
                                      f"EWR Δ={_kewr_d} CDW Δ={_kcdw_d} SMQ Δ={_ksmq_d} "
                                      f"composite={_inland_rate}°F/cycle "
                                      f"accel={_ex_snap.get('obs_snap_warming_accel')}")
                except Exception as _sig_e:
                    print(f"  ⚠️  Derived signals skipped: {_sig_e}")

                # ── Cache complete snapshot for fallback on next API failures ──
                cache_snapshot(_CITY_KEY, _ex_snap)

                payload["atm_snapshot"] = json.dumps(_ex_snap)
                atm_keys = sum(1 for k in _ex_snap.keys() if k.startswith("atm_") or k.startswith("mm_") or k.startswith("ens_"))
                obs_keys = sum(1 for k in _ex_snap.keys() if k.startswith("obs_snap_"))
                valid_keys = sum(1 for v in _ex_snap.values() if v is not None and not (isinstance(v, float) and math.isnan(v)))
                print(f"📸 Obs panel refreshed in stable snapshot "
                      f"({len(_ex_snap)} total keys, {atm_keys} atm/mm/ens, {obs_keys} obs_snap, {valid_keys} valid values)")
        except Exception as _e:
            print(f"⚠️  Obs panel refresh skipped: {_e}")

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
        payload["is_canonical"] = is_canonical_write
        if is_canonical_write:
            payload["ml_bucket_canonical"] = ml["ml_bucket"]
            payload["ml_f_canonical"] = ml["ml_f"]
            print(f"🏛️ Canonical prediction set: {ml['ml_f']}°F → {ml['ml_bucket']}")
        else:
            # ── CRITICAL: Preserve canonical fields on intraday writes ──
            # Supabase merge-duplicates does wholesale row replacement.
            # If we don't include ml_f_canonical / ml_bucket_canonical in this upsert,
            # they'll be overwritten with NULL even though they were set on the
            # first (canonical) write. Carry them forward to prevent data loss.
            # (Only on intraday refresh, when is_canonical_write=False)
            if isinstance(existing, dict):
                if existing.get("ml_f_canonical") is not None:
                    payload["ml_f_canonical"] = existing["ml_f_canonical"]
                if existing.get("ml_bucket_canonical") is not None:
                    payload["ml_bucket_canonical"] = existing["ml_bucket_canonical"]
            _log_ml_revision(target_date_iso, "today_for_today", ml, "first_write")
            # Store morning atmospheric + obs baseline on canonical write.
            # On subsequent runs, live_atm and live_obs are compared against this
            # snapshot to detect intraday shifts BEFORE agencies update forecasts.
            # CRITICAL: Always write snapshot, even if live_atm fetch failed.
            # An empty snapshot (just obs_snap_* keys) is better than NULL which
            # breaks dashboard rendering and stable-cycle recovery.
            snap = {}
            if live_atm and any(
                v is not None and not (isinstance(v, float) and math.isnan(v))
                for v in live_atm.values()
            ):
                # Populate atm_* / mm_* / ens_* / nws_* keys from live_atm
                snap = {k: live_atm[k] for k in _ATM_SNAPSHOT_KEYS if k in live_atm}
            # Fill any missing atm_* / mm_* keys from cache to ensure Multi-Model Spread / HRRR vs NWS
            # cards always have data even if Open-Meteo ensemble or Synoptic fetch failed.
            snap = fill_missing_from_cache(_CITY_KEY, snap)
            # Always write snapshot on canonical (first-write) path, even if empty.
            # An empty snapshot dict still allows _add_obs_to_snap to populate obs_snap_*
            # display keys, which is essential for the "Live Observation Sources" panel.
            # NULL snapshot breaks dashboard panel rendering and makes the cards disappear.
            if True:  # Always run on canonical write to ensure snapshot is always persisted
                    # Inject NWS sequence features — computed inside _compute_ml_prediction()
                    # but not returned via live_atm (which is Open-Meteo only).
                    _inject_nws_sequence_to_snap(snap, nws_latest, target_date_iso, rows)
                    # Add obs snapshot keys — critical for cold-start trigger detection
                    # Pass live_atm so Synoptic/NYSM written back by _compute_ml_prediction
                    # via prefetched_atm are included (they're not in live_obs).
                    # Pass city_key so if live_obs is empty, cached obs are used as fallback.
                    _add_obs_to_snap(snap, live_obs, live_atm, city_key=_CITY_KEY)
                    # ── obs_snap_morning_cloud anchor ────────────────────────────────
                    # Freeze cloud cover at canonical (first-of-day) write so that
                    # intraday stratus clearing can be detected: if morning cloud ≥50%
                    # and afternoon cloud ≤25% the cap is dissolving — warm-side risk.
                    _cld_at_open = snap.get("obs_snap_cloud_cover")
                    if _cld_at_open is not None:
                        snap["obs_snap_morning_cloud"] = _cld_at_open
                        print(f"  ☁️  Morning cloud anchor: {_cld_at_open}%")
                    # ── mm_morning_* daily anchor ──────────────────────────────────────
                    # Written once here at canonical (first-of-day) write and NEVER
                    # overwritten on subsequent intraday upserts.  Serves as the
                    # ultimate fallback in the carry-forward block (~40 lines above)
                    # when consecutive Open-Meteo ensemble failures have depleted BOTH
                    # live_atm AND the stored_snapshot_dict of all mm_* values.
                    #
                    # Naming: mm_spread → mm_morning_spread
                    #         atm_925mb_hrrr_max → mm_morning_atm_925mb_hrrr_max
                    # (all under the mm_morning_ namespace so the restoration loop
                    # `_pk.startswith("mm_")` carries them through non-canonical writes)
                    _morning_anchored = []
                    for _ck in _MM_CARRY_KEYS:
                        _v = snap.get(_ck)
                        if _v is not None and not (isinstance(_v, float) and math.isnan(_v)):
                            snap[_morning_anchor_key(_ck)] = _v
                            _morning_anchored.append(_ck)
                    if _morning_anchored:
                        print(f"  🌅 Morning anchor written: {len(_morning_anchored)} keys "
                              f"(mm_morning_spread={snap.get('mm_morning_spread')}, "
                              f"mm_morning_hrrr_max={snap.get('mm_morning_hrrr_max')}, "
                              f"mm_morning_nbm_max={snap.get('mm_morning_nbm_max')})")
                    else:
                        print(f"  ⚠️  Morning anchor: no mm_* values available at canonical "
                              f"write — cascading-null fallback unavailable for today")

                    # ── Carry flip fields before overwriting snapshot (canonical path) ──
                    # On the canonical write (first-of-day) there are no prior flip
                    # fields to carry — but on a re-canonical (edge case: D1 row being
                    # claimed as D0) there could be.  Safe to always call.
                    if isinstance(existing, dict):
                        _carry_flip_fields(snap, existing)

                    # ── Cache complete snapshot for fallback on next API failures ──
                    cache_snapshot(_CITY_KEY, snap)

                    payload["atm_snapshot"] = json.dumps(snap)
                    obs_populated = sum(
                        1 for v in (live_obs or {}).values()
                        if v is not None and not (isinstance(v, float) and math.isnan(v))
                    )
                    atm_keys = sum(1 for k in snap.keys() if k.startswith("atm_") or k.startswith("mm_") or k.startswith("ens_"))
                    obs_keys = sum(1 for k in snap.keys() if k.startswith("obs_snap_"))
                    print(f"📸 Baseline stored: {len(snap)} total keys ({atm_keys} atm/mm/ens, {obs_keys} obs_snap), "
                          f"live_atm_had_data={bool(live_atm and any(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in live_atm.values()))}, "
                          f"obs_hour={snap.get('obs_snap_hour')}, "
                          f"obs_temp={snap.get('obs_snap_temp')}, "
                          f"obs_populated={obs_populated}")

    # Fetch Kalshi market odds + compute bet signal
    market_probs = _fetch_kalshi_market_probs(target_date_iso)
    # IMPORTANT: Only save the snapshot at canonical (first) write time.
    # Subsequent 30-min upserts must NOT overwrite it — by end-of-day Kalshi
    # shows settled prices (winner=0.995, losers=0.005) which completely inverts
    # the edge calculation. The canonical snapshot captures live pre-settlement
    # market prices, which is what the bet_signal and ml_edge should reflect.
    if market_probs and is_canonical_write:
        payload["kalshi_market_snapshot"] = json.dumps(market_probs)

    # Detect intraday bucket shift for stability downgrade.
    # If the stored bucket differs from what we're about to write, the model just
    # changed its mind — don't call it STRONG BET until it holds for another cycle.
    _prev_bucket = existing.get("ml_bucket") if isinstance(existing, dict) else None
    _bucket_just_changed = (
        not is_canonical_write        # canonical = first write, no prior bucket
        and _prev_bucket is not None
        and ml is not None
        and ml.get("ml_bucket") is not None
        and _prev_bucket != ml["ml_bucket"]
    )
    if _bucket_just_changed:
        print(f"  🔀 Bucket shifted: {_prev_bucket} → {ml['ml_bucket']} "
              f"(bet signal will be downgraded this cycle for stability)")

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

                # Also re-check bucket change against the final Kalshi-aligned bucket
                _bucket_just_changed = _bucket_just_changed or (
                    not is_canonical_write
                    and _prev_bucket is not None
                    and _prev_bucket != kalshi_bucket
                )

                # Store second-best Kalshi bucket for later outcome tracking
                sorted_aligned = sorted(kalshi_aligned.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_aligned) > 1 and sorted_aligned[1][1] >= 0.02:
                    payload["ml_bucket_2"] = sorted_aligned[1][0]
                    payload["ml_bucket_2_prob"] = round(sorted_aligned[1][1], 4)
                    print(f"🥈 Bucket 2: {sorted_aligned[1][0]} ({sorted_aligned[1][1]:.0%})")

                if direct_bucket and kalshi_bucket != direct_bucket:
                    print(f"ℹ️ Direct map ({direct_bucket}) differs from prob-aligned ({kalshi_bucket}) "
                          f"— keeping prob-aligned (higher expected accuracy)")

                signal, edge, kelly_f = _compute_bet_signal(
                    payload["ml_confidence"], payload["ml_bucket"], market_probs,
                    bucket_just_changed=_bucket_just_changed,
                )
                payload["bet_signal"] = signal
                payload["ml_edge"] = edge
                print(f"🎯 Bet signal: {signal} (edge={edge:+.0%}, kelly={kelly_f:.1%})")
                if is_canonical_write:
                    canonical_bucket = payload.get("ml_bucket", "")
                    payload["market_prob_at_prediction"] = round(market_probs.get(canonical_bucket, 0.0), 4)
                    payload["kelly_fraction"] = kelly_f
                    print(f"📌 market_prob_at_prediction={payload['market_prob_at_prediction']:.4f} for bucket '{canonical_bucket}'")
                    print(f"📐 kelly_fraction={kelly_f:.1%} (half-Kelly — bet this % of bankroll)")
        elif direct_bucket:
            payload["ml_bucket"] = direct_bucket
            _bucket_just_changed = _bucket_just_changed or (
                not is_canonical_write
                and _prev_bucket is not None
                and _prev_bucket != direct_bucket
            )
            signal, edge, kelly_f = _compute_bet_signal(
                ml["ml_confidence"], direct_bucket, market_probs,
                bucket_just_changed=_bucket_just_changed,
            )
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%}, kelly={kelly_f:.1%})")
            if is_canonical_write:
                payload["market_prob_at_prediction"] = round(market_probs.get(direct_bucket, 0.0), 4)
                payload["kelly_fraction"] = kelly_f
                print(f"📌 market_prob_at_prediction={payload['market_prob_at_prediction']:.4f} for bucket '{direct_bucket}'")
                print(f"📐 kelly_fraction={kelly_f:.1%} (half-Kelly — bet this % of bankroll)")

    # ── Inject flip-history tracking into atm_snapshot (no schema change) ───
    # Persists ml_flip_occurred=True even when the bucket RETURNS to canonical.
    # Without this the dashboard shows no evidence a flip ever happened.
    #
    # CONFIRMATION GATE: a non-canonical bucket must appear in TWO consecutive
    # runs before it is recorded as a confirmed flip.  This suppresses single-run
    # noise flashes (e.g. 76-77 from a 2-min Synoptic outage) without delaying
    # genuine sustained shifts.  During the 2-min heating-window polling cycle
    # the gate adds at most a 2-minute lag to real signals.
    #
    # State machine — stored in atm_snapshot JSONB:
    #   ml_flip_pending_bucket  str  — non-canonical bucket seen last run but not
    #                                  yet confirmed (cleared on canonical return or
    #                                  when a different bucket appears next run)
    #   ml_flip_occurred        bool — True only after confirmation (≥2 consecutive
    #                                  runs showing the same non-canonical bucket)
    #   ml_flip_count           int  — number of CONFIRMED flips away from canonical
    #   ml_flip_bucket          str  — most recent CONFIRMED non-canonical bucket
    #   ml_flip_history         list — every CONFIRMED non-canonical bucket in order
    #                                  e.g. ["78-79"] after 76-77 pending was discarded
    #
    # Transition rules:
    #   curr ≠ canon + prev_pending == curr  → CONFIRM  (set occurred, append history)
    #   curr ≠ canon + prev_pending ≠ curr   → PENDING  (store pending, hold occurred)
    #   curr == canon + had_pending          → DISCARD  (clear pending silently)
    #   curr == canon + had_confirmed        → PRESERVE (keep history intact)
    if not is_canonical_write and isinstance(existing, dict):
        _canon_bkt = existing.get("ml_bucket_canonical")
        _curr_bkt  = payload.get("ml_bucket")
        if not _canon_bkt:
            print(f"  ⚠️  Flip tracking skipped: ml_bucket_canonical not set yet "
                  f"(canonical write hasn't fired — likely first run of day)")
        elif not _curr_bkt:
            print(f"  ⚠️  Flip tracking skipped: ml_bucket absent from payload "
                  f"(ml may be None this cycle — payload keys: {list(payload.keys())[:8]})")
        if _canon_bkt and _curr_bkt:
            # Load whichever snapshot we're writing, or the existing one as fallback
            _fs_raw = payload.get("atm_snapshot") or existing.get("atm_snapshot")
            try:
                _fs = (json.loads(_fs_raw) if isinstance(_fs_raw, str)
                       else (_fs_raw.copy() if isinstance(_fs_raw, dict) else {})) \
                       if _fs_raw else {}
            except Exception:
                _fs = {}
            # Read prior flip state from the existing (pre-write) snapshot
            _pfs_raw = existing.get("atm_snapshot")
            try:
                _pfs = (json.loads(_pfs_raw) if isinstance(_pfs_raw, str)
                        else (_pfs_raw if isinstance(_pfs_raw, dict) else {})) \
                        if _pfs_raw else {}
            except Exception:
                _pfs = {}
            _prev_flip_count  = int(_pfs.get("ml_flip_count") or 0)
            _prev_pending     = _pfs.get("ml_flip_pending_bucket")   # unconfirmed last run
            _prev_confirmed   = bool(_pfs.get("ml_flip_occurred"))   # ever confirmed?
            # Recover existing confirmed history list
            _prev_history = _pfs.get("ml_flip_history") or []
            if isinstance(_prev_history, str):
                try:    _prev_history = json.loads(_prev_history)
                except: _prev_history = []
            if not isinstance(_prev_history, list):
                _prev_history = []

            if _curr_bkt != _canon_bkt:
                if _prev_pending == _curr_bkt:
                    # ── CONFIRMED: same non-canonical bucket seen twice in a row ──
                    _fs["ml_flip_occurred"]       = True
                    _fs["ml_flip_count"]          = _prev_flip_count + 1
                    _fs["ml_flip_bucket"]         = _curr_bkt
                    _fs["ml_flip_pending_bucket"] = None   # clear — no longer pending
                    # Append to confirmed history; deduplicate tail
                    if not _prev_history or _prev_history[-1] != _curr_bkt:
                        _fs["ml_flip_history"] = _prev_history + [_curr_bkt]
                    else:
                        _fs["ml_flip_history"] = _prev_history
                    _chain = " → ".join([_canon_bkt] + _fs["ml_flip_history"])
                    print(f"  ✅ Flip #{_fs['ml_flip_count']} CONFIRMED (2 consecutive runs): {_chain}")
                else:
                    # ── PENDING: first time seeing this non-canonical bucket ──
                    # Do NOT set ml_flip_occurred — gate not yet passed.
                    # Preserve any prior confirmed flip state so history isn't lost.
                    _fs["ml_flip_pending_bucket"] = _curr_bkt
                    if _prev_confirmed:
                        _fs["ml_flip_occurred"] = True
                        _fs["ml_flip_count"]    = _prev_flip_count
                        _fs["ml_flip_bucket"]   = _pfs.get("ml_flip_bucket", _curr_bkt)
                        _fs["ml_flip_history"]  = _prev_history
                    print(f"  ⏳ Flip PENDING ({_canon_bkt} → {_curr_bkt}) — "
                          f"awaiting one more run to confirm (gate: 2 consecutive)")
            else:
                # curr_bkt == canon_bkt: back at canonical
                _fs["ml_flip_pending_bucket"] = None   # always clear pending on return
                if _prev_confirmed:
                    # Preserve confirmed history
                    _fs["ml_flip_occurred"] = True
                    _fs["ml_flip_count"]    = _prev_flip_count
                    _fs["ml_flip_bucket"]   = _pfs.get("ml_flip_bucket", _curr_bkt)
                    _fs["ml_flip_history"]  = _prev_history
                    _chain = " → ".join([_canon_bkt] + _prev_history + ["back"])
                    print(f"  ↩️  Returned to canonical — confirmed flip record preserved: {_chain}")
                elif _prev_pending:
                    # Was only pending — discard silently (never made it to the display)
                    print(f"  🗑️  Pending flip ({_prev_pending}) discarded — "
                          f"returned to canonical before confirmation, nothing shown")
            if _fs:
                payload["atm_snapshot"] = json.dumps(_fs)

    # ── Fallback: ensure atm_snapshot is always written ──────────────────────
    # If we've gotten this far without setting payload["atm_snapshot"], it means:
    #   - ML didn't recompute (ml_recomputed=False)
    #   - AND obs-panel refresh didn't run (either not a stable cycle OR no existing row)
    #   - AND flip tracking didn't set it
    # This happens on the very first prediction write of a new day before ML runs.
    # Fetch atmospheric + observational features so Live Observation Sources has data.
    if "atm_snapshot" not in payload:
        try:
            fallback_atm = _fetch_atmospheric_features(target_date_iso)
            # Also fetch live observations (KNYC, regional, Synoptic, MANH, etc.)
            fallback_obs = _fetch_observation_features(target_date_iso) or {}

            # Build snapshot with both ATM and OBS data
            fallback_snap = fallback_atm or {}
            if fallback_obs:
                _add_obs_to_snap(fallback_snap, fallback_obs, fallback_atm)

            if fallback_snap and any(
                v is not None and not (isinstance(v, float) and math.isnan(v))
                for v in fallback_snap.values()
            ):
                # Cache for future fallback
                cache_snapshot(_CITY_KEY, fallback_snap)
                payload["atm_snapshot"] = json.dumps(fallback_snap)
                atm_keys = sum(1 for k in fallback_snap.keys() if k.startswith("atm_") or k.startswith("mm_"))
                obs_keys = sum(1 for k in fallback_snap.keys() if k.startswith("obs_snap_"))
                valid_keys = sum(1 for v in fallback_snap.values() if v is not None and not (isinstance(v, float) and math.isnan(v)))
                print(f"📸 Fallback atm_snapshot written: {len(fallback_snap)} total keys "
                      f"({atm_keys} atm/mm, {obs_keys} obs_snap, {valid_keys} non-null)")
            else:
                # No data available yet — don't write empty snapshot
                print(f"⚠️ Fallback atm_snapshot: no atmospheric/observational data available yet")
        except Exception as _fb_e:
            print(f"⚠️ Fallback atm_snapshot fetch failed: {_fb_e}")

    supabase_upsert(payload)
    # Log the 3-hourly snapshot archive
    atm_keys_archived = sum(1 for k in payload.get("atm_snapshot", "{}").count(":") if k) if isinstance(payload.get("atm_snapshot"), str) else 0
    print(f"📦 3-hourly snapshot archived for hour {snapshot_hour:02d}:00 EST "
          f"(prediction={payload.get('prediction_value', 'N/A')}°F)")

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
    # Pre-fetch atmospheric features so we can (a) pass to ML to avoid a second API
    # call and (b) dump a meaningful atm_snapshot to tomorrow's Supabase row — this
    # powers the Multi-Model Spread and NWS Jump cards in D+1 mode on the dashboard.
    live_atm_tm: dict = {}
    try:
        live_atm_tm = _fetch_atmospheric_features(tomorrow_iso)
        # Cache fallback: Fill missing values from cached snapshot if API partial failure
        live_atm_tm = fill_missing_from_cache(_CITY_KEY, live_atm_tm)
    except Exception as _atm_e:
        print(f"⚠️ write_today_for_tomorrow: _fetch_atmospheric_features failed: {_atm_e}")
        # Try cache as last resort if fetch completely fails
        try:
            cached = get_cached_snapshot(_CITY_KEY)
            if cached:
                live_atm_tm = cached
                print(f"  🔄 Using cached atmospheric snapshot for D+1 ({len(cached)} keys)")
        except Exception as _cache_e:
            print(f"  ⚠️  Cache fallback also failed: {_cache_e}")

    # ── Carry-forward: seed live_atm_tm with previous D+1 snapshot's mm_* values
    # if the fresh fetch missed them (same partial-fetch issue as write_today_for_today).
    # Reads from the existing tomorrow row's atm_snapshot in Supabase.
    _MM_CARRY_KEYS_TM = (
        "mm_spread", "mm_std", "mm_mean",
        "mm_hrrr_max", "mm_ecmwf_max", "mm_nbm_max", "mm_gem_max", "mm_icon_max",
        "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff",
        "mm_icon_gfs_diff", "mm_gem_ecmwf_diff",
        "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
        "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff",
        "atm_925mb_hrrr_max", "atm_925mb_hrrr_mean",
        "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
        "atm_925mb_gfs_hrrr_diff",
    )
    if live_atm_tm and isinstance(existing, dict):
        try:
            _prev_snap_tm_raw = existing.get("atm_snapshot")
            _prev_snap_tm = (
                json.loads(_prev_snap_tm_raw) if isinstance(_prev_snap_tm_raw, str)
                else (_prev_snap_tm_raw if isinstance(_prev_snap_tm_raw, dict) else {})
            )
            _carried_tm = []
            for _ck in _MM_CARRY_KEYS_TM:
                _fresh_v = live_atm_tm.get(_ck)
                _prev_v  = _prev_snap_tm.get(_ck)
                if (_fresh_v is None or (isinstance(_fresh_v, float) and math.isnan(_fresh_v))) \
                        and _prev_v is not None:
                    live_atm_tm[_ck] = _prev_v
                    _carried_tm.append(_ck)
            if _carried_tm:
                print(f"  🔁 D+1 carry-forward: {len(_carried_tm)} mm_/atm_ keys from prev snapshot "
                      f"(ensemble fetch partial)")
        except Exception as _cf_e:
            print(f"  ⚠️  D+1 carry-forward skipped: {_cf_e}")

    ml = _compute_ml_prediction(rows, tomorrow_iso, prefetched_atm=live_atm_tm if live_atm_tm else None)
    if ml:
        trigger = "first_write" if existing is _LOCK_NOT_FOUND else "intraday_refresh"
        _log_ml_revision(tomorrow_iso, "today_for_tomorrow", ml, trigger)

    # Only skip if we have NOTHING to write (no NWS, no ML, no atm snapshot)
    # At midnight NWS data may not be available yet, but we should still write the atmospheric snapshot
    if bcp_tm is None and ml is None and not live_atm_tm:
        print("⏭️ today_for_tomorrow: no forecasts and no atmospheric data available."); return

    # Canonical for D1 = first ML prediction made AFTER the Kalshi market opens.
    # Requiring market_probs ensures the bucket is mapped to real Kalshi structure,
    # not the model's internal 1°F buckets (which don't match Kalshi's daily structure).
    # Market opens ~10am ET the day before. Runs before 10am will skip canonical write.
    existing_has_canonical_d1 = isinstance(existing, dict) and existing.get("ml_bucket_canonical") is not None
    is_canonical_write = (existing is _LOCK_NOT_FOUND or not existing_has_canonical_d1) and ml is not None and bool(tomorrow_market_probs)

    ts = now_nyc().isoformat()
    now_hour = int(now_nyc().strftime("%H"))
    snapshot_hour = (now_hour // 3) * 3
    # D+1 also gets hourly snapshots for training progression
    idem_key = f"{_CITY_KEY}:{MODEL_VERSION}:{tomorrow_iso}:{snapshot_hour:02d}"

    payload = {
        "idempotency_key": idem_key,
        "timestamp": ts,
        "timestamp_et": ts,
        "target_date": tomorrow_iso,
        "snapshot_hour": snapshot_hour,  # Archive D+1 snapshots at 3-hour intervals
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
        else:
            # ── CRITICAL: Preserve canonical fields on intraday writes ──
            # Supabase merge-duplicates does wholesale row replacement.
            # If we don't include ml_f_canonical / ml_bucket_canonical in this upsert,
            # they'll be overwritten with NULL even though they were set on the
            # first (canonical) write. Carry them forward to prevent data loss.
            # (Only on intraday_refresh, when is_canonical_write=False)
            if isinstance(existing, dict):
                if existing.get("ml_f_canonical") is not None:
                    payload["ml_f_canonical"] = existing["ml_f_canonical"]
                if existing.get("ml_bucket_canonical") is not None:
                    payload["ml_bucket_canonical"] = existing["ml_bucket_canonical"]

    # Use already-fetched Kalshi market odds (from lock check above)
    market_probs = tomorrow_market_probs
    # Only save snapshot at canonical (first) write — same reason as write_today_for_today:
    # settled end-of-day prices (0.995/0.005) overwrite live prices and invert bet_signal.
    if market_probs and is_canonical_write:
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

                # Store second-best Kalshi bucket for later outcome tracking
                sorted_aligned = sorted(kalshi_aligned.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_aligned) > 1 and sorted_aligned[1][1] >= 0.02:
                    payload["ml_bucket_2"] = sorted_aligned[1][0]
                    payload["ml_bucket_2_prob"] = round(sorted_aligned[1][1], 4)
                    print(f"🥈 Bucket 2: {sorted_aligned[1][0]} ({sorted_aligned[1][1]:.0%})")

                if direct_bucket and kalshi_bucket != direct_bucket:
                    print(f"ℹ️ Direct map ({direct_bucket}) differs from prob-aligned ({kalshi_bucket}) "
                          f"— keeping prob-aligned (higher expected accuracy)")

                signal, edge, kelly_f = _compute_bet_signal(
                    payload["ml_confidence"], payload["ml_bucket"], market_probs
                )
                payload["bet_signal"] = signal
                payload["ml_edge"] = edge
                print(f"🎯 Bet signal: {signal} (edge={edge:+.0%}, kelly={kelly_f:.1%})")
                if is_canonical_write:
                    canonical_bucket = payload.get("ml_bucket", "")
                    payload["market_prob_at_prediction"] = round(market_probs.get(canonical_bucket, 0.0), 4)
                    payload["kelly_fraction"] = kelly_f
                    print(f"📌 market_prob_at_prediction={payload['market_prob_at_prediction']:.4f} for bucket '{canonical_bucket}'")
                    print(f"📐 kelly_fraction={kelly_f:.1%} (half-Kelly — bet this % of bankroll)")
        elif direct_bucket:
            payload["ml_bucket"] = direct_bucket
            signal, edge, kelly_f = _compute_bet_signal(
                ml["ml_confidence"], direct_bucket, market_probs
            )
            payload["bet_signal"] = signal
            payload["ml_edge"] = edge
            print(f"🎯 Bet signal: {signal} (edge={edge:+.0%}, kelly={kelly_f:.1%})")
            if is_canonical_write:
                payload["market_prob_at_prediction"] = round(market_probs.get(direct_bucket, 0.0), 4)
                payload["kelly_fraction"] = kelly_f
                print(f"📌 market_prob_at_prediction={payload['market_prob_at_prediction']:.4f} for bucket '{direct_bucket}'")
                print(f"📐 kelly_fraction={kelly_f:.1%} (half-Kelly — bet this % of bankroll)")

    # Write nws_d0 (= tomorrow's current NWS forecast) so the dashboard can show
    # "D+1 NWS: 55°F" immediately from ml.tomorrowNwsD0 without waiting for a
    # full write_today_for_today cycle.
    payload["nws_d0"] = nws_latest_tm

    # Build and write atm_snapshot for tomorrow's row so the Live Sources panel
    # can show Multi-Model Spread, NWS Jump, and observation sources in D+1 mode.
    # CRITICAL: Must include obs_snap_* keys so dashboard cards (Synoptic, MANH, regions) don't go blank.
    _ATM_SNAP_KEYS = (
        # ── Live observation sources (for Live Sources panel in D+1 mode) ─────
        # KNYC (Central Park) observations
        "obs_snap_knyc_temp", "obs_snap_knyc_max", "obs_snap_knyc_vs_nws",
        "obs_snap_knyc_obs_at", "obs_snap_knyc_rate", "obs_snap_knyc_rate_delta",
        # Regional stations (JFK/LGA for NYC)
        "obs_snap_kjfk_temp", "obs_snap_kjfk_vs_knyc", "obs_snap_kjfk_obs_at",
        "obs_snap_klga_temp", "obs_snap_klga_vs_knyc", "obs_snap_klga_obs_at",
        # Inland stations (EWR/TEB/CDW/SMQ for NYC)
        "obs_snap_kewr_temp", "obs_snap_kewr_vs_knyc", "obs_snap_kewr_obs_at",
        "obs_snap_kteb_temp", "obs_snap_kteb_vs_knyc", "obs_snap_kteb_obs_at",
        "obs_snap_kteb_delta",
        "obs_snap_kcdw_temp", "obs_snap_kcdw_obs_at", "obs_snap_kcdw_delta",
        "obs_snap_ksmq_temp", "obs_snap_ksmq_obs_at", "obs_snap_ksmq_delta",
        # Manhattan Mesonet (5-min fill-in for NYC)
        "obs_snap_manh_temp", "obs_snap_manh_vs_knyc", "obs_snap_manh_obs_at",
        # Synoptic network (5mi radius) — KEY for Synoptic distribution card
        "obs_snap_syn_mean", "obs_snap_syn_min", "obs_snap_syn_max",
        "obs_snap_syn_spread", "obs_snap_syn_vs_nws", "obs_snap_syn_count",
        # WU PWS (citizen weather stations)
        "obs_snap_wu_mean", "obs_snap_wu_vs_nws", "obs_snap_wu_spread", "obs_snap_wu_count",
        # Regional and gradient indicators
        "obs_snap_regional_spread", "obs_snap_regional_vs_nws",
        "obs_snap_coastal_vs_inland", "obs_snap_airport_spread",
        "obs_snap_inland_gradient", "obs_snap_inland_warming_rate",
        # Derived nowcast signals
        "obs_snap_morning_cloud", "obs_snap_stratus_clearing",
        "obs_snap_warming_accel",
        # ── Model and atmospheric features ────────────────────────────────────
        "mm_spread", "mm_std", "mm_mean",
        "mm_hrrr_ecmwf_diff", "mm_hrrr_gfs_diff", "mm_ecmwf_gfs_diff",
        "mm_hrrr_max", "mm_ecmwf_max", "mm_icon_max", "mm_gem_max",
        # v6 high-accuracy models
        "mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff",
        "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff",
        # v6 HRRR-specific pressure levels
        "atm_925mb_hrrr_max", "atm_925mb_hrrr_mean", "atm_925mb_gfs_hrrr_diff",
        "atm_850mb_hrrr_max", "atm_850mb_hrrr_mean",
        # v6 radiosonde
        "raob_925mb_temp", "raob_850mb_temp", "raob_700mb_temp",
        "raob_925mb_gfs_diff", "raob_925mb_hrrr_diff",
        "nws_overnight_jump", "nws_d1_final", "nws_last",
        "atm_bl_height_max", "atm_cape", "atm_precip_prob",
        "atm_rh_850", "atm_temp_850", "atm_temp_925",
        "atm_wind_speed", "atm_wind_dir",
        "ens_spread", "ens_mean",
        # Plume monitoring
        "atm_plume_monitoring",
    )
    if live_atm_tm:
        # Filter both None AND NaN — json.dumps emits bare 'NaN' for np.nan which is
        # not valid JSON and causes Supabase JSONB to reject the entire atm_snapshot,
        # leaving the Multi-Model Spread and HRRR vs NWS Gap cards blank on D+1.
        def _valid_snap(v):
            if v is None: return False
            if isinstance(v, float) and math.isnan(v): return False
            return True
        snap_tm = {k: live_atm_tm[k] for k in _ATM_SNAP_KEYS if _valid_snap(live_atm_tm.get(k))}
        if snap_tm:
            # Carry D+1 flip fields so we don't erase them on every ATM refresh.
            # D+1 flip tracking runs a few lines below; _carry_flip_fields only
            # copies fields not already in snap_tm, so the state machine can still
            # overwrite them cleanly.
            if isinstance(existing, dict):
                _carry_flip_fields(snap_tm, existing)
            payload["atm_snapshot"] = json.dumps(snap_tm)
            print(f"  📊 D+1 atm_snapshot: mm_spread={snap_tm.get('mm_spread')}, "
                  f"mm_mean={snap_tm.get('mm_mean')}, hrrr={snap_tm.get('mm_hrrr_max')}")

    # ── D+1 flip tracking (no confirmation gate — 30-min polling, less noisy) ──
    # Tomorrow's prediction reruns every 30 min as today's obs flow in.
    # No two-run confirmation gate needed (lower polling frequency = less jitter).
    # Tracks bucket changes vs the D+1 canonical so the dashboard card can show
    # "Opened 83-84, now 85-86" just like today's card does.
    if not is_canonical_write and isinstance(existing, dict):
        _d1_canon = existing.get("ml_bucket_canonical")
        _d1_curr  = payload.get("ml_bucket")
        if not _d1_canon:
            print(f"  ⚠️  D+1 flip tracking skipped: ml_bucket_canonical not set yet")
        elif not _d1_curr:
            print(f"  ⚠️  D+1 flip tracking skipped: ml_bucket absent from payload")
        if _d1_canon and _d1_curr:
            _d1_snap_raw = payload.get("atm_snapshot") or existing.get("atm_snapshot")
            try:
                _d1_fs = (json.loads(_d1_snap_raw) if isinstance(_d1_snap_raw, str)
                          else (_d1_snap_raw.copy() if isinstance(_d1_snap_raw, dict) else {})) \
                         if _d1_snap_raw else {}
            except Exception:
                _d1_fs = {}
            _d1_pfs_raw = existing.get("atm_snapshot")
            try:
                _d1_pfs = (json.loads(_d1_pfs_raw) if isinstance(_d1_pfs_raw, str)
                           else (_d1_pfs_raw if isinstance(_d1_pfs_raw, dict) else {})) \
                          if _d1_pfs_raw else {}
            except Exception:
                _d1_pfs = {}
            _d1_prev_count   = int(_d1_pfs.get("ml_flip_count") or 0)
            _d1_prev_pending = _d1_pfs.get("ml_flip_pending_bucket")
            _d1_prev_history = _d1_pfs.get("ml_flip_history") or []
            if isinstance(_d1_prev_history, str):
                try:    _d1_prev_history = json.loads(_d1_prev_history)
                except: _d1_prev_history = []
            if not isinstance(_d1_prev_history, list):
                _d1_prev_history = []

            if _d1_curr != _d1_canon:
                if _d1_prev_pending == _d1_curr:
                    # ── CONFIRMED: 2 consecutive runs agree on same non-canonical bucket ──
                    _d1_fs["ml_flip_occurred"]       = True
                    _d1_fs["ml_flip_count"]          = _d1_prev_count + 1
                    _d1_fs["ml_flip_bucket"]         = _d1_curr
                    _d1_fs["ml_flip_pending_bucket"] = None
                    if not _d1_prev_history or _d1_prev_history[-1] != _d1_curr:
                        _d1_fs["ml_flip_history"] = _d1_prev_history + [_d1_curr]
                    else:
                        _d1_fs["ml_flip_history"] = _d1_prev_history
                    _d1_chain = " → ".join([_d1_canon] + _d1_fs["ml_flip_history"])
                    print(f"  ✅ D+1 shift #{_d1_fs['ml_flip_count']} CONFIRMED (2 consecutive runs): {_d1_chain}")
                else:
                    # ── PENDING: first run seeing this non-canonical bucket ──
                    _d1_fs["ml_flip_pending_bucket"] = _d1_curr
                    if _d1_pfs.get("ml_flip_occurred"):
                        # Preserve any prior confirmed state
                        _d1_fs["ml_flip_occurred"] = True
                        _d1_fs["ml_flip_count"]    = _d1_prev_count
                        _d1_fs["ml_flip_bucket"]   = _d1_pfs.get("ml_flip_bucket", _d1_curr)
                        _d1_fs["ml_flip_history"]  = _d1_prev_history
                    print(f"  ⏳ D+1 pending ({_d1_canon} → {_d1_curr}) — awaiting one more run to confirm")
            elif _d1_pfs.get("ml_flip_occurred"):
                # Returned to canonical — preserve confirmed history, clear pending
                _d1_fs["ml_flip_pending_bucket"] = None
                _d1_fs["ml_flip_occurred"] = True
                _d1_fs["ml_flip_count"]    = _d1_prev_count
                _d1_fs["ml_flip_bucket"]   = _d1_pfs.get("ml_flip_bucket", _d1_curr)
                _d1_fs["ml_flip_history"]  = _d1_prev_history
            else:
                # Back to canonical with no prior confirmed flip — clear any pending silently
                _d1_fs["ml_flip_pending_bucket"] = None
            if _d1_fs:
                payload["atm_snapshot"] = json.dumps(_d1_fs)

    # ── Fallback: ensure atm_snapshot is always written (D+1) ──────────────────
    # Same logic as write_today_for_today: if we haven't set atm_snapshot yet,
    # fetch atmospheric + observational features for tomorrow's Live Sources panel.
    if "atm_snapshot" not in payload:
        try:
            fallback_atm_tm = _fetch_atmospheric_features(tomorrow_iso)
            fallback_obs_tm = _fetch_observation_features(tomorrow_iso) or {}

            # Build snapshot with both ATM and OBS data
            fallback_snap_tm = fallback_atm_tm or {}
            if fallback_obs_tm:
                _add_obs_to_snap(fallback_snap_tm, fallback_obs_tm, fallback_atm_tm)

            if fallback_snap_tm and any(
                v is not None and not (isinstance(v, float) and math.isnan(v))
                for v in fallback_snap_tm.values()
            ):
                # Cache for future fallback
                cache_snapshot(_CITY_KEY, fallback_snap_tm)
                payload["atm_snapshot"] = json.dumps(fallback_snap_tm)
                atm_keys = sum(1 for k in fallback_snap_tm.keys() if k.startswith("atm_") or k.startswith("mm_"))
                obs_keys = sum(1 for k in fallback_snap_tm.keys() if k.startswith("obs_snap_"))
                print(f"📸 D+1 fallback atm_snapshot written: {len(fallback_snap_tm)} total keys "
                      f"({atm_keys} atm/mm, {obs_keys} obs_snap)")
            else:
                print(f"⚠️ D+1 fallback atm_snapshot: no atmospheric/observational data available yet")
        except Exception as _fb_e:
            print(f"⚠️ D+1 fallback atm_snapshot fetch failed: {_fb_e}")

    supabase_upsert(payload)
    # Log the 3-hourly D+1 snapshot archive
    print(f"📦 D+1 3-hourly snapshot archived for hour {snapshot_hour:02d}:00 EST "
          f"(prediction={payload.get('prediction_value', 'N/A')}°F)")

def write_both_snapshots() -> None:
    rows, _ = _read_all_rows()
    # Score yesterday's prediction against actual high
    try: score_yesterday_prediction(rows)
    except Exception as e: print("⚠️ score_yesterday_prediction failed:", e)
    # Backfill any historical rows where canonical was set but result was never scored
    try: backfill_canonical_results()
    except Exception as e: print("⚠️ backfill_canonical_results failed:", e)
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
    s.add_parser("backfill_obs_historical")
    s.add_parser("backfill_high_timing")
    args = p.parse_args()

    set_city(args.city)
    global _CITY_KEY
    _CITY_KEY = args.city
    print(f"[prediction_writer] city={args.city}")

    if args.cmd == "collect_obs":              collect_nws_observations(args.city)
    elif args.cmd == "backfill_obs":             backfill_observation_features(args.city)
    elif args.cmd == "backfill_obs_historical": backfill_obs_historical(args.city)
    elif args.cmd == "backfill_high_timing":    backfill_high_timing_features(args.city)
    elif args.cmd == "today_for_today":    write_today_for_today(args.date)
    elif args.cmd == "today_for_tomorrow": write_today_for_tomorrow(args.date)
    else: write_both_snapshots()

if __name__ == "__main__": _cli()
