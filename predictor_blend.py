"""
predictor_blend.py — production library for confidence-weighted KNN + HRRR blending.

The moat:
  KNN finds K=20 most similar historical days by weighted feature distance.
  Their actual highs form a prediction distribution; median = KNN prediction,
  std = KNN confidence (low std = neighbors agreed = trust KNN).
  Final prediction blends with HRRR based on KNN's confidence.

Backtest result (588 days, leave-one-out):
  HRRR alone MAE:  2.216°F
  Blend MAE:       1.940°F  (improvement +0.276°F)

Used by prediction_writer.py at inference time. Loads multiyear CSV once
per process (cached); KNN search per prediction is ~50ms.

Conservatism:
  - Returns None if not enough features or training rows available
  - Caller falls back to HRRR-direct in that case
  - Production-safe: never raises; always returns either (pred, weight) or None
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Configuration (tuned via backtest) ────────────────────────────────────
FEATURES = [
    "mm_hrrr_max", "mm_nbm_max", "atm_925mb_temp_mean",
    "atm_bl_height_max", "atm_cloud_cover_max",
    "atm_wind_dir_sin", "atm_wind_dir_cos",
    "atm_dewpoint_mean", "atm_solar_radiation_peak",
    "doy_sin", "doy_cos",
]

FEATURE_WEIGHTS = {
    "mm_hrrr_max":              3.0,
    "mm_nbm_max":               2.0,
    "atm_925mb_temp_mean":      1.5,
    "atm_bl_height_max":        0.5,
    "atm_cloud_cover_max":      1.0,
    "atm_wind_dir_sin":         0.8,
    "atm_wind_dir_cos":         0.8,
    "atm_dewpoint_mean":        1.0,
    "atm_solar_radiation_peak": 1.2,
    "doy_sin":                  1.0,
    "doy_cos":                  1.0,
}

K_NEIGHBORS = 20
BLEND_SCALE = 10.0
MIN_KNN_WEIGHT = 0.10
MIN_REQUIRED_FEATURES = 5  # need at least this many populated test features

# Cached on first load (per process)
_CACHE: dict = {}


def _load_multiyear(city_prefix: str = "") -> Optional[dict]:
    """Load multiyear CSV + precompute normalization stats. Cached."""
    cache_key = f"{city_prefix}data"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    csv_path = Path(f"{city_prefix}multiyear_atmospheric.csv")
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Add seasonal features
    try:
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df["doy"] = df["target_date"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    except Exception:
        return None

    # Need actual_high + mm_hrrr_max + at least MIN_REQUIRED_FEATURES other features
    df = df[df["actual_high"].notna() & df["mm_hrrr_max"].notna()].copy()
    if len(df) < 50:
        return None

    available_feats = [c for c in FEATURES if c in df.columns]
    if len(available_feats) < MIN_REQUIRED_FEATURES:
        return None

    # Restrict to rows with all available features populated
    complete = df[df[available_feats].notna().all(axis=1)].reset_index(drop=True)
    if len(complete) < 50:
        return None

    # Normalize features (z-score)
    feat_means = complete[available_feats].mean()
    feat_stds = complete[available_feats].std().replace(0, 1)
    norm_feats = (complete[available_feats] - feat_means) / feat_stds

    bundle = {
        "df": complete,
        "feat_means": feat_means.to_dict(),
        "feat_stds": feat_stds.to_dict(),
        "X": norm_feats.to_numpy(dtype=float),
        "actuals": complete["actual_high"].to_numpy(dtype=float),
        "available_feats": available_feats,
        "weight_arr": np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in available_feats]),
    }
    _CACHE[cache_key] = bundle
    return bundle


def predict_blend(
    live_features: dict,
    hrrr_value: float,
    city_prefix: str = "",
    k: int = K_NEIGHBORS,
    scale: float = BLEND_SCALE,
    min_w: float = MIN_KNN_WEIGHT,
) -> Optional[dict]:
    """
    Returns dict with:
      - "blend": final prediction (HRRR + KNN weighted)
      - "knn_pred": KNN's median-of-neighbors prediction
      - "knn_std": std of neighbor actuals (KNN's uncertainty)
      - "weight_knn": how much weight given to KNN (0 to 1)
      - "n_neighbors": number of neighbors used
    Or None if KNN can't run (insufficient data / features).
    """
    if hrrr_value is None or not np.isfinite(hrrr_value):
        return None

    bundle = _load_multiyear(city_prefix)
    if bundle is None:
        return None

    # Add seasonal features to live_features if not already there
    if "doy_sin" not in live_features or "doy_cos" not in live_features:
        try:
            from datetime import datetime
            today_doy = datetime.now().timetuple().tm_yday
            live_features = dict(live_features)
            live_features.setdefault("doy_sin", np.sin(2 * np.pi * today_doy / 365))
            live_features.setdefault("doy_cos", np.cos(2 * np.pi * today_doy / 365))
        except Exception:
            pass

    # Build test feature vector (normalized using training stats)
    avail = bundle["available_feats"]
    weight_arr = bundle["weight_arr"]
    means = bundle["feat_means"]
    stds = bundle["feat_stds"]

    test_vec_raw = []
    test_avail_mask = []
    for c in avail:
        v = live_features.get(c)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            test_vec_raw.append(0.0)
            test_avail_mask.append(False)
        else:
            test_vec_raw.append(float(v))
            test_avail_mask.append(True)
    n_available_test = sum(test_avail_mask)
    if n_available_test < MIN_REQUIRED_FEATURES:
        return None

    # Normalize test vector using training stats
    test_vec = np.array(test_vec_raw, dtype=float)
    test_vec_norm = np.array([
        (test_vec[i] - means.get(c, 0)) / stds.get(c, 1) if test_avail_mask[i] else 0.0
        for i, c in enumerate(avail)
    ])

    # Use only the features present in the test row for distance calculation
    use_idx = [i for i, ok in enumerate(test_avail_mask) if ok]
    if len(use_idx) < MIN_REQUIRED_FEATURES:
        return None

    X = bundle["X"][:, use_idx]
    test_use = test_vec_norm[use_idx]
    weight_use = weight_arr[use_idx]

    # Weighted Euclidean distance
    diff = X - test_use
    dists = np.sqrt(((diff ** 2) * weight_use).sum(axis=1))
    k_use = min(k, len(dists))
    top_k_idx = np.argpartition(dists, k_use - 1)[:k_use]

    neighbor_actuals = bundle["actuals"][top_k_idx]
    knn_pred = float(np.median(neighbor_actuals))
    knn_std = float(np.std(neighbor_actuals))

    # Confidence-weighted blend
    weight_knn = max(min_w, min(1.0, 1.0 - knn_std / scale))
    blend = weight_knn * knn_pred + (1.0 - weight_knn) * hrrr_value

    return {
        "blend": float(blend),
        "knn_pred": knn_pred,
        "knn_std": knn_std,
        "weight_knn": float(weight_knn),
        "hrrr": float(hrrr_value),
        "n_neighbors": int(k_use),
        "n_features_used": len(use_idx),
    }
