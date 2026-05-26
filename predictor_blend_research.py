#!/usr/bin/env python3
"""
predictor_blend_research.py — Phase 3 research

Tests CONFIDENCE-WEIGHTED blending of KNN + HRRR. Uses KNN's own internal
uncertainty (neighbor variance) to decide how much to trust KNN vs HRRR
on each prediction.

Hypothesis:
  When today's 20 nearest historical days had tight outcomes (low variance),
  KNN is confident → trust it. When they ranged widely, defer to HRRR.

  No pre-day classifier needed. KNN judges its own confidence per-prediction.

If this works (≥0.3°F MAE improvement over HRRR), it's a viable production
strategy — uses 4yr data + 186 features WHERE they have signal, defers to
HRRR where they don't.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

FEATURES = [
    "mm_hrrr_max", "mm_nbm_max", "atm_925mb_temp_mean",
    "atm_bl_height_max", "atm_cloud_cover_max",
    "atm_wind_dir_sin", "atm_wind_dir_cos",
    "atm_dewpoint_mean", "atm_solar_radiation_peak",
]

KNN_WEIGHTS = {
    "mm_hrrr_max": 3.0, "mm_nbm_max": 2.0,
    "atm_925mb_temp_mean": 1.5, "atm_bl_height_max": 0.5,
    "atm_cloud_cover_max": 1.0,
    "atm_wind_dir_sin": 0.8, "atm_wind_dir_cos": 0.8,
    "atm_dewpoint_mean": 1.0, "atm_solar_radiation_peak": 1.2,
    "doy_sin": 1.0, "doy_cos": 1.0,
}

K = 20

# Blending function tested as a hyperparameter:
#   weight_knn = max(MIN_W, 1 - neighbor_std / SCALE)
# When neighbor_std = 0, weight_knn = 1 (full KNN)
# When neighbor_std = SCALE, weight_knn = 0 (full HRRR)
# When neighbor_std > SCALE, weight_knn = MIN_W (floor)
MIN_W = 0.1   # never go fully HRRR — always include 10% KNN to never hide signal
SCALES_TO_TEST = [2.0, 3.0, 4.0, 5.0, 6.0]


def load_data(csv_path: str = "multiyear_atmospheric.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date
    df["doy"] = pd.to_datetime(df["target_date"]).dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    return df


def predict_knn_with_uncertainty(test_features, train_features, train_actuals, weights, k):
    """Returns (median_prediction, neighbor_std)."""
    diff = train_features - test_features
    dists = np.sqrt(((diff ** 2) * weights).sum(axis=1))
    k_use = min(k, len(train_actuals))
    top_k_idx = np.argpartition(dists, k_use - 1)[:k_use]
    neighbor_actuals = train_actuals[top_k_idx]
    return float(np.median(neighbor_actuals)), float(np.std(neighbor_actuals))


def main() -> int:
    df = load_data()
    df = df[df["actual_high"].notna() & df["mm_hrrr_max"].notna()].copy().reset_index(drop=True)
    feature_cols = [c for c in FEATURES + ["doy_sin", "doy_cos"] if c in df.columns]
    complete = df[df[feature_cols].notna().all(axis=1)].reset_index(drop=True)
    print(f"Working with {len(complete)} HRRR-anchored complete-feature days")

    feat_means = complete[feature_cols].mean()
    feat_stds = complete[feature_cols].std().replace(0, 1)
    norm_feats = (complete[feature_cols] - feat_means) / feat_stds

    actuals = complete["actual_high"].to_numpy(dtype=float)
    hrrr_vals = complete["mm_hrrr_max"].to_numpy(dtype=float)
    X = norm_feats.to_numpy(dtype=float)
    weight_arr = np.array([KNN_WEIGHTS.get(c, 1.0) for c in feature_cols])

    n = len(complete)
    print(f"\nLeave-one-out backtest (n={n})...")

    knn_preds = np.zeros(n)
    knn_stds = np.zeros(n)
    for i in range(n):
        if i % 200 == 0:
            print(f"  {i}/{n}...")
        train_idx = np.array([j for j in range(n) if j != i])
        pred, std = predict_knn_with_uncertainty(
            X[i], X[train_idx], actuals[train_idx], weight_arr, K,
        )
        knn_preds[i] = pred
        knn_stds[i] = std

    # Baselines
    hrrr_mae = np.abs(hrrr_vals - actuals).mean()
    knn_mae = np.abs(knn_preds - actuals).mean()
    print(f"\nBaselines:")
    print(f"  HRRR alone MAE: {hrrr_mae:.3f}°F")
    print(f"  KNN alone MAE:  {knn_mae:.3f}°F")

    # Sweep SCALE
    print(f"\nBlend sweep (weight_knn = max({MIN_W}, 1 - neighbor_std / SCALE)):")
    print(f"  {'SCALE':>6} {'Blend MAE':>11s} {'vs HRRR':>10s} {'avg w_knn':>10s} {'knn-leaning days':>20s}")

    best_scale = None
    best_mae = float("inf")
    for scale in SCALES_TO_TEST:
        weights_knn = np.maximum(MIN_W, 1 - knn_stds / scale)
        blend = weights_knn * knn_preds + (1 - weights_knn) * hrrr_vals
        blend_mae = np.abs(blend - actuals).mean()
        improvement = hrrr_mae - blend_mae
        avg_w = float(weights_knn.mean())
        # Days where blend leaned >50% KNN
        knn_leaning = int((weights_knn > 0.5).sum())
        print(f"  {scale:>6.1f} {blend_mae:>10.3f}°F {improvement:>+9.3f}°F {avg_w:>10.2f} "
              f"{knn_leaning:>10d}/{n} ({100*knn_leaning/n:.0f}%)")
        if blend_mae < best_mae:
            best_mae = blend_mae
            best_scale = scale

    print(f"\n🎯 Best SCALE: {best_scale}  →  Blend MAE {best_mae:.3f}°F  "
          f"(vs HRRR {hrrr_mae:.3f}°F, improvement {hrrr_mae - best_mae:+.3f}°F)")

    # Where did the blend help vs hurt?
    weights_knn_best = np.maximum(MIN_W, 1 - knn_stds / best_scale)
    blend_best = weights_knn_best * knn_preds + (1 - weights_knn_best) * hrrr_vals
    blend_err = blend_best - actuals
    hrrr_err = hrrr_vals - actuals

    # Subset analysis: when blend leaned heavily on KNN (>0.7), did it win?
    knn_heavy = weights_knn_best > 0.7
    if knn_heavy.sum() > 0:
        sub_blend = np.abs(blend_err[knn_heavy]).mean()
        sub_hrrr = np.abs(hrrr_err[knn_heavy]).mean()
        print(f"\nWhen KNN weight > 0.7 ({knn_heavy.sum()} days):")
        print(f"  Blend MAE: {sub_blend:.2f}°F  vs HRRR MAE: {sub_hrrr:.2f}°F  "
              f"(diff {sub_hrrr - sub_blend:+.2f}°F)")

    knn_light = weights_knn_best < 0.3
    if knn_light.sum() > 0:
        sub_blend = np.abs(blend_err[knn_light]).mean()
        sub_hrrr = np.abs(hrrr_err[knn_light]).mean()
        print(f"\nWhen KNN weight < 0.3 ({knn_light.sum()} days, deferred to HRRR):")
        print(f"  Blend MAE: {sub_blend:.2f}°F  vs HRRR MAE: {sub_hrrr:.2f}°F  "
              f"(diff {sub_hrrr - sub_blend:+.2f}°F)")

    # Save results
    out = pd.DataFrame({
        "target_date": complete["target_date"].values,
        "actual": actuals, "hrrr": hrrr_vals,
        "knn_pred": knn_preds, "knn_std": knn_stds,
        "blend_pred": blend_best, "weight_knn": weights_knn_best,
        "blend_err": blend_err, "hrrr_err": hrrr_err,
    })
    out.to_csv("predictor_blend_results.csv", index=False)
    print(f"\nResults saved to predictor_blend_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
