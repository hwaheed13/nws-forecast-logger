#!/usr/bin/env python3
"""
exp_v16_sweep.py — one-off experiment: can v16 be made stronger on the
CORRECTED ground truth with honest date-grouped CV?

Variants (per city):
  A baseline   — current production config (all 186 features, heavy reg)
  B pruned     — drop features with <10% coverage in the training pool
  C relaxed    — lighter regularization (depth 4, leaf 15, l2 0.5, lr 0.05)
  D pruned+rel — B and C combined

Prints CV MAE (residual), bucket acc, and improvement vs HRRR-alone for each.
Does NOT save any models — read-only experiment.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from model_config import FEATURE_COLS_V16
from train_models import NYCTemperatureModelTrainer


def build_pool(city: str):
    t = NYCTemperatureModelTrainer(city_key=city)
    t.load_data()
    t.build_feature_matrix()
    t.train_v2()   # does the multiyear merge + obs proxy fills
    t.train_v3()
    # replicate train_v16's derived-feature computation + pool filter
    t._compute_bl_safeguard_features()
    t._compute_model_vs_nws_features()
    t._compute_blind_spot_features()
    t._compute_v15_features()
    df = t.features_df
    for col in FEATURE_COLS_V16:
        if col not in df.columns:
            df[col] = np.nan
    mask = df["actual_high"].notna() & df["mm_hrrr_max"].notna()
    pool = df[mask].copy().sort_values("target_date").reset_index(drop=True)
    return pool


def date_grouped_splits(dates, n_splits=5):
    uniq = np.array(sorted(dates.unique()))
    out = []
    for tr_d, te_d in TimeSeriesSplit(n_splits=n_splits).split(uniq):
        tr, te = set(uniq[tr_d]), set(uniq[te_d])
        out.append((np.flatnonzero(dates.isin(tr).to_numpy()),
                    np.flatnonzero(dates.isin(te).to_numpy())))
    return out


BASE = dict(max_iter=800, learning_rate=0.03, max_depth=3, min_samples_leaf=25,
            l2_regularization=1.5, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42)
RELAXED = dict(BASE, max_depth=4, min_samples_leaf=15, l2_regularization=0.5,
               learning_rate=0.05)


def evaluate(pool, feature_cols, params, label):
    X = pool[feature_cols]
    hrrr = pool["mm_hrrr_max"].astype(float)
    y = pool["actual_high"].astype(float) - hrrr
    splits = date_grouped_splits(pool["target_date"].astype(str))
    est = HistGradientBoostingRegressor(**params)
    mae = -cross_val_score(est, X, y, cv=splits, scoring="neg_mean_absolute_error")
    hrrr_arr, actual_arr = hrrr.to_numpy(), pool["actual_high"].astype(float).to_numpy()

    def bucket(est_, X_, y_):
        idx = X_.index.to_numpy()
        preds = est_.predict(X_) + hrrr_arr[idx]
        return float(np.mean(np.abs(preds - actual_arr[idx]) <= 1))

    bkt = cross_val_score(est, X, y, cv=splits, scoring=bucket)
    baseline = float(np.abs(y).mean())
    print(f"  {label:<14} nfeat={len(feature_cols):3d}  "
          f"CV_MAE={np.mean(mae):.3f}  bucket={np.mean(bkt):.1%}  "
          f"vs_HRRR={baseline - np.mean(mae):+.3f}")
    return float(np.mean(mae))


def main(city: str):
    print(f"\n{'═'*60}\nEXPERIMENT {city.upper()}\n{'═'*60}")
    pool = build_pool(city)
    print(f"\nPool: {len(pool)} rows, HRRR-alone MAE "
          f"{(pool['actual_high'].astype(float) - pool['mm_hrrr_max'].astype(float)).abs().mean():.3f}")

    cov = pool[FEATURE_COLS_V16].notna().mean()
    pruned = [c for c in FEATURE_COLS_V16 if cov[c] >= 0.10]
    dropped = sorted(set(FEATURE_COLS_V16) - set(pruned))
    print(f"Pruning drops {len(dropped)} features <10% coverage: {dropped[:12]}{'…' if len(dropped) > 12 else ''}\n")

    print("RESULTS (honest date-grouped CV, official actuals):")
    evaluate(pool, list(FEATURE_COLS_V16), BASE,    "A baseline")
    evaluate(pool, pruned,                 BASE,    "B pruned")
    evaluate(pool, list(FEATURE_COLS_V16), RELAXED, "C relaxed")
    evaluate(pool, pruned,                 RELAXED, "D pruned+rel")


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["nyc", "lax"]):
        main(c)
