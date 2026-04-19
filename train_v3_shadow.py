#!/usr/bin/env python3
# train_v3_shadow.py — Train a SHADOW atm_predictor from forecast-sourced data.
#
# Mirrors _train_atm_predictor() in train_models.py but reads
# multiyear_atmospheric_forecast.csv (historical-forecast-api, GFS/ECMWF)
# instead of multiyear_atmospheric.csv (archive-api, ERA5).
#
# Output: {prefix}atm_predictor_forecast.pkl
# Consumed by prediction_writer.py shadow block; production remains on
# atm_predictor.pkl until shadow logs prove forecast-sourced wins.
#
# Usage:
#   python train_v3_shadow.py --city nyc
#   python train_v3_shadow.py --all

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from city_config import get_city_config, CITIES
from model_config import ATM_PREDICTOR_INPUT_COLS


def train_shadow(city_key: str) -> str | None:
    cfg = get_city_config(city_key)
    prefix = cfg.get("model_prefix", "")
    src = f"{prefix}multiyear_atmospheric_forecast.csv"
    out = f"{prefix}atm_predictor_forecast.pkl"

    print(f"\n{'='*60}")
    print(f"Training SHADOW atm_predictor for {cfg['label']}")
    print(f"{'='*60}")

    if not os.path.exists(src):
        print(f"❌ {src} not found. Run backfill_atmospheric_forecast.py --city {city_key}")
        return None

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from {src}")

    # actual_high lives in the ERA5 CSV — merge it in on target_date
    era5_src = f"{prefix}multiyear_atmospheric.csv"
    if "actual_high" not in df.columns:
        if not os.path.exists(era5_src):
            print(f"❌ Need {era5_src} for actual_high labels")
            return None
        era5 = pd.read_csv(era5_src, usecols=["target_date", "actual_high"])
        era5["target_date"] = era5["target_date"].astype(str).str[:10]
        df["target_date"] = df["target_date"].astype(str).str[:10]
        df = df.merge(era5, on="target_date", how="inner")
        print(f"Merged actual_high labels — {len(df)} rows remain")

    # Add temporal features (same derivation as _build_multiyear_features)
    df["target_date"] = pd.to_datetime(df["target_date"])
    doy = df["target_date"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["target_date"].dt.month
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    # Ensure all input cols exist
    for col in ATM_PREDICTOR_INPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    if "actual_high" not in df.columns:
        print("❌ actual_high missing from CSV")
        return None

    df = df.dropna(subset=["actual_high"]).reset_index(drop=True)

    X = df[ATM_PREDICTOR_INPUT_COLS]
    y = df["actual_high"]

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for tr, te in tscv.split(X):
        m = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0, random_state=42,
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        cv_maes.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))

    print(f"Shadow CV MAE: {np.mean(cv_maes):.2f}°F "
          f"(folds: {[f'{s:.2f}' for s in cv_maes]})")
    print(f"Trained on {len(df)} days, {len(ATM_PREDICTOR_INPUT_COLS)} features")

    final = HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=15, l2_regularization=1.0, random_state=42,
    )
    final.fit(X, y)

    save_data = {
        "model": final,
        "features": ATM_PREDICTOR_INPUT_COLS,
        "cv_mae": round(float(np.mean(cv_maes)), 2),
        "n_training_days": len(df),
        "source": "historical-forecast-api",
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(out, "wb") as f:
        pickle.dump(save_data, f)
    print(f"✅ Saved shadow predictor to {out}")

    # Compare to production predictor
    prod = f"{prefix}atm_predictor.pkl"
    if os.path.exists(prod):
        try:
            with open(prod, "rb") as f:
                prod_data = pickle.load(f)
            prod_mae = prod_data.get("cv_mae")
            if prod_mae is not None:
                delta = prod_mae - float(np.mean(cv_maes))
                print(f"\nProduction atm_predictor CV MAE: {prod_mae:.2f}°F")
                print(f"Shadow delta: {delta:+.2f}°F "
                      f"({'shadow better' if delta > 0 else 'shadow worse' if delta < 0 else 'tie'})")
                print("(In-sample CV comparison only; the real test is live shadow logs.)")
        except Exception as e:
            print(f"(Could not compare to production predictor: {e})")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="nyc")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for c in CITIES:
            train_shadow(c)
    else:
        train_shadow(args.city)
    return 0


if __name__ == "__main__":
    sys.exit(main())
