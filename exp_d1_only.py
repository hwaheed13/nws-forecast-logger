#!/usr/bin/env python3
"""One-off: is the day-before-lead HRRR residual learnable on a HOMOGENEOUS
d1-anchored pool (no mixed-anchor confound)? Decides whether v16 adopts the
d1 anchor or reverts to hindcast."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from model_config import FEATURE_COLS_V16
from exp_v16_sweep import build_pool, date_grouped_splits, BASE

for city in ("nyc", "lax"):
    pool = build_pool(city)
    d1 = pd.to_numeric(pool.get("mm_hrrr_max_d1"), errors="coerce")
    sub = pool[d1.notna()].reset_index(drop=True)
    y = sub["actual_high"].astype(float) - pd.to_numeric(sub["mm_hrrr_max_d1"])
    X = sub[FEATURE_COLS_V16]
    splits = date_grouped_splits(sub["target_date"].astype(str))
    est = HistGradientBoostingRegressor(**BASE)
    mae = -cross_val_score(est, X, y, cv=splits, scoring="neg_mean_absolute_error")
    print(f"RESULT {city}: d1-only pool={len(sub)} rows "
          f"({sub['target_date'].min()}→{sub['target_date'].max()})", flush=True)
    print(f"RESULT {city}: d1-HRRR-alone MAE {y.abs().mean():.3f} | "
          f"v16-on-d1 CV MAE {np.mean(mae):.3f} | "
          f"moat {y.abs().mean() - np.mean(mae):+.3f}", flush=True)
