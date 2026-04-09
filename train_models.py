# train_models.py
from __future__ import annotations

import json
import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from model_config import (
    FEATURE_COLS, FEATURE_COLS_V2, FEATURE_COLS_V3, ACCU_NWS_FALLBACK,
    ATM_PREDICTOR_INPUT_COLS, ATM_PREDICTOR_COLS, FORECAST_REVISION_COLS,
)

warnings.filterwarnings("ignore")


def _safe_parse_ts(x):
    """
    Robust per-row timestamp parser for inputs like:
      '2025-09-12 14:59:22 EDT', '2025-09-14 04:37:00', ' 2025-09-15 04:46:39 EDT '
    Normalizes to America/New_York then returns naive local time.
    """
    if x in (None, "", "None", "null", "NaT"):
        return pd.NaT

    s = str(x).strip()
    # normalize common tz abbreviations
    if "EDT" in s:
        s = s.replace("EDT", "-04:00").strip()
    elif "EST" in s:
        s = s.replace("EST", "-05:00").strip()

    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        # assume NYC if no tz
        if getattr(ts, "tz", None) is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
        return ts.tz_localize(None)
    except Exception:
        return pd.NaT


def _normalize_date_col(df: pd.DataFrame, col: str):
    """Coerce a date-like column to 'YYYY-MM-DD' strings with NaNs dropped later by caller."""
    if col in df.columns:
        coerced = pd.to_datetime(df[col], errors="coerce").dt.date
        df[col] = coerced.astype("string")  # keeps 'YYYY-MM-DD' or <NA>


MIN_DAYS_FOR_TRAINING = 60  # minimum days with actual data before training is viable


class NYCTemperatureModelTrainer:
    def __init__(self, city_key: str = "nyc"):
        from city_config import get_city_config
        self.city_key = city_key
        self.city_cfg = get_city_config(city_key)
        self.nws_csv = self.city_cfg["nws_csv"]
        self.accu_csv = self.city_cfg["accu_csv"]
        self.model_prefix = self.city_cfg.get("model_prefix", "")
        self.nws_df: pd.DataFrame | None = None
        self.accu_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None
        self.temp_model = None
        self.residual_std: float = 2.0  # default, updated during training
        self.cv_mae_scores: list[float] = []
        self.cv_bucket_acc_scores: list[float] = []

    def _require(self, df, needed, name):
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {', '.join(missing)}")

    def load_data(self):
        print(f"Loading CSV files for {self.city_cfg['label']}...")
        # NWS file required
        self.nws_df = pd.read_csv(self.nws_csv)
        self._require(
            self.nws_df,
            [
                "timestamp",
                "forecast_or_actual",
                "predicted_high",
                "target_date",
                "cli_date",
                "actual_high",
            ],
            self.nws_csv,
        )

        # AccuWeather optional
        try:
            self.accu_df = pd.read_csv(self.accu_csv)
            if not self.accu_df.empty:
                self._require(
                    self.accu_df,
                    ["timestamp", "forecast_or_actual", "predicted_high", "target_date"],
                    self.accu_csv,
                )
        except FileNotFoundError:
            print("AccuWeather file not found, proceeding with NWS only")
            self.accu_df = pd.DataFrame()

        # timestamps with safe parser
        self.nws_df["timestamp"] = self.nws_df["timestamp"].apply(_safe_parse_ts)
        self.nws_df = self.nws_df.dropna(subset=["timestamp"])

        if not self.accu_df.empty:
            # bail if column has structured objects
            if self.accu_df["timestamp"].apply(lambda v: isinstance(v, (dict, list))).any():
                print("AccuWeather timestamp column is structured. Skipping AccuWeather.")
                self.accu_df = pd.DataFrame()
            else:
                self.accu_df["timestamp"] = self.accu_df["timestamp"].apply(_safe_parse_ts)
                invalid = float(self.accu_df["timestamp"].isna().mean())
                if invalid > 0.5:
                    print(f"AccuWeather timestamps too messy ({invalid:.0%} invalid). Skipping AccuWeather.")
                    self.accu_df = pd.DataFrame()
                else:
                    self.accu_df = self.accu_df.dropna(subset=["timestamp"])

        # normalize date columns to sortable, consistent strings
        _normalize_date_col(self.nws_df, "target_date")
        _normalize_date_col(self.nws_df, "cli_date")
        if not self.accu_df.empty:
            _normalize_date_col(self.accu_df, "target_date")

        print(f"Loaded {len(self.nws_df)} NWS rows, {len(self.accu_df)} AccuWeather rows")

    def extract_features_for_date(self, target_date: str, prev_day_high: float = None):
        # NWS forecasts for the target date
        nws_forecasts = self.nws_df[
            (self.nws_df["target_date"] == target_date)
            & (self.nws_df["forecast_or_actual"] == "forecast")
        ].copy()

        # AccuWeather forecasts for the target date (optional)
        if not self.accu_df.empty:
            accu_forecasts = self.accu_df[
                (self.accu_df["target_date"] == target_date)
                & (self.accu_df["forecast_or_actual"] == "forecast")
            ].copy()
        else:
            accu_forecasts = pd.DataFrame()

        # Actual row for the date
        actual_row = self.nws_df[
            (self.nws_df["cli_date"] == target_date)
            & (self.nws_df["forecast_or_actual"] == "actual")
        ]

        if actual_row.empty or nws_forecasts.empty:
            return None

        actual_high = float(actual_row.iloc[0]["actual_high"])

        # Sort by issuance time
        nws_forecasts = nws_forecasts.sort_values("timestamp")
        if not accu_forecasts.empty:
            accu_forecasts = accu_forecasts.sort_values("timestamp")

        nws_values = nws_forecasts["predicted_high"].astype(float).values
        month = int(target_date.split("-")[1])

        # Day of year for cyclical encoding
        doy = datetime.strptime(target_date, "%Y-%m-%d").timetuple().tm_yday

        # --- NWS features ---
        features = {
            "nws_first": nws_values[0] if len(nws_values) > 0 else np.nan,
            "nws_last": nws_values[-1] if len(nws_values) > 0 else np.nan,
            "nws_max": nws_values.max() if len(nws_values) > 0 else np.nan,
            "nws_min": nws_values.min() if len(nws_values) > 0 else np.nan,
            "nws_mean": nws_values.mean() if len(nws_values) > 0 else np.nan,
            "nws_count": len(nws_values),
            "nws_spread": (nws_values.max() - nws_values.min()) if len(nws_values) > 0 else 0.0,
            "nws_std": float(nws_values.std()) if len(nws_values) > 1 else 0.0,
            "nws_trend": float(nws_values[-1] - nws_values[0]) if len(nws_values) > 1 else 0.0,
            "forecast_velocity": self._calculate_velocity(nws_values),
            "forecast_acceleration": self._calculate_acceleration(nws_values),
        }

        # --- Intraday revision feature (NWS) ---
        # How much did NWS revise its forecast after 9 AM local?
        # Signal: agencies correcting their morning forecast predict actual follows revision.
        # nws_forecasts["timestamp"] is already naive local time (via _safe_parse_ts).
        nws_before_9am = nws_forecasts[nws_forecasts["timestamp"].dt.hour < 9]
        if not nws_before_9am.empty and len(nws_values) > 0:
            nws_at_9am = float(nws_before_9am.iloc[-1]["predicted_high"])
            features["nws_post_9am_delta"] = features["nws_last"] - nws_at_9am
        else:
            features["nws_post_9am_delta"] = np.nan  # no pre-9am forecast (D1-only days)

        # --- AccuWeather features (full parity with NWS) ---
        has_accu = not accu_forecasts.empty
        if has_accu:
            accu_values = accu_forecasts["predicted_high"].astype(float).values
            features["accu_first"] = accu_values[0] if len(accu_values) > 0 else np.nan
            features["accu_last"] = accu_values[-1] if len(accu_values) > 0 else np.nan
            features["accu_max"] = accu_values.max() if len(accu_values) > 0 else np.nan
            features["accu_min"] = accu_values.min() if len(accu_values) > 0 else np.nan
            features["accu_mean"] = accu_values.mean() if len(accu_values) > 0 else np.nan
            features["accu_spread"] = (accu_values.max() - accu_values.min()) if len(accu_values) > 0 else 0.0
            features["accu_std"] = float(accu_values.std()) if len(accu_values) > 1 else 0.0
            features["accu_trend"] = float(accu_values[-1] - accu_values[0]) if len(accu_values) > 1 else 0.0
            features["accu_count"] = len(accu_values)
        else:
            features["accu_first"] = np.nan
            features["accu_last"] = np.nan
            features["accu_max"] = np.nan
            features["accu_min"] = np.nan
            features["accu_mean"] = np.nan
            features["accu_spread"] = 0.0
            features["accu_std"] = 0.0
            features["accu_trend"] = 0.0
            features["accu_count"] = 0

        # --- Intraday revision feature (AccuWeather) ---
        if has_accu and not accu_forecasts.empty:
            accu_before_9am = accu_forecasts[accu_forecasts["timestamp"].dt.hour < 9]
            if not accu_before_9am.empty:
                accu_at_9am = float(accu_before_9am.iloc[-1]["predicted_high"])
                features["accu_post_9am_delta"] = features["accu_last"] - accu_at_9am
            else:
                features["accu_post_9am_delta"] = np.nan
        else:
            features["accu_post_9am_delta"] = np.nan

        # --- Cross-source features ---
        if not np.isnan(features["nws_last"]) and not np.isnan(features["accu_last"]):
            features["nws_accu_spread"] = abs(features["nws_last"] - features["accu_last"])
        else:
            features["nws_accu_spread"] = 0.0

        if not np.isnan(features["nws_mean"]) and not np.isnan(features.get("accu_mean", np.nan)):
            features["nws_accu_mean_diff"] = features["nws_mean"] - features["accu_mean"]
        else:
            features["nws_accu_mean_diff"] = 0.0

        # --- Temporal features ---
        features["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
        features["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)
        features["month"] = month
        features["is_summer"] = int(month in [6, 7, 8])
        features["is_winter"] = int(month in [12, 1, 2])

        # --- Rolling bias (placeholder, computed in build_feature_matrix) ---
        features["rolling_bias_7d"] = 0.0
        features["rolling_bias_21d"] = 0.0
        features["rolling_ml_error_7d"] = 0.0

        # --- Data availability flag ---
        features["has_accu_data"] = int(has_accu)

        # --- Overnight carryover detection features ---
        # prev_day_high: yesterday's actual high (passed in from build_feature_matrix)
        features["prev_day_high"] = float(prev_day_high) if prev_day_high is not None else np.nan
        # prev_day_temp_drop: large positive = overnight carryover risk
        if prev_day_high is not None and not np.isnan(features["nws_last"]):
            features["prev_day_temp_drop"] = float(prev_day_high) - features["nws_last"]
        else:
            features["prev_day_temp_drop"] = np.nan
        # midnight_temp: filled later from atmospheric data merge
        features["midnight_temp"] = np.nan

        # --- MOS max temp — NaN for training data (not available in archive) ---
        features["mos_max_temp"] = np.nan

        # --- Observation proxy features — NaN here, filled after atmospheric merge ---
        # For recent dates, the atmospheric merge brings intraday temps from
        # atmospheric_data.csv. After merge, we compute obs proxy features
        # in _compute_observation_proxy_features().
        from model_config import OBSERVATION_COLS
        for col in OBSERVATION_COLS:
            features[col] = np.nan

        # --- Target variables (not features, used for training) ---
        features["actual_high"] = actual_high
        features["winning_bucket"] = f"{int(actual_high)}-{int(actual_high)+1}"
        features["target_date"] = target_date

        return features

    @staticmethod
    def _calculate_velocity(values):
        if len(values) < 2:
            return 0.0
        return float(np.diff(values).mean())

    @staticmethod
    def _calculate_acceleration(values):
        if len(values) < 3:
            return 0.0
        diffs = np.diff(values)
        acc = np.diff(diffs)
        return float(acc.mean()) if len(acc) > 0 else 0.0

    def _compute_rolling_biases(self):
        """Compute rolling mean bias from prior days for each row (no data leakage)."""
        df = self.features_df.sort_values("target_date").reset_index(drop=True)

        # Per-day bias: actual_high - nws_mean
        # For multi-year rows where nws_mean is NaN, daily_bias will be NaN too
        daily_bias = (df["actual_high"] - df["nws_mean"]).values

        rolling_7 = []
        rolling_21 = []
        for i in range(len(df)):
            # Strictly prior days only, skip NaN biases
            prior_7 = [b for b in daily_bias[max(0, i - 7):i]
                        if not np.isnan(b)]
            rolling_7.append(float(np.mean(prior_7)) if prior_7 else 0.0)
            prior_21 = [b for b in daily_bias[max(0, i - 21):i]
                         if not np.isnan(b)]
            rolling_21.append(float(np.mean(prior_21)) if prior_21 else 0.0)

        df["rolling_bias_7d"] = rolling_7
        df["rolling_bias_21d"] = rolling_21

        # rolling_ml_error_7d: requires historical ml_f from Supabase
        # Falls back to 0.0 if not available (model handles gracefully)
        try:
            import os
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY")
            if supabase_url and supabase_key:
                from supabase import create_client
                sb = create_client(supabase_url, supabase_key)
                city_key = getattr(self, 'city_key', 'nyc')
                resp = sb.table("prediction_logs").select("target_date,ml_f").eq("city", city_key).in_("lead_used", ["today_for_today", "D0"]).not_.is_("ml_f", "null").execute()
                ml_pred_map = {}
                for row in (resp.data or []):
                    d = str(row.get("target_date", ""))[:10]
                    mf = row.get("ml_f")
                    if d and mf is not None:
                        try:
                            ml_pred_map[d] = float(mf)
                        except (TypeError, ValueError):
                            pass

                actual_map = dict(zip(df["target_date"].astype(str), df["actual_high"].values))
                rolling_ml_err = []
                dates_list = df["target_date"].astype(str).tolist()
                for i, d in enumerate(dates_list):
                    prior_errors = []
                    for j in range(max(0, i-14), i):
                        pd_ = dates_list[j]
                        mf = ml_pred_map.get(pd_)
                        ah = actual_map.get(pd_)
                        if mf is not None and ah is not None and not np.isnan(ah):
                            prior_errors.append(float(ah) - float(mf))
                    recent_7 = prior_errors[-7:] if len(prior_errors) >= 1 else []
                    rolling_ml_err.append(float(np.mean(recent_7)) if recent_7 else 0.0)
                df["rolling_ml_error_7d"] = rolling_ml_err
            else:
                df["rolling_ml_error_7d"] = 0.0
        except Exception as e:
            print(f"  ⚠️ rolling_ml_error_7d fallback to 0: {e}")
            df["rolling_ml_error_7d"] = 0.0
        self.features_df = df

    def build_feature_matrix(self):
        print("\nExtracting features for all dates...")
        dates_with_actuals = (
            self.nws_df.loc[self.nws_df["forecast_or_actual"] == "actual", "cli_date"]
            .dropna()
            .tolist()
        )

        # clean and sort by actual date
        dates_with_actuals = sorted(
            {d for d in dates_with_actuals if isinstance(d, str) and d and d != "<NA>"},
            key=lambda d: pd.to_datetime(d, errors="coerce"),
        )

        # Build a lookup of actual highs by date for prev_day_high computation
        actual_lookup = {}
        for _, row in self.nws_df[self.nws_df["forecast_or_actual"] == "actual"].iterrows():
            cli_d = row.get("cli_date")
            ah = row.get("actual_high")
            try:
                if cli_d and not pd.isna(ah):
                    actual_lookup[str(cli_d)] = float(ah)
            except (ValueError, TypeError):
                pass

        features_list = []
        for date in dates_with_actuals:
            # Look up previous day's actual high
            try:
                prev_dt = datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
                prev_date_str = prev_dt.strftime("%Y-%m-%d")
                prev_high = actual_lookup.get(prev_date_str)
            except (ValueError, TypeError):
                prev_high = None

            feat = self.extract_features_for_date(date, prev_day_high=prev_high)
            if feat:
                features_list.append(feat)

        if not features_list:
            print("  No features from forecast data. Will rely on multi-year data for v2.")
            self.features_df = pd.DataFrame()
            return

        self.features_df = pd.DataFrame(features_list)

        # Fill NaN AccuWeather features with NWS equivalents
        for accu_col, nws_col in ACCU_NWS_FALLBACK.items():
            if accu_col in self.features_df.columns and nws_col in self.features_df.columns:
                self.features_df[accu_col] = self.features_df[accu_col].fillna(
                    self.features_df[nws_col]
                )
        # Fill remaining NaN columns with defaults
        for col in ["accu_spread", "accu_std", "accu_trend", "nws_accu_spread", "nws_accu_mean_diff"]:
            if col in self.features_df.columns:
                self.features_df[col] = self.features_df[col].fillna(0.0)

        # Compute rolling biases from prior days
        self._compute_rolling_biases()

        # Integrity check
        missing_any = [c for c in FEATURE_COLS if c not in self.features_df.columns]
        if missing_any:
            raise ValueError(f"Feature matrix missing cols: {', '.join(missing_any)}")

        print(f"Built features for {len(self.features_df)} days")
        print(f"  Date range: {self.features_df['target_date'].min()} to {self.features_df['target_date'].max()}")
        print(f"  Days with AccuWeather data: {int(self.features_df['has_accu_data'].sum())}")
        return self.features_df

    def train_temperature_model(self):
        """
        Train a bias-correction model that corrects the BEST available forecast.

        Base = accu_last when AccuWeather data exists, else nws_last.
        Target = actual_high - base (small correction).
        Final prediction = base + model.predict(features).

        AccuWeather is the stronger source (47% bucket accuracy vs NWS 22%),
        so correcting AccuWeather gives the model a much better starting point.
        The 'has_accu_data' feature lets the model learn different correction
        patterns for each regime.
        """
        X = self.features_df[FEATURE_COLS]
        y_actual = self.features_df["actual_high"]
        nws_last = self.features_df["nws_last"]
        accu_last = self.features_df["accu_last"]

        # Best-available base: AccuWeather when present, NWS fallback
        base = accu_last.copy()
        base[base.isna()] = nws_last[base.isna()]
        self.features_df["_base"] = base  # stash for save_models

        y_bias = y_actual - base

        n_accu = int((~accu_last.isna()).sum())
        n_nws_only = int(accu_last.isna().sum())
        print(f"\nTraining bias-correction model (target = actual - best_base)...")
        print(f"  Base source: AccuWeather for {n_accu} days, NWS fallback for {n_nws_only} days")
        print(f"  Bias stats: mean={y_bias.mean():.2f}, std={y_bias.std():.2f}, "
              f"min={y_bias.min():.1f}, max={y_bias.max():.1f}")

        # Baseline comparisons
        baseline_nws_mae = float((y_actual - nws_last).abs().mean())
        baseline_best_mae = float((y_actual - base).abs().mean())
        baseline_nws_bucket = sum(int(n) == int(a) for n, a in zip(nws_last, y_actual)) / len(y_actual)
        baseline_best_bucket = sum(int(b) == int(a) for b, a in zip(base, y_actual)) / len(y_actual)

        print(f"  Baseline MAE (nws_last):      {baseline_nws_mae:.2f}°F ({baseline_nws_bucket:.0%} bucket)")
        print(f"  Baseline MAE (best_base):     {baseline_best_mae:.2f}°F ({baseline_best_bucket:.0%} bucket)")

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores = []
        bucket_acc_scores = []
        all_residuals = []

        for tr, te in tscv.split(X):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X.iloc[te])

            # Final prediction = base + predicted_bias
            pred_temp = base.iloc[te].values + pred_bias

            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())

            # Bucket accuracy: does floor(pred) == floor(actual)?
            pred_buckets = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc_scores.append(correct / len(actual_buckets))

        self.cv_mae_scores = mae_scores
        self.cv_bucket_acc_scores = bucket_acc_scores
        self.residual_std = float(np.std(all_residuals))

        print(f"  CV MAE (bias-corrected):      {np.mean(mae_scores):.2f}°F (folds: {[f'{s:.2f}' for s in mae_scores]})")
        print(f"  CV Bucket Accuracy:           {np.mean(bucket_acc_scores):.1%}")
        print(f"  Residual Std (for buckets):   {self.residual_std:.2f}°F")

        # Train final model on all data
        self.temp_model = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        self.temp_model.fit(X, y_bias)

        # Per-season diagnostics
        print("\n  Per-season diagnostics:")
        all_pred_bias = self.temp_model.predict(X)
        all_pred_temp = base.values + all_pred_bias
        seasons = {"summer": [6, 7, 8], "fall": [9, 10, 11], "winter": [12, 1, 2], "spring": [3, 4, 5]}
        for name, months in seasons.items():
            mask = self.features_df["month"].isin(months)
            n = int(mask.sum())
            if n > 0:
                season_mae = mean_absolute_error(y_actual[mask], all_pred_temp[mask])
                season_baseline = float((y_actual[mask] - base[mask]).abs().mean())
                print(f"    {name}: {season_mae:.2f}°F (baseline: {season_baseline:.2f}°F, {n} days)")

        return self.temp_model

    def save_models(self):
        print(f"\nSaving models (prefix='{self.model_prefix}')...")

        # Save temperature model
        with open(f"{self.model_prefix}temp_model.pkl", "wb") as f:
            pickle.dump(self.temp_model, f)

        # Save bucket info (residual_std for Gaussian bucket derivation)
        bucket_info = {
            "residual_std": self.residual_std,
            "method": "gaussian_from_regression",
        }
        with open(f"{self.model_prefix}bucket_model.pkl", "wb") as f:
            pickle.dump(bucket_info, f)

        # In-sample metrics: model predicts bias, final temp = base + bias
        X_all = self.features_df[FEATURE_COLS]
        y = self.features_df["actual_high"]
        base = self.features_df["_base"]  # best-available (accu_last or nws_last)
        pred_bias_all = self.temp_model.predict(X_all)
        temp_pred_all = base.values + pred_bias_all
        insample_mae = float(mean_absolute_error(y, temp_pred_all))

        # In-sample bucket accuracy
        pred_buckets = [f"{int(p)}-{int(p)+1}" for p in temp_pred_all]
        actual_buckets = self.features_df["winning_bucket"].tolist()
        insample_bucket_acc = sum(1 for p, a in zip(pred_buckets, actual_buckets) if p == a) / len(actual_buckets)

        # Per-season metrics
        seasonal = {}
        seasons = {"summer": [6, 7, 8], "fall": [9, 10, 11], "winter": [12, 1, 2], "spring": [3, 4, 5]}
        for name, months in seasons.items():
            mask = self.features_df["month"].isin(months)
            n = int(mask.sum())
            if n > 0:
                seasonal[name] = {
                    "days": n,
                    "insample_mae": round(float(mean_absolute_error(y[mask], temp_pred_all[mask])), 2),
                }

        # Feature importances via permutation importance (on bias target)
        y_bias = y - base
        perm_result = permutation_importance(
            self.temp_model, X_all, y_bias, n_repeats=10, random_state=42,
        )
        importances = dict(zip(FEATURE_COLS, perm_result.importances_mean))
        top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]

        metadata = {
            "trained_on": datetime.now().isoformat(),
            "num_days": int(len(self.features_df)),
            "date_range": {
                "start": str(self.features_df["target_date"].min()),
                "end": str(self.features_df["target_date"].max()),
            },
            "feature_columns": FEATURE_COLS,
            "residual_std": self.residual_std,
            "bucket_method": "gaussian_from_regression",
            "base_source": "accu_last (fallback: nws_last)",
            "model_type": "HistGradientBoostingRegressor",
            "model_performance": {
                "cv_temperature_mae": round(float(np.mean(self.cv_mae_scores)), 2),
                "cv_bucket_accuracy": round(float(np.mean(self.cv_bucket_acc_scores)), 4),
                "insample_temperature_mae": round(insample_mae, 2),
                "insample_bucket_accuracy": round(insample_bucket_acc, 4),
                "residual_std": round(self.residual_std, 2),
            },
            "seasonal_performance": seasonal,
            "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_features],
            "hyperparameters": {
                "max_iter": 300,
                "max_depth": 3,
                "learning_rate": 0.03,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "max_leaf_nodes": 15,
                "cv_splits": 5,
            },
        }

        with open(f"{self.model_prefix}model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("Models saved successfully!")
        print(f"  CV Temperature MAE: {metadata['model_performance']['cv_temperature_mae']:.2f}°F")
        print(f"  CV Bucket Accuracy: {metadata['model_performance']['cv_bucket_accuracy']:.1%}")
        print(f"  In-sample MAE: {insample_mae:.2f}°F")
        print(f"  Residual Std: {self.residual_std:.2f}°F")
        print(f"\n  Top features: {', '.join(n for n, _ in top_features[:5])}")

    # ═══════════════════════════════════════════════════════════════════
    # v2 training: atmospheric features + bucket classifier
    # ═══════════════════════════════════════════════════════════════════

    def _load_multiyear_data(self) -> pd.DataFrame | None:
        """
        Load multi-year atmospheric data for training expansion.

        These rows have actual_high + atmospheric features but NO forecast data.
        All forecast features (nws_*, accu_*) will be NaN — HistGradientBoosting
        handles this natively.
        """
        prefix = self.model_prefix
        csv_path = f"{prefix}multiyear_atmospheric.csv"
        try:
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} multi-year rows from {csv_path}")
            return df
        except FileNotFoundError:
            print(f"  No multi-year data file: {csv_path}")
            print(f"     Run: python backfill_multiyear.py --city {self.city_key}")
            return None

    def _build_multiyear_features(self, multiyear_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert multi-year atmospheric data into feature rows compatible with
        features_df format.

        Key innovation: PERSISTENCE FORECAST as proxy.
        For each historical day, yesterday's actual high serves as a "forecast."
        This gives the classifier real forecast→actual training pairs (1,277 of them)
        instead of skipping these rows entirely.

        The model learns atmospheric patterns that cause forecasts to miss —
        the same patterns apply whether the forecast is from NWS or persistence.
        """
        from model_config import (
            ATMOSPHERIC_COLS, ENSEMBLE_COLS, MULTIMODEL_COLS, INTRADAY_CURVE_COLS,
            MOS_COLS, OBSERVATION_COLS,
        )

        # Sort by date so we can compute persistence (yesterday's high)
        multiyear_df = multiyear_df.sort_values("target_date").reset_index(drop=True)

        rows = []
        prev_actual_high = None

        for _, row in multiyear_df.iterrows():
            target_date = str(row["target_date"])
            actual_high = float(row["actual_high"])

            try:
                dt = datetime.strptime(target_date, "%Y-%m-%d")
            except ValueError:
                prev_actual_high = actual_high
                continue

            doy = dt.timetuple().tm_yday
            month = dt.month

            # Persistence forecast: yesterday's actual high
            persistence = prev_actual_high
            prev_actual_high = actual_high

            if persistence is None:
                # First day in the dataset — no "yesterday" to use
                continue

            features = {
                # Forecast features — NaN for multi-year data
                # (no real NWS/AccuWeather forecasts exist)
                "nws_first": np.nan, "nws_last": np.nan,
                "nws_max": np.nan, "nws_min": np.nan, "nws_mean": np.nan,
                "nws_spread": np.nan, "nws_std": np.nan, "nws_trend": np.nan,
                "nws_count": np.nan,
                "forecast_velocity": np.nan, "forecast_acceleration": np.nan,
                "accu_first": np.nan, "accu_last": np.nan,
                "accu_max": np.nan, "accu_min": np.nan, "accu_mean": np.nan,
                "accu_spread": np.nan, "accu_std": np.nan, "accu_trend": np.nan,
                "accu_count": np.nan,
                "nws_accu_spread": np.nan, "nws_accu_mean_diff": np.nan,
                "rolling_bias_7d": np.nan, "rolling_bias_21d": np.nan,
                "rolling_ml_error_7d": 0.0,
                "has_accu_data": 0,

                # Overnight carryover detection features
                "prev_day_high": persistence,  # same as persistence forecast for multiyear
                "prev_day_temp_drop": np.nan,  # no NWS forecast for multiyear data
                "midnight_temp": row.get("midnight_temp", np.nan),

                # Persistence proxy forecast — used as center by the classifier
                "_persistence_forecast": persistence,

                # Intraday revision features — NaN for multiyear (no intraday forecast data)
                "nws_post_9am_delta": np.nan,
                "accu_post_9am_delta": np.nan,

                # Temporal features — computed from date
                "day_of_year_sin": np.sin(2 * np.pi * doy / 365),
                "day_of_year_cos": np.cos(2 * np.pi * doy / 365),
                "month": month,
                "is_summer": int(month in [6, 7, 8]),
                "is_winter": int(month in [12, 1, 2]),

                # Target
                "actual_high": actual_high,
                "winning_bucket": f"{int(actual_high)}-{int(actual_high)+1}",
                "target_date": target_date,
            }

            # Atmospheric features — from the multi-year data
            for col in ATMOSPHERIC_COLS:
                features[col] = row.get(col, np.nan)

            # Intraday curve features — from the multi-year data
            for col in INTRADAY_CURVE_COLS:
                features[col] = row.get(col, np.nan)

            # Ensemble — always NaN for historical (no ensemble archive)
            for col in ENSEMBLE_COLS:
                features[col] = np.nan

            # Multimodel — use values from CSV when backfilled (HRRR/GFS/ECMWF historical
            # forecast archive). Falls back to NaN for dates before backfill coverage.
            for col in MULTIMODEL_COLS:
                features[col] = row.get(col, np.nan)

            # MOS — NaN for historical (not available in archive)
            for col in MOS_COLS:
                features[col] = np.nan

            # Observation proxy features — computed from archive intraday data
            # Simulates what the model would see at noon during live inference.
            # For archive data, intraday features ARE real observations.
            obs_noon_temp = row.get("intra_temp_noon", np.nan)
            obs_9am_temp = row.get("intra_temp_9am", np.nan)
            obs_3pm_temp = row.get("intra_temp_3pm", np.nan)

            features["obs_latest_temp"] = obs_noon_temp if pd.notna(obs_noon_temp) else np.nan
            features["obs_latest_hour"] = 12.0  # fixed as-of hour for training
            # Running max up to noon: max of 9am and noon
            obs_temps = [t for t in [obs_9am_temp, obs_noon_temp] if pd.notna(t)]
            features["obs_max_so_far"] = max(obs_temps) if obs_temps else np.nan
            # 6hr max: max of morning observations (same as max_so_far for noon cutoff)
            features["obs_6hr_max"] = features["obs_max_so_far"]
            # Delta vs intraday forecast: ~0 for archive (same source)
            features["obs_vs_intra_forecast"] = 0.0

            # Wind from atmospheric features
            features["obs_wind_speed"] = row.get("atm_wind_mean", np.nan)
            features["obs_wind_gust"] = row.get("atm_wind_max", np.nan)
            features["obs_wind_dir_sin"] = row.get("atm_wind_dir_sin", np.nan)
            features["obs_wind_dir_cos"] = row.get("atm_wind_dir_cos", np.nan)

            # Cloud cover (archive has mean %, convert to 0-1 scale)
            cloud_mean = row.get("atm_cloud_cover_mean", np.nan)
            features["obs_cloud_cover"] = round(cloud_mean / 100.0, 2) if pd.notna(cloud_mean) else np.nan

            # Heating rate: (noon - 9am) / 3 hours
            if pd.notna(obs_noon_temp) and pd.notna(obs_9am_temp):
                features["obs_heating_rate"] = round((obs_noon_temp - obs_9am_temp) / 3.0, 2)
            else:
                features["obs_heating_rate"] = np.nan

            # Obs max vs persistence forecast (our proxy "NWS" forecast)
            if persistence is not None and features["obs_max_so_far"] is not None and not np.isnan(features["obs_max_so_far"]):
                features["obs_temp_vs_forecast_max"] = round(features["obs_max_so_far"] - persistence, 1)
            else:
                features["obs_temp_vs_forecast_max"] = np.nan

            rows.append(features)

        result = pd.DataFrame(rows)
        n_with_persistence = result["_persistence_forecast"].notna().sum() if len(result) > 0 else 0
        print(f"  Built {len(result)} multi-year feature rows "
              f"({n_with_persistence} with persistence forecast)")

        # Show persistence error distribution
        if len(result) > 0 and n_with_persistence > 0:
            pers = result["_persistence_forecast"]
            actual = result["actual_high"]
            errors = (actual - pers).dropna()
            print(f"  Persistence error: mean={errors.mean():.1f}°F, "
                  f"MAE={errors.abs().mean():.1f}°F, std={errors.std():.1f}°F")

            months = result["month"].value_counts().sort_index()
            season_counts = {
                "winter": int(result["is_winter"].sum()),
                "spring": int(result["month"].isin([3, 4, 5]).sum()),
                "summer": int(result["is_summer"].sum()),
                "fall": int(result["month"].isin([9, 10, 11]).sum()),
            }
            print(f"  Season distribution: {season_counts}")

        return result

    def _load_atmospheric_features(self) -> pd.DataFrame | None:
        """Load atmospheric features CSV if it exists."""
        prefix = self.model_prefix
        atm_csv = f"{prefix}atmospheric_data.csv"
        try:
            df = pd.read_csv(atm_csv)
            print(f"  Loaded {len(df)} atmospheric feature rows from {atm_csv}")
            return df
        except FileNotFoundError:
            print(f"  ⚠️ No atmospheric data file: {atm_csv}")
            print(f"     Run: python backfill_atmospheric.py --city {self.city_key}")
            return None

    def _merge_atmospheric_features(self, atm_df: pd.DataFrame) -> None:
        """Merge atmospheric features into self.features_df by target_date."""
        if atm_df is None or self.features_df is None:
            return

        # Ensure target_date is string in both
        atm_df["target_date"] = atm_df["target_date"].astype(str)

        # Drop non-feature columns from atmospheric data before merge
        drop_cols = ["city", "target_date"]
        atm_feature_cols = [c for c in atm_df.columns if c not in drop_cols]

        # Merge on target_date
        merged = self.features_df.merge(
            atm_df[["target_date"] + atm_feature_cols],
            on="target_date",
            how="left",
        )

        # Count how many atmospheric features were matched
        sample_col = atm_feature_cols[0] if atm_feature_cols else None
        if sample_col:
            matched = merged[sample_col].notna().sum()
            print(f"  Matched atmospheric data for {matched}/{len(merged)} days")

        self.features_df = merged

    def _load_observation_features(self) -> pd.DataFrame | None:
        """Load real NWS observation features CSV if it exists."""
        prefix = self.model_prefix
        obs_csv = f"{prefix}observation_data.csv"
        try:
            df = pd.read_csv(obs_csv)
            print(f"  Loaded {len(df)} real observation feature rows from {obs_csv}")
            non_zero = df["obs_vs_intra_forecast"].dropna()
            non_zero = non_zero[non_zero != 0]
            print(f"  obs_vs_intra_forecast: {len(non_zero)} non-zero values")
            return df
        except FileNotFoundError:
            print(f"  ⚠️ No observation data file: {obs_csv}")
            print(f"     Run: python prediction_writer.py --city {self.city_key} backfill_obs")
            return None

    def _merge_observation_features(self, obs_df: pd.DataFrame) -> None:
        """Merge REAL observation features into self.features_df, overwriting proxy values.

        For dates where real NWS observations exist in observation_data.csv,
        replaces the proxy obs features (obs_vs_intra_forecast=0) with real
        values (obs_vs_intra_forecast=actual delta). This is critical —
        the model needs real non-zero deltas to learn.
        """
        from model_config import OBSERVATION_COLS

        if obs_df is None or self.features_df is None:
            return

        obs_df["target_date"] = obs_df["target_date"].astype(str)

        # Safety check: deduplicate by target_date (keep last occurrence)
        dupes = obs_df["target_date"].duplicated(keep="last")
        if dupes.any():
            n_dupes = dupes.sum()
            print(f"  ⚠️ Removed {n_dupes} duplicate target_dates from observation_data.csv")
            obs_df = obs_df[~dupes].copy()

        # Only merge observation feature columns (not city, target_date)
        obs_cols_in_csv = [c for c in obs_df.columns if c in OBSERVATION_COLS]
        if not obs_cols_in_csv:
            print(f"  ⚠️ No observation feature columns found in CSV")
            return

        # For each date in obs_df, overwrite the proxy values in features_df
        obs_indexed = obs_df.set_index("target_date")
        dates_updated = 0
        for date_str in obs_indexed.index:
            mask = self.features_df["target_date"] == date_str
            if mask.sum() == 0:
                continue
            for col in obs_cols_in_csv:
                val = obs_indexed.loc[date_str, col]
                if pd.notna(val):
                    self.features_df.loc[mask, col] = val
            dates_updated += 1

        print(f"  Overwrote proxy obs features with real data for {dates_updated} dates")

    def _compute_observation_proxy_features(self) -> None:
        """Fill observation proxy features for recent data rows after atmospheric merge.

        For rows that came from extract_features_for_date() (recent NWS/AccuWeather data),
        the observation proxy columns were initialized as NaN. Now that atmospheric merge
        brought intraday features, we can compute the obs proxy from them.

        Simulates "as-of noon" observations using the archived intraday curve data.
        """
        df = self.features_df
        if df is None or df.empty:
            return

        # Only fill rows where obs_latest_temp is NaN (i.e., not already set by multiyear builder)
        mask = df["obs_latest_temp"].isna() & df["intra_temp_noon"].notna()
        if mask.sum() == 0:
            print(f"  Observation proxy: no rows to fill (all already set or no intraday data)")
            return

        # obs_latest_temp = noon temp
        df.loc[mask, "obs_latest_temp"] = df.loc[mask, "intra_temp_noon"]
        df.loc[mask, "obs_latest_hour"] = 12.0

        # obs_max_so_far: max of 9am and noon
        df.loc[mask, "obs_max_so_far"] = df.loc[mask, ["intra_temp_9am", "intra_temp_noon"]].max(axis=1)

        # obs_6hr_max: same as max_so_far for noon cutoff
        df.loc[mask, "obs_6hr_max"] = df.loc[mask, "obs_max_so_far"]

        # obs_vs_intra_forecast: ~0 for archive data (same source)
        df.loc[mask, "obs_vs_intra_forecast"] = 0.0

        # Wind from atmospheric features
        df.loc[mask, "obs_wind_speed"] = df.loc[mask, "atm_wind_mean"]
        df.loc[mask, "obs_wind_gust"] = df.loc[mask, "atm_wind_max"]
        df.loc[mask, "obs_wind_dir_sin"] = df.loc[mask, "atm_wind_dir_sin"]
        df.loc[mask, "obs_wind_dir_cos"] = df.loc[mask, "atm_wind_dir_cos"]

        # Cloud cover (atmospheric has %, convert to 0-1)
        cloud = df.loc[mask, "atm_cloud_cover_mean"]
        df.loc[mask, "obs_cloud_cover"] = (cloud / 100.0).round(2)

        # Heating rate: (noon - 9am) / 3
        noon = df.loc[mask, "intra_temp_noon"]
        am9 = df.loc[mask, "intra_temp_9am"]
        df.loc[mask, "obs_heating_rate"] = ((noon - am9) / 3.0).round(2)

        # obs_temp_vs_forecast_max: obs_max_so_far - nws_last
        obs_max = df.loc[mask, "obs_max_so_far"]
        nws_last = df.loc[mask, "nws_last"]
        df.loc[mask, "obs_temp_vs_forecast_max"] = (obs_max - nws_last).round(1)

        self.features_df = df
        filled = mask.sum()
        print(f"  Observation proxy: filled {filled} recent rows with as-of-noon features")

    def _train_atm_predictor(self) -> None:
        """
        Train first-stage atmospheric temperature predictor on historical data.

        Uses 1,278 multi-year days (atmospheric + temporal features only) to learn:
            atmospheric_conditions → actual daily high

        The predictor's output becomes 2 features for the classifier:
          - atm_predicted_high: what the atmosphere says the high should be
          - atm_vs_forecast_diff: NWS forecast - atm_predicted_high

        Trained on multi-year data ONLY (no forecast days), so predictions
        on the 232 forecast days are fully out-of-sample = no data leakage.
        """
        print(f"\n  Training atmospheric predictor (first-stage ML model)...")

        # Train on multi-year rows only (rows without any forecast data)
        multiyear_mask = self.features_df["nws_last"].isna() & self.features_df["accu_last"].isna()
        train_df = self.features_df[multiyear_mask].copy()

        if len(train_df) < 100:
            print(f"  ⚠️ Only {len(train_df)} multi-year rows — skipping atmospheric predictor")
            self.features_df["atm_predicted_high"] = np.nan
            self.features_df["atm_vs_forecast_diff"] = np.nan
            return

        # Ensure all input features exist
        for col in ATM_PREDICTOR_INPUT_COLS:
            if col not in self.features_df.columns:
                self.features_df[col] = np.nan

        X_train = train_df[ATM_PREDICTOR_INPUT_COLS]
        y_train = train_df["actual_high"]

        # Cross-validate on multi-year data to report quality
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        for tr, te in tscv.split(X_train):
            m = HistGradientBoostingRegressor(
                max_iter=200, max_depth=4, learning_rate=0.05,
                min_samples_leaf=15, l2_regularization=1.0,
                random_state=42,
            )
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            pred = m.predict(X_train.iloc[te])
            cv_maes.append(mean_absolute_error(y_train.iloc[te], pred))

        print(f"  Atmospheric predictor CV MAE: {np.mean(cv_maes):.2f}°F "
              f"(folds: {[f'{s:.2f}' for s in cv_maes]})")
        print(f"  Trained on {len(train_df)} historical days "
              f"({len(ATM_PREDICTOR_INPUT_COLS)} features)")

        # Train final model on all multi-year data
        atm_predictor = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0,
            random_state=42,
        )
        atm_predictor.fit(X_train, y_train)

        # Predict on ALL rows (forecast days are out-of-sample)
        X_all = self.features_df[ATM_PREDICTOR_INPUT_COLS]
        atm_preds = atm_predictor.predict(X_all)
        self.features_df["atm_predicted_high"] = atm_preds

        # Forecast divergence: NWS - atmospheric prediction
        # Positive = NWS predicts higher than atmosphere expects
        # Negative = NWS predicts lower than atmosphere expects
        nws_last = self.features_df["nws_last"]
        self.features_df["atm_vs_forecast_diff"] = nws_last - atm_preds

        # Show quality on forecast days (out-of-sample for atmospheric predictor)
        forecast_mask = self.features_df["nws_last"].notna()
        if forecast_mask.sum() > 0:
            fc_atm = atm_preds[forecast_mask]
            fc_actual = self.features_df.loc[forecast_mask, "actual_high"].values
            fc_mae = mean_absolute_error(fc_actual, fc_atm)
            print(f"  Out-of-sample MAE on forecast days: {fc_mae:.2f}°F")

            # How well does divergence predict forecast error?
            nws_vals = self.features_df.loc[forecast_mask, "nws_last"].values
            nws_errors = fc_actual - nws_vals  # positive = NWS was too low
            atm_diffs = nws_vals - fc_atm  # positive = NWS higher than atmosphere
            corr = np.corrcoef(atm_diffs, nws_errors)[0, 1] if len(atm_diffs) > 1 else 0.0
            print(f"  Divergence ↔ NWS error correlation: {corr:.3f}")
            if corr < -0.1:
                print(f"    → When NWS is higher than atmosphere, NWS tends to be too HIGH")
            elif corr > 0.1:
                print(f"    → When NWS is higher than atmosphere, actual tends to be HIGHER")

        # Save atmospheric predictor
        save_data = {
            "model": atm_predictor,
            "features": ATM_PREDICTOR_INPUT_COLS,
            "cv_mae": round(float(np.mean(cv_maes)), 2),
            "n_training_days": len(train_df),
        }
        with open(f"{self.model_prefix}atm_predictor.pkl", "wb") as f:
            pickle.dump(save_data, f)
        print(f"  Saved atmospheric predictor to {self.model_prefix}atm_predictor.pkl")

    def train_v2(self) -> None:
        """
        Train the v2 enhanced pipeline:
        1. Load and merge atmospheric features
        2. Train enhanced regression model with FEATURE_COLS_V2
        3. Train bucket classifier
        4. Save all models
        """
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v2 Training: Atmospheric Features + Bucket Classifier")
        print(f"{'═'*60}")

        # Load atmospheric features for existing dates
        if not self.features_df.empty:
            atm_df = self._load_atmospheric_features()
            if atm_df is not None:
                self._merge_atmospheric_features(atm_df)

        # Load multi-year historical data for training expansion
        multiyear_df = self._load_multiyear_data()
        if multiyear_df is not None:
            multiyear_features = self._build_multiyear_features(multiyear_df)
            if len(multiyear_features) > 0:
                # Exclude multi-year dates that overlap with existing data
                if not self.features_df.empty:
                    existing_dates = set(self.features_df["target_date"].astype(str).tolist())
                    multiyear_features = multiyear_features[
                        ~multiyear_features["target_date"].isin(existing_dates)
                    ]
                print(f"  After excluding overlaps: {len(multiyear_features)} new rows")

                # Concatenate with existing features
                original_count = len(self.features_df)
                self.features_df = pd.concat(
                    [self.features_df, multiyear_features],
                    ignore_index=True,
                )
                # Sort by date for proper temporal cross-validation
                self.features_df = self.features_df.sort_values("target_date").reset_index(drop=True)
                print(f"  Training data expanded: {original_count} → {len(self.features_df)} days")

        # Fill observation proxy features for recent rows (after atmospheric merge)
        self._compute_observation_proxy_features()

        # Determine which v2 columns are actually available
        available_v2_cols = [c for c in FEATURE_COLS_V2 if c in self.features_df.columns]
        missing_v2 = [c for c in FEATURE_COLS_V2 if c not in self.features_df.columns]
        if missing_v2:
            print(f"  Missing v2 columns (will be NaN): {missing_v2[:5]}{'...' if len(missing_v2) > 5 else ''}")
            # Add missing columns as NaN (HistGradientBoosting handles NaN)
            for col in missing_v2:
                self.features_df[col] = np.nan

        print(f"  Using {len(FEATURE_COLS_V2)} v2 features ({len(available_v2_cols)} with data, "
              f"{len(missing_v2)} NaN)")

        # --- Train enhanced regression model with v2 features ---
        # Regression only works on rows WITH forecast data (need a base for bias).
        # Multi-year rows (no forecasts) are excluded from regression but
        # included in classifier training below.
        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        n_total = len(self.features_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts (of {n_total} total)")

        residual_std_v2 = 2.0  # default
        self.features_df["_regression_pred"] = np.nan

        if n_forecast >= MIN_DAYS_FOR_TRAINING:
            X_v2_reg = forecast_df[FEATURE_COLS_V2]
            y_actual_reg = forecast_df["actual_high"]
            nws_last_reg = forecast_df["nws_last"]
            accu_last_reg = forecast_df["accu_last"]

            base_reg = accu_last_reg.copy()
            base_reg[base_reg.isna()] = nws_last_reg[base_reg.isna()]
            y_bias_reg = y_actual_reg - base_reg

            print(f"Training v2 regression model ({len(FEATURE_COLS_V2)} features, {n_forecast} rows)...")

            tscv = TimeSeriesSplit(n_splits=5)
            mae_scores_v2 = []
            bucket_acc_v2 = []
            all_residuals_v2 = []

            for tr, te in tscv.split(X_v2_reg):
                model = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=3, learning_rate=0.03,
                    min_samples_leaf=20, l2_regularization=1.0,
                    max_leaf_nodes=15, random_state=42,
                )
                model.fit(X_v2_reg.iloc[tr], y_bias_reg.iloc[tr])
                pred_bias = model.predict(X_v2_reg.iloc[te])
                pred_temp = base_reg.iloc[te].values + pred_bias

                mae_scores_v2.append(mean_absolute_error(y_actual_reg.iloc[te], pred_temp))
                all_residuals_v2.extend((y_actual_reg.iloc[te].values - pred_temp).tolist())

                pred_buckets = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
                actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual_reg.iloc[te]]
                correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
                bucket_acc_v2.append(correct / len(actual_buckets))

            residual_std_v2 = float(np.std(all_residuals_v2))
            print(f"  v2 Regression CV MAE: {np.mean(mae_scores_v2):.2f}°F "
                  f"(v1: {np.mean(self.cv_mae_scores):.2f}°F)")
            print(f"  v2 Regression CV Bucket Acc: {np.mean(bucket_acc_v2):.1%} "
                  f"(v1: {np.mean(self.cv_bucket_acc_scores):.1%})")
            print(f"  v2 Residual Std: {residual_std_v2:.2f}°F")

            # Train final v2 regression model on all forecast rows
            v2_regressor = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            v2_regressor.fit(X_v2_reg, y_bias_reg)

            # Add regression predictions for rows that have forecasts.
            self.features_df.loc[has_forecast_mask, "_regression_pred"] = (
                base_reg.values + v2_regressor.predict(X_v2_reg)
            )

            # Save v2 regression model
            with open(f"{self.model_prefix}temp_model_v2.pkl", "wb") as f:
                pickle.dump(v2_regressor, f)
            with open(f"{self.model_prefix}bucket_model_v2.pkl", "wb") as f:
                pickle.dump({
                    "residual_std": residual_std_v2,
                    "method": "gaussian_from_v2_regression",
                }, f)
            print(f"  Saved v2 regression model")
        else:
            print(f"  ⚠️  Only {n_forecast} forecast rows — skipping v2 regression.")
            print(f"     Classifier will train on {n_total} multi-year atmospheric rows.")

        # --- Train atmospheric predictor (first-stage model) ---
        # Uses ALL 1,278 historical days to learn atmosphere → temperature.
        # Its predictions become features for the classifier.
        self._train_atm_predictor()

        # --- Train bucket classifier ---
        # Train on days with real forecasts (NWS or AccuWeather).
        # Now includes atm_predicted_high and atm_vs_forecast_diff features
        # from the atmospheric predictor trained on 1,278 historical days.
        forecast_df = self.features_df[
            self.features_df["nws_last"].notna() | self.features_df["accu_last"].notna()
        ].copy().reset_index(drop=True)
        print(f"\n  Classifier training on {len(forecast_df)} forecast days "
              f"(excluding {len(self.features_df) - len(forecast_df)} no-forecast rows)")
        classifier = BucketClassifier()
        classifier.train(forecast_df, feature_cols=FEATURE_COLS_V2, residual_std=residual_std_v2)
        classifier.save(f"{self.model_prefix}bucket_classifier.pkl")

        # --- Save v2 metadata ---
        v2_metadata = {
            "trained_on": datetime.now().isoformat(),
            "version": "v2_atmospheric_classifier",
            "num_days": int(len(self.features_df)),
            "date_range": {
                "start": str(self.features_df["target_date"].min()),
                "end": str(self.features_df["target_date"].max()),
            },
            "v1_performance": {
                "cv_mae": round(float(np.mean(self.cv_mae_scores)), 2) if hasattr(self, 'cv_mae_scores') and self.cv_mae_scores else None,
                "cv_bucket_accuracy": round(float(np.mean(self.cv_bucket_acc_scores)), 4) if hasattr(self, 'cv_bucket_acc_scores') and self.cv_bucket_acc_scores else None,
            },
            "v2_regression": {
                "cv_mae": round(float(np.mean(mae_scores_v2)), 2) if 'mae_scores_v2' in dir() else None,
                "cv_bucket_accuracy": round(float(np.mean(bucket_acc_v2)), 4) if 'bucket_acc_v2' in dir() else None,
                "residual_std": round(residual_std_v2, 2),
                "num_features": len(FEATURE_COLS_V2),
            },
            "v2_classifier": classifier.training_stats,
            "feature_columns_v2": FEATURE_COLS_V2,
        }

        with open(f"{self.model_prefix}model_metadata_v2.json", "w") as f:
            json.dump(v2_metadata, f, indent=2)
        print(f"\n  Saved v2 metadata to {self.model_prefix}model_metadata_v2.json")

        # Print comparison summary
        print(f"\n{'─'*50}")
        if n_forecast >= MIN_DAYS_FOR_TRAINING:
            print(f"COMPARISON: v1 vs v2")
            print(f"{'─'*50}")
            print(f"  Regression MAE:       v1={np.mean(self.cv_mae_scores):.2f}°F → v2={np.mean(mae_scores_v2):.2f}°F")
            print(f"  Regression Bucket:    v1={np.mean(self.cv_bucket_acc_scores):.1%} → v2={np.mean(bucket_acc_v2):.1%}")
        else:
            print(f"v2 CLASSIFIER ONLY (no regression — insufficient forecast data)")
            print(f"{'─'*50}")
        print(f"  Classifier Bucket:    {classifier.cv_bucket_accuracy:.1%}")
        print(f"{'─'*50}")

    def train_v3(self) -> None:
        """
        Train v3 unified model: single regression on ALL data (forecast + multi-year).

        Key differences from v2:
        - ONE model predicts actual_high directly (not bias from a base)
        - Trained on ALL 1,540+ days (forecast days have full features,
          multi-year days have atmospheric + temporal, rest NaN)
        - No separate classifier — bucket probs come from Gaussian mapping
        - The model LEARNS when to trust forecasts vs atmospheric data
        - HistGradientBoosting handles NaN natively (multi-year rows)

        Output: temp_model_v3.pkl, model_metadata_v3.json
        """
        print(f"\n{'═'*60}")
        print(f"v3 Training: Unified Direct Regression")
        print(f"{'═'*60}")

        # v3 runs AFTER v2, so features_df already has atmospheric features
        # and multi-year data merged.  No need to reload.

        # Ensure all v3 columns exist
        for col in FEATURE_COLS_V3:
            if col not in self.features_df.columns:
                self.features_df[col] = np.nan

        # Also train the atmospheric predictor (needed at inference for
        # atm_predicted_high and atm_vs_forecast_diff features)
        self._train_atm_predictor()

        # Prepare training data — ALL rows with actual_high
        df = self.features_df[self.features_df["actual_high"].notna()].copy()
        n_total = len(df)
        n_forecast = df["nws_last"].notna().sum()
        n_multiyear = n_total - n_forecast
        print(f"\n  v3 unified training: {n_total} total days "
              f"({n_forecast} with forecasts, {n_multiyear} atmospheric-only)")

        X = df[FEATURE_COLS_V3]
        y = df["actual_high"]
        has_forecast = df["nws_last"].notna().values

        # Upweight forecast days so the model learns to use NWS/AccuWeather
        # when available, instead of always relying on atmospheric features.
        # Without this, 84% multi-year data dominates and forecast features
        # get ignored.
        forecast_weight = max(1, int(n_multiyear / max(n_forecast, 1)))
        sample_weights = np.where(has_forecast, forecast_weight, 1.0)
        print(f"  Sample weights: forecast days={forecast_weight}x, multi-year=1x")

        # Cross-validate with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        cv_bucket_accs = []
        all_residuals = []
        # Track forecast-days-only residuals for better residual_std
        forecast_residuals = []
        # Also track forecast-only MAE separately
        forecast_maes = []

        for tr, te in tscv.split(X):
            model = HistGradientBoostingRegressor(
                max_iter=400, max_depth=4, learning_rate=0.03,
                min_samples_leaf=15, l2_regularization=1.0,
                max_leaf_nodes=20, random_state=42,
            )
            model.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weights[tr])
            pred = model.predict(X.iloc[te])

            cv_maes.append(mean_absolute_error(y.iloc[te], pred))
            residuals = (y.iloc[te].values - pred)
            all_residuals.extend(residuals.tolist())

            # Track forecast-day residuals separately
            te_forecast_mask = has_forecast[te]
            if te_forecast_mask.any():
                forecast_residuals.extend(residuals[te_forecast_mask].tolist())
                forecast_maes.append(mean_absolute_error(
                    y.iloc[te].values[te_forecast_mask], pred[te_forecast_mask]
                ))

            # Bucket accuracy — compute separately for forecast days only
            te_forecast_pred = pred[te_forecast_mask]
            te_forecast_actual = y.iloc[te].values[te_forecast_mask]
            if len(te_forecast_pred) > 0:
                pred_buckets = [f"{int(round(p))}-{int(round(p))+1}" for p in te_forecast_pred]
                actual_buckets = [f"{int(round(a))}-{int(round(a))+1}" for a in te_forecast_actual]
                correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
                cv_bucket_accs.append(correct / len(actual_buckets))

        overall_mae = float(np.mean(cv_maes))
        forecast_mae = float(np.mean(forecast_maes)) if forecast_maes else overall_mae
        overall_bucket_acc = float(np.mean(cv_bucket_accs)) if cv_bucket_accs else 0
        # Use forecast-day residuals for residual_std (tighter, since at inference
        # we always have forecast data)
        residual_std = float(np.std(forecast_residuals)) if forecast_residuals else float(np.std(all_residuals))

        print(f"  v3 CV MAE (all): {overall_mae:.2f}°F")
        print(f"  v3 CV MAE (forecast days only): {forecast_mae:.2f}°F")
        print(f"  v3 CV 1°F Bucket Accuracy (forecast days): {overall_bucket_acc:.1%} "
              f"(per fold: {[f'{a:.1%}' for a in cv_bucket_accs]})")
        print(f"  v3 Residual Std (forecast days): {residual_std:.2f}°F")

        # Train final model on all data (with sample weights)
        v3_model = HistGradientBoostingRegressor(
            max_iter=400, max_depth=4, learning_rate=0.03,
            min_samples_leaf=15, l2_regularization=1.0,
            max_leaf_nodes=20, random_state=42,
        )
        v3_model.fit(X, y, sample_weight=sample_weights)

        # Feature importance
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(v3_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = sorted(
            zip(FEATURE_COLS_V3, perm.importances_mean),
            key=lambda x: x[1], reverse=True,
        )
        print(f"\n  Top 10 features:")
        for name, imp in importances[:10]:
            print(f"    {name:30s} {imp:.4f}")

        # Save model
        v3_data = {
            "model": v3_model,
            "features": FEATURE_COLS_V3,
            "residual_std": residual_std,
            "cv_mae": overall_mae,
            "cv_bucket_accuracy": overall_bucket_acc,
            "n_training_days": n_total,
            "n_forecast_days": int(n_forecast),
            "n_multiyear_days": int(n_multiyear),
        }
        with open(f"{self.model_prefix}temp_model_v3.pkl", "wb") as f:
            pickle.dump(v3_data, f)
        print(f"\n  Saved v3 model to {self.model_prefix}temp_model_v3.pkl")

        # Save metadata
        v3_metadata = {
            "trained_on": datetime.now().isoformat(),
            "version": "v3_unified_regression",
            "num_days": n_total,
            "num_forecast_days": int(n_forecast),
            "num_multiyear_days": int(n_multiyear),
            "date_range": {
                "start": str(df["target_date"].min()),
                "end": str(df["target_date"].max()),
            },
            "performance": {
                "cv_mae": round(overall_mae, 2),
                "cv_bucket_accuracy_1f": round(overall_bucket_acc, 4),
                "residual_std": round(residual_std, 2),
                "per_fold_mae": [round(m, 2) for m in cv_maes],
                "per_fold_bucket_acc": [round(a, 4) for a in cv_bucket_accs],
            },
            "top_features": [
                {"name": name, "importance": round(float(imp), 4)}
                for name, imp in importances[:15]
            ],
            "hyperparameters": {
                "max_iter": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_samples_leaf": 15,
                "l2_regularization": 1.0,
                "max_leaf_nodes": 20,
            },
            "feature_columns": FEATURE_COLS_V3,
        }
        with open(f"{self.model_prefix}model_metadata_v3.json", "w") as f:
            json.dump(v3_metadata, f, indent=2)
        print(f"  Saved v3 metadata to {self.model_prefix}model_metadata_v3.json")

    def train_v4(self) -> None:
        """
        Train v4 model: v2 architecture + 12 real-time observation features (84 total).

        Uses FEATURE_COLS_V4 which adds obs_latest_temp, obs_max_so_far, obs_6hr_max,
        obs_vs_intra_forecast, obs_wind_speed/gust/dir, obs_cloud_cover, obs_heating_rate,
        obs_temp_vs_forecast_max to the existing 72 v2 features.

        Observation proxy features were already computed in train_v2() via
        _compute_observation_proxy_features() and _build_multiyear_features().
        """
        from model_config import FEATURE_COLS_V4, OBSERVATION_COLS
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v4 Training: v2 + Real-Time Observation Features ({len(FEATURE_COLS_V4)} features)")
        print(f"{'═'*60}")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data available. Run train_v2() first.")
            return

        # Merge REAL observation features (overwrites proxy values where available)
        obs_df = self._load_observation_features()
        if obs_df is not None:
            self._merge_observation_features(obs_df)

        # Ensure all v4 columns exist (fill missing with NaN)
        available_v4_cols = [c for c in FEATURE_COLS_V4 if c in self.features_df.columns]
        missing_v4 = [c for c in FEATURE_COLS_V4 if c not in self.features_df.columns]
        if missing_v4:
            print(f"  Missing v4 columns (will be NaN): {missing_v4[:5]}{'...' if len(missing_v4) > 5 else ''}")
            for col in missing_v4:
                self.features_df[col] = np.nan

        # Check observation feature coverage
        obs_populated = 0
        for col in OBSERVATION_COLS:
            if col in self.features_df.columns:
                n = self.features_df[col].notna().sum()
                if n > 0:
                    obs_populated += 1
        print(f"  Observation features with data: {obs_populated}/{len(OBSERVATION_COLS)}")
        print(f"  Using {len(FEATURE_COLS_V4)} v4 features ({len(available_v4_cols)} with data, "
              f"{len(missing_v4)} NaN)")

        # --- Train v4 regression model ---
        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        n_total = len(self.features_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts (of {n_total} total)")

        residual_std_v4 = 2.0
        mae_scores_v4 = []
        bucket_acc_v4 = []

        if n_forecast >= MIN_DAYS_FOR_TRAINING:
            X_v4_reg = forecast_df[FEATURE_COLS_V4]
            y_actual_reg = forecast_df["actual_high"]
            nws_last_reg = forecast_df["nws_last"]
            accu_last_reg = forecast_df["accu_last"]

            base_reg = accu_last_reg.copy()
            base_reg[base_reg.isna()] = nws_last_reg[base_reg.isna()]
            y_bias_reg = y_actual_reg - base_reg

            print(f"  Training v4 regression model ({len(FEATURE_COLS_V4)} features, {n_forecast} rows)...")

            tscv = TimeSeriesSplit(n_splits=5)
            all_residuals_v4 = []

            for tr, te in tscv.split(X_v4_reg):
                model = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=3, learning_rate=0.03,
                    min_samples_leaf=20, l2_regularization=1.0,
                    max_leaf_nodes=15, random_state=42,
                )
                model.fit(X_v4_reg.iloc[tr], y_bias_reg.iloc[tr])
                pred_bias = model.predict(X_v4_reg.iloc[te])
                pred_temp = base_reg.iloc[te].values + pred_bias

                mae_scores_v4.append(mean_absolute_error(y_actual_reg.iloc[te], pred_temp))
                all_residuals_v4.extend((y_actual_reg.iloc[te].values - pred_temp).tolist())

                pred_buckets = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
                actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual_reg.iloc[te]]
                correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
                bucket_acc_v4.append(correct / len(actual_buckets))

            residual_std_v4 = float(np.std(all_residuals_v4))
            print(f"  v4 Regression CV MAE: {np.mean(mae_scores_v4):.2f}°F")
            print(f"  v4 Regression CV Bucket Acc: {np.mean(bucket_acc_v4):.1%}")
            print(f"  v4 Residual Std: {residual_std_v4:.2f}°F")

            # Train final v4 regression model on all forecast rows
            v4_regressor = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            v4_regressor.fit(X_v4_reg, y_bias_reg)

            # Save v4 regression model
            with open(f"{self.model_prefix}temp_model_v4.pkl", "wb") as f:
                pickle.dump(v4_regressor, f)
            with open(f"{self.model_prefix}bucket_model_v4.pkl", "wb") as f:
                pickle.dump({
                    "residual_std": residual_std_v4,
                    "method": "gaussian_from_v4_regression",
                }, f)
            print(f"  Saved v4 regression model")
        else:
            print(f"  ⚠️  Only {n_forecast} forecast rows — skipping v4 regression.")

        # --- Train v4 bucket classifier ---
        forecast_df = self.features_df[
            self.features_df["nws_last"].notna() | self.features_df["accu_last"].notna()
        ].copy().reset_index(drop=True)
        print(f"\n  Classifier training on {len(forecast_df)} forecast days "
              f"(excluding {len(self.features_df) - len(forecast_df)} no-forecast rows)")
        classifier = BucketClassifier()
        classifier.train(forecast_df, feature_cols=FEATURE_COLS_V4, residual_std=residual_std_v4)
        classifier.save(f"{self.model_prefix}bucket_classifier_v4.pkl")

        # --- Save v4 metadata ---
        v4_metadata = {
            "trained_on": datetime.now().isoformat(),
            "version": "v4_observation_features",
            "num_days": int(len(self.features_df)),
            "date_range": {
                "start": str(self.features_df["target_date"].min()),
                "end": str(self.features_df["target_date"].max()),
            },
            "v4_regression": {
                "cv_mae": round(float(np.mean(mae_scores_v4)), 2) if mae_scores_v4 else None,
                "cv_bucket_accuracy": round(float(np.mean(bucket_acc_v4)), 4) if bucket_acc_v4 else None,
                "residual_std": round(residual_std_v4, 2),
                "num_features": len(FEATURE_COLS_V4),
            },
            "v4_classifier": classifier.training_stats,
            "observation_features": OBSERVATION_COLS,
            "feature_columns_v4": FEATURE_COLS_V4,
        }

        with open(f"{self.model_prefix}model_metadata_v4.json", "w") as f:
            json.dump(v4_metadata, f, indent=2)
        print(f"\n  Saved v4 metadata to {self.model_prefix}model_metadata_v4.json")

        # Print comparison with v2
        print(f"\n{'─'*50}")
        print(f"COMPARISON: v2 vs v4")
        print(f"{'─'*50}")
        # Load v2 metadata for comparison
        try:
            with open(f"{self.model_prefix}model_metadata_v2.json") as f:
                v2_meta = json.load(f)
            v2_mae = v2_meta.get("v2_regression", {}).get("cv_mae")
            v2_bucket = v2_meta.get("v2_regression", {}).get("cv_bucket_accuracy")
            if v2_mae and mae_scores_v4:
                print(f"  Regression MAE:    v2={v2_mae:.2f}°F → v4={np.mean(mae_scores_v4):.2f}°F")
            if v2_bucket and bucket_acc_v4:
                print(f"  Regression Bucket: v2={v2_bucket:.1%} → v4={np.mean(bucket_acc_v4):.1%}")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        print(f"  v4 Classifier Bucket: {classifier.cv_bucket_accuracy:.1%}")
        print(f"{'─'*50}")

    def run(self, v2: bool = False, v4: bool = False):
        label = self.city_cfg["label"]
        print("=" * 60)
        print(f"{label} Temperature Model Training")
        print("=" * 60)
        self.load_data()

        # Check minimum data threshold
        actual_days = self.nws_df[
            (self.nws_df["forecast_or_actual"] == "actual")
            & self.nws_df["actual_high"].notna()
            & (self.nws_df["actual_high"] != "")
        ]["cli_date"].nunique()

        # Check if multi-year data can supplement insufficient forecast data
        multiyear_path = f"{self.model_prefix}multiyear_atmospheric.csv"
        has_multiyear = os.path.exists(multiyear_path)

        if actual_days < MIN_DAYS_FOR_TRAINING and not (v2 and has_multiyear):
            print(f"\n⚠️  Only {actual_days} days with actual data for {label}.")
            print(f"    Need at least {MIN_DAYS_FOR_TRAINING} days before training is viable. Skipping.")
            return

        if actual_days >= MIN_DAYS_FOR_TRAINING:
            self.build_feature_matrix()
            self.train_temperature_model()  # always train v1 (backward compat)
            self.save_models()
        else:
            print(f"\n⚠️  Only {actual_days} forecast days, but multi-year data available.")
            print(f"    Skipping v1 regression. Training v2 classifier on atmospheric data only.")
            # Build minimal feature matrix from what we have (even if sparse)
            self.build_feature_matrix()

        if v2:
            self.train_v2()
            # Also train v3 (unified model) using the same expanded data
            self.train_v3()

        if v4:
            if not v2:
                # v4 needs v2 to run first (builds feature matrix with atmospheric + obs proxy)
                self.train_v2()
                self.train_v3()
            self.train_v4()

        print("\nTraining complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train temperature prediction model")
    parser.add_argument("--city", default="nyc", help="City key (nyc, lax, etc.)")
    parser.add_argument("--all", action="store_true", help="Train all cities")
    parser.add_argument("--v2", action="store_true",
                        help="Also train v2 (atmospheric features + bucket classifier)")
    parser.add_argument("--v4", action="store_true",
                        help="Also train v4 (v2 + real-time observation features)")
    args = parser.parse_args()

    if args.all:
        from city_config import CITIES
        for city_key in CITIES:
            print(f"\n{'#' * 60}")
            print(f"# Training: {city_key}")
            print(f"{'#' * 60}\n")
            trainer = NYCTemperatureModelTrainer(city_key=city_key)
            trainer.run(v2=args.v2, v4=args.v4)
    else:
        trainer = NYCTemperatureModelTrainer(city_key=args.city)
        trainer.run(v2=args.v2, v4=args.v4)
