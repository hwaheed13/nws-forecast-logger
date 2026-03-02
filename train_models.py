# train_models.py
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from model_config import FEATURE_COLS, ACCU_NWS_FALLBACK

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


class NYCTemperatureModelTrainer:
    def __init__(self):
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
        print("Loading CSV files...")
        # NWS file required
        self.nws_df = pd.read_csv("nws_forecast_log.csv")
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
            "nws_forecast_log.csv",
        )

        # AccuWeather optional
        try:
            self.accu_df = pd.read_csv("accuweather_log.csv")
            if not self.accu_df.empty:
                self._require(
                    self.accu_df,
                    ["timestamp", "forecast_or_actual", "predicted_high", "target_date"],
                    "accuweather_log.csv",
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

    def extract_features_for_date(self, target_date: str):
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

        # --- Data availability flag ---
        features["has_accu_data"] = int(has_accu)

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
        daily_bias = (df["actual_high"] - df["nws_mean"]).values

        rolling_7 = []
        rolling_21 = []
        for i in range(len(df)):
            # Strictly prior days only
            prior_7 = daily_bias[max(0, i - 7):i]
            rolling_7.append(float(np.mean(prior_7)) if len(prior_7) > 0 else 0.0)
            prior_21 = daily_bias[max(0, i - 21):i]
            rolling_21.append(float(np.mean(prior_21)) if len(prior_21) > 0 else 0.0)

        df["rolling_bias_7d"] = rolling_7
        df["rolling_bias_21d"] = rolling_21
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

        features_list = []
        for date in dates_with_actuals:
            feat = self.extract_features_for_date(date)
            if feat:
                features_list.append(feat)

        if not features_list:
            raise ValueError("No features built. Check that your CSVs contain actuals and forecasts.")

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
        print("\nSaving models...")

        # Save temperature model
        with open("temp_model.pkl", "wb") as f:
            pickle.dump(self.temp_model, f)

        # Save bucket info (residual_std for Gaussian bucket derivation)
        bucket_info = {
            "residual_std": self.residual_std,
            "method": "gaussian_from_regression",
        }
        with open("bucket_model.pkl", "wb") as f:
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

        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("Models saved successfully!")
        print(f"  CV Temperature MAE: {metadata['model_performance']['cv_temperature_mae']:.2f}°F")
        print(f"  CV Bucket Accuracy: {metadata['model_performance']['cv_bucket_accuracy']:.1%}")
        print(f"  In-sample MAE: {insample_mae:.2f}°F")
        print(f"  Residual Std: {self.residual_std:.2f}°F")
        print(f"\n  Top features: {', '.join(n for n, _ in top_features[:5])}")

    def run(self):
        print("=" * 60)
        print("NYC Temperature Model Training (v2 — full-year, Gaussian buckets)")
        print("=" * 60)
        self.load_data()
        self.build_feature_matrix()
        self.train_temperature_model()
        self.save_models()
        print("\nTraining complete.")


if __name__ == "__main__":
    trainer = NYCTemperatureModelTrainer()
    trainer.run()
