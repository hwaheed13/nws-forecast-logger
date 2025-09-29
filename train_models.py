# train_models.py
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "nws_first",
    "nws_last",
    "nws_max",
    "nws_min",
    "nws_mean",
    "nws_spread",
    "nws_std",
    "nws_trend",
    "nws_count",
    "forecast_velocity",
    "forecast_acceleration",
    "accu_last",
    "nws_accu_spread",
    "month",
    "is_summer",
    "is_winter",
]


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


class NYCTemperatureModelTrainer:
    def __init__(self):
        self.nws_df = None
        self.accu_df = None
        self.features_df = None
        self.temp_model = None
        self.bucket_model = None

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

        print(f"Loaded {len(self.nws_df)} NWS rows, {len(self.accu_df)} AccuWeather rows")

    def extract_features_for_date(self, target_date):
        nws_forecasts = self.nws_df[
            (self.nws_df["target_date"] == target_date)
            & (self.nws_df["forecast_or_actual"] == "forecast")
        ].copy()

        if not self.accu_df.empty:
            accu_forecasts = self.accu_df[
                (self.accu_df["target_date"] == target_date)
                & (self.accu_df["forecast_or_actual"] == "forecast")
            ].copy()
        else:
            accu_forecasts = pd.DataFrame()

        actual_row = self.nws_df[
            (self.nws_df["cli_date"] == target_date)
            & (self.nws_df["forecast_or_actual"] == "actual")
        ]

        if actual_row.empty or nws_forecasts.empty:
            return None

        actual_high = float(actual_row.iloc[0]["actual_high"])

        nws_forecasts = nws_forecasts.sort_values("timestamp")
        if not accu_forecasts.empty:
            accu_forecasts = accu_forecasts.sort_values("timestamp")

        nws_values = nws_forecasts["predicted_high"].astype(float).values
        month = int(target_date.split("-")[1])

        features = {
            "nws_first": nws_values[0] if len(nws_values) > 0 else np.nan,
            "nws_last": nws_values[-1] if len(nws_values) > 0 else np.nan,
            "nws_max": nws_values.max() if len(nws_values) > 0 else np.nan,
            "nws_min": nws_values.min() if len(nws_values) > 0 else np.nan,
            "nws_mean": nws_values.mean() if len(nws_values) > 0 else np.nan,
            "nws_median": float(np.median(nws_values)) if len(nws_values) > 0 else np.nan,
            "nws_count": int(len(nws_values)),
            "nws_spread": float(nws_values.max() - nws_values.min()) if len(nws_values) > 0 else 0.0,
            "nws_std": float(nws_values.std()) if len(nws_values) > 1 else 0.0,
            "nws_trend": float(nws_values[-1] - nws_values[0]) if len(nws_values) > 1 else 0.0,
            "forecast_velocity": self.calculate_velocity(nws_values),
            "forecast_acceleration": self.calculate_acceleration(nws_values),
            "accu_last": float(accu_forecasts.iloc[-1]["predicted_high"]) if not accu_forecasts.empty else np.nan,
            "accu_count": int(len(accu_forecasts)),
            "nws_accu_spread": np.nan,
            "month": month,
            "is_summer": int(month in [6, 7, 8]),
            "is_winter": int(month in [12, 1, 2]),
            "actual_high": actual_high,
            "winning_bucket": f"{int(actual_high)}-{int(actual_high)+1}",
            "target_date": target_date,
        }

        if not np.isnan(features["nws_last"]) and not np.isnan(features["accu_last"]):
            features["nws_accu_spread"] = abs(features["nws_last"] - features["accu_last"])

        return features

    def calculate_velocity(self, values):
        if len(values) < 2:
            return 0.0
        return float(np.diff(values).mean())

    def calculate_acceleration(self, values):
        if len(values) < 3:
            return 0.0
        diffs = np.diff(values)
        acc = np.diff(diffs)
        return float(acc.mean()) if len(acc) > 0 else 0.0

    def build_feature_matrix(self):
        print("\nExtracting features for all dates...")
        dates_with_actuals = (
            self.nws_df[self.nws_df["forecast_or_actual"] == "actual"]["cli_date"].unique()
        )
        features_list = []
        for date in sorted(dates_with_actuals):
            feat = self.extract_features_for_date(date)
            if feat:
                features_list.append(feat)

        if not features_list:
            raise ValueError("No features built. Ensure CSVs have forecasts and actuals.")

        self.features_df = pd.DataFrame(features_list)

        if "accu_last" in self.features_df.columns:
            self.features_df["accu_last"] = self.features_df["accu_last"].fillna(self.features_df["nws_last"])
        self.features_df["nws_accu_spread"] = self.features_df["nws_accu_spread"].fillna(0.0)

        missing_any = [c for c in FEATURE_COLS if c not in self.features_df.columns]
        if missing_any:
            raise ValueError(f"Feature matrix missing cols: {', '.join(missing_any)}")

        print(f"Built features for {len(self.features_df)} days")
        return self.features_df

    def train_temperature_model(self):
        print("\nTraining temperature prediction model...")
        X = self.features_df[FEATURE_COLS]
        y = self.features_df["actual_high"]

        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores = []
        for tr, te in tscv.split(X):
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            mae_scores.append(mean_absolute_error(y.iloc[te], pred))

        print(f"Cross-validation MAE: {np.mean(mae_scores):.2f}°F")
        self.temp_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.temp_model.fit(X, y)
        return self.temp_model

    def train_bucket_model(self):
        print("\nTraining Kalshi bucket prediction model...")
        X = self.features_df[FEATURE_COLS]
        y = self.features_df["winning_bucket"]

        tscv = TimeSeriesSplit(n_splits=3)
        acc_scores = []
        for tr, te in tscv.split(X):
            model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            acc_scores.append(accuracy_score(y.iloc[te], pred))

        print(f"Cross-validation accuracy: {np.mean(acc_scores):.1%}")
        self.bucket_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.bucket_model.fit(X, y)
        return self.bucket_model

    def save_models(self):
        print("\nSaving models...")
        with open("temp_model.pkl", "wb") as f:
            pickle.dump(self.temp_model, f)
        with open("bucket_model.pkl", "wb") as f:
            pickle.dump(self.bucket_model, f)

        X_all = self.features_df[FEATURE_COLS]
        temp_pred_all = self.temp_model.predict(X_all)
        bucket_pred_all = self.bucket_model.predict(X_all)

        metadata = {
            "trained_on": datetime.now().isoformat(),
            "num_days": int(len(self.features_df)),
            "date_range": {
                "start": str(self.features_df["target_date"].min()),
                "end": str(self.features_df["target_date"].max()),
            },
            "feature_columns": FEATURE_COLS,
            "model_performance": {
                "temperature_mae": float(mean_absolute_error(self.features_df["actual_high"], temp_pred_all)),
                "bucket_accuracy": float(accuracy_score(self.features_df["winning_bucket"], bucket_pred_all)),
            },
        }

        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("Models saved successfully!")
        print(f"Temperature MAE: {metadata['model_performance']['temperature_mae']:.2f}°F")
        print(f"Bucket accuracy: {metadata['model_performance']['bucket_accuracy']:.1%}")

    def run(self):
        print("=" * 50)
        print("NYC Temperature Model Training")
        print("=" * 50)
        self.load_data()
        self.build_feature_matrix()
        self.train_temperature_model()
        self.train_bucket_model()
        self.save_models()
        print("\nTraining complete.")


if __name__ == "__main__":
    trainer = NYCTemperatureModelTrainer()
    trainer.run()
