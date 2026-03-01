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

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except Exception:
    # ImportError if not installed; XGBoostError on Mac if libomp missing
    HAS_XGBOOST = False

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
    "accu_mean",
    "nws_accu_spread",
    "month",
    "day_of_year",
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
        self.bucket_model = None
        self.temp_model_name: str = "GradientBoosting"
        self.bucket_model_name: str = "RandomForest"

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

        # AccuWeather values
        accu_values = (
            accu_forecasts["predicted_high"].astype(float).values
            if not accu_forecasts.empty
            else np.array([])
        )

        month = int(target_date.split("-")[1])
        # day_of_year: captures smooth seasonal curve better than binary is_summer/is_winter
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            day_of_year = dt.timetuple().tm_yday
        except Exception:
            day_of_year = np.nan

        features = {
            "nws_first": nws_values[0] if len(nws_values) > 0 else np.nan,
            "nws_last": nws_values[-1] if len(nws_values) > 0 else np.nan,
            "nws_max": nws_values.max() if len(nws_values) > 0 else np.nan,
            "nws_min": nws_values.min() if len(nws_values) > 0 else np.nan,
            "nws_mean": nws_values.mean() if len(nws_values) > 0 else np.nan,
            "nws_median": float(np.median(nws_values)) if len(nws_values) > 0 else np.nan,
            "nws_count": len(nws_values),
            "nws_spread": (nws_values.max() - nws_values.min()) if len(nws_values) > 0 else 0.0,
            "nws_std": float(nws_values.std()) if len(nws_values) > 1 else 0.0,
            "nws_trend": float(nws_values[-1] - nws_values[0]) if len(nws_values) > 1 else 0.0,
            "forecast_velocity": self.calculate_velocity(nws_values),
            "forecast_acceleration": self.calculate_acceleration(nws_values),
            "accu_last": float(accu_values[-1]) if len(accu_values) > 0 else np.nan,
            "accu_mean": float(accu_values.mean()) if len(accu_values) > 0 else np.nan,
            "accu_count": int(len(accu_values)),
            "nws_accu_spread": np.nan,  # filled below if both present
            "month": month,
            "day_of_year": day_of_year,
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

        # Fill NaNs: AccuWeather missing → fall back to NWS equivalent
        if "accu_last" in self.features_df.columns:
            self.features_df["accu_last"] = self.features_df["accu_last"].fillna(
                self.features_df["nws_last"]
            )
        if "accu_mean" in self.features_df.columns:
            self.features_df["accu_mean"] = self.features_df["accu_mean"].fillna(
                self.features_df["nws_mean"]
            )
        self.features_df["nws_accu_spread"] = self.features_df["nws_accu_spread"].fillna(0.0)

        # Basic integrity check
        missing_any = [c for c in FEATURE_COLS if c not in self.features_df.columns]
        if missing_any:
            raise ValueError(f"Feature matrix missing cols: {', '.join(missing_any)}")

        print(f"Built features for {len(self.features_df)} days")
        return self.features_df

    def _cv_mae(self, model_factory, X, y, n_splits=3):
        """Run TimeSeriesSplit CV and return mean MAE."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, te in tscv.split(X):
            m = model_factory()
            m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(mean_absolute_error(y.iloc[te], m.predict(X.iloc[te])))
        return float(np.mean(scores))

    def _cv_acc(self, model_factory, X, y, n_splits=3):
        """Run TimeSeriesSplit CV and return mean accuracy."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, te in tscv.split(X):
            m = model_factory()
            m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
        return float(np.mean(scores))

    def train_temperature_model(self):
        print("\nTraining temperature prediction model...")
        X = self.features_df[FEATURE_COLS]
        y = self.features_df["actual_high"]

        candidates = {
            "GradientBoosting": lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
        }
        if HAS_XGBOOST:
            candidates["XGBoost"] = lambda: XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
        else:
            print("  (xgboost not installed — using GradientBoosting only)")

        best_name, best_mae, best_factory = None, float("inf"), None
        for name, factory in candidates.items():
            mae = self._cv_mae(factory, X, y)
            print(f"  {name} CV MAE: {mae:.3f}°F")
            if mae < best_mae:
                best_mae, best_name, best_factory = mae, name, factory

        print(f"  → Selected: {best_name} (CV MAE {best_mae:.3f}°F)")
        self.temp_model = best_factory()
        self.temp_model.fit(X, y)
        self.temp_model_name = best_name
        return self.temp_model

    def train_bucket_model(self):
        print("\nTraining Kalshi bucket prediction model...")
        X = self.features_df[FEATURE_COLS]
        y = self.features_df["winning_bucket"]

        candidates = {
            "RandomForest": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=6, random_state=42
            ),
        }
        if HAS_XGBOOST:
            candidates["XGBoost"] = lambda: XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, verbosity=0
            )

        best_name, best_acc, best_factory = None, -1.0, None
        for name, factory in candidates.items():
            acc = self._cv_acc(factory, X, y)
            print(f"  {name} CV accuracy: {acc:.1%}")
            if acc > best_acc:
                best_acc, best_name, best_factory = acc, name, factory

        print(f"  → Selected: {best_name} (CV accuracy {best_acc:.1%})")
        self.bucket_model = best_factory()
        self.bucket_model.fit(X, y)
        self.bucket_model_name = best_name
        return self.bucket_model

    def save_models(self):
        print("\nSaving models...")
        with open("temp_model.pkl", "wb") as f:
            pickle.dump(self.temp_model, f)
        with open("bucket_model.pkl", "wb") as f:
            pickle.dump(self.bucket_model, f)

        # In-sample sanity metrics
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
            "temp_model_type": self.temp_model_name,
            "bucket_model_type": self.bucket_model_name,
            "model_performance": {
                "temperature_mae": float(
                    mean_absolute_error(self.features_df["actual_high"], temp_pred_all)
                ),
                "bucket_accuracy": float(
                    accuracy_score(self.features_df["winning_bucket"], bucket_pred_all)
                ),
            },
        }

        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("Models saved successfully!")
        print(f"Temperature model: {self.temp_model_name}")
        print(f"Bucket model: {self.bucket_model_name}")
        print(f"Temperature MAE (in-sample): {metadata['model_performance']['temperature_mae']:.2f}°F")
        print(f"Bucket accuracy (in-sample): {metadata['model_performance']['bucket_accuracy']:.1%}")

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
