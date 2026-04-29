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


def _record_coverage(model_prefix: str, version: str, counts: dict) -> None:
    """
    Append per-version feature-coverage counts to {prefix}coverage_report.json.

    Each --vN training invocation runs in a fresh Python process, so we
    accumulate via the JSON file. The retrain workflow reads this file
    after training and fails the run if any tracked feature regressed
    from non-zero on `main` to zero in this run (e.g. the cli_date bug
    that silently dropped v13 entrainment from 1225 → 0).

    counts:  {feature_name: int_count}
    """
    path = f"{model_prefix}coverage_report.json"
    try:
        with open(path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        report = {}
    # Coerce to int (numpy int64 → int) so json is happy
    report[version] = {k: int(v) for k, v in counts.items()}
    report["_updated_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"  ↳ recorded coverage[{version}] → {path}")


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

    # ---------------------------------------------------------------- #
    # Airtight version-deployment gate                                  #
    #                                                                   #
    # A version's model can only deploy if it has enough forecast rows  #
    # where ALL of its new features are simultaneously populated.       #
    # If not, training is skipped (no .pkl written) and the cascade     #
    # in prediction_writer.py falls back to the previous version.       #
    #                                                                   #
    # WHY: HistGradientBoosting tolerates NaN natively. A version       #
    # whose new features are populated on <X% of training rows will     #
    # learn an aggressive split rule from the rare populated subset     #
    # and then over-fire that rule at inference (where features are     #
    # always live-computed and present). This is the documented cause   #
    # of the 2026-04 D+1 cold-bias regression on tomorrow predictions.  #
    # ---------------------------------------------------------------- #
    def _gate_and_filter_for_version(
        self,
        version: str,
        key_features: "list[str]",
        forecast_df: "pd.DataFrame",
        min_rows: int = 500,
    ):
        """Gate v_n training on coverage of key_features.

        Returns a filtered forecast_df (rows where ALL key_features are
        populated) on success, or None if below the min_rows threshold —
        in which case the caller MUST `return` early without writing any
        .pkl files. The cascade auto-falls-back to v_{n-1}.

        Filtering (not just gating) is intentional: it ensures the model
        trains on the same feature distribution it will see at inference
        time, eliminating the NaN-branch overfit pattern.
        """
        available = [c for c in key_features if c in forecast_df.columns]
        if not available:
            print(f"  ⛔ {version} GATE: none of {key_features} present in dataframe — SKIPPING.")
            print(f"     Cascade will fall back to prior version.")
            return None
        counts = {c: int(forecast_df[c].notna().sum()) for c in available}
        populated_mask = forecast_df[available].notna().all(axis=1)
        n_populated = int(populated_mask.sum())
        print(f"  {version} key-feature populated counts: {counts}")
        print(f"  {version} forecast rows with ALL key features populated: {n_populated}")
        if n_populated < min_rows:
            print(f"  ⛔ {version} GATE FAILED: only {n_populated} fully-populated rows "
                  f"< {min_rows} threshold.")
            print(f"     SKIPPING training. Cascade falls back to prior version.")
            print(f"     To deploy {version}: backfill or accumulate more rows where ALL "
                  f"{available} fire simultaneously.")
            # CRITICAL: remove any stale .pkl from a previous retrain so the
            # cascade in prediction_writer.py actually falls back. Without this,
            # FileNotFoundError can't fire because the old broken model still
            # sits on disk and gets loaded.
            import os as _os
            prefix = self.model_prefix
            for stale in (
                f"{prefix}bcp_{version}_regressor.pkl",
                f"{prefix}bcp_{version}_classifier.pkl",
                f"{prefix}bcp_{version}_feature_cols.pkl",
            ):
                try:
                    if _os.path.exists(stale):
                        _os.remove(stale)
                        print(f"     removed stale {stale}")
                except OSError as _exc:
                    print(f"     warning: could not remove {stale}: {_exc}")
            # Replace metadata with a skipped-marker so the audit trail is
            # clear and check_coverage_regression knows this version is
            # intentionally absent (not a regression).
            try:
                import json as _json
                from datetime import datetime as _dt
                with open(f"{prefix}model_metadata_{version}.json", "w") as _f:
                    _json.dump({
                        "trained_on": _dt.now().isoformat(),
                        "version": version,
                        "skipped": True,
                        "reason": (
                            f"airtight gate failed: only {n_populated} forecast rows "
                            f"have all of {available} populated (need ≥{min_rows})"
                        ),
                        "key_feature_counts": counts,
                        "min_rows_required": min_rows,
                    }, _f, indent=2)
            except OSError as _exc:
                print(f"     warning: could not write skipped-marker metadata: {_exc}")
            return None
        print(f"  ✓ {version} gate PASSED: training on {n_populated} fully-populated rows "
              f"(eliminates NaN-branch overfit).")
        return forecast_df[populated_mask].copy()

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

    def _load_supabase_snapshot_features(self) -> pd.DataFrame | None:
        """
        Load atmospheric feature snapshots directly from Supabase prediction_logs.

        These rows capture the EXACT features the model saw at inference time
        (Open-Meteo FORECAST data), which is higher quality than re-fetching
        from the Open-Meteo ARCHIVE (which returns actuals, not forecasts).

        Returns a DataFrame keyed by target_date with all atm_snapshot keys
        plus nws_last / accu_last / ml_actual_high, ready to merge into
        self.features_df as supplementary / override rows.
        """
        import os, json
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            # Previously this returned None and silently fell back to the CSV
            # path, which doesn't extract atm_snapshot JSONB-derived features.
            # That meant production retrains during a creds outage would
            # silently train on a narrower feature set than inference uses.
            # Hard-fail instead — if the workflow doesn't have creds, the
            # workflow is broken, not the model.
            if os.environ.get("ALLOW_SUPABASE_FALLBACK") == "1":
                print("  ⚠️ Supabase creds not set — ALLOW_SUPABASE_FALLBACK=1, "
                      "skipping snapshot path (CSV-only training, narrower features)")
                return None
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_SERVICE_ROLE not set. Refusing to "
                "train silently from CSVs without atm_snapshot features. "
                "Set ALLOW_SUPABASE_FALLBACK=1 to override."
            )
        try:
            from supabase import create_client
            sb = create_client(url, key)
            # Select ONLY columns that exist in the prediction_logs schema.
            # All forecast-related fields (nws_last, accu_last, rolling_bias_*,
            # nws_first, etc.) live inside atm_snapshot JSONB — not as top-level cols.
            # Also pull the 3 v13 BL safeguard top-level columns — these are
            # populated by backfill_v13_features.py / prediction_writer.py and
            # are NOT inside atm_snapshot. Without this, forecast-day training
            # rows get 0 BL feature coverage and v13 collapses to v12.
            resp = (
                sb.table("prediction_logs")
                .select("target_date,ml_actual_high,atm_snapshot,"
                        "entrainment_temp_diff,marine_containment,inland_strength")
                .eq("city", self.city_key)
                .in_("lead_used", ["today_for_today", "D0"])
                .not_.is_("atm_snapshot", "null")
                .not_.is_("ml_actual_high", "null")
                .execute()
            )
            rows = resp.data or []
            if not rows:
                print("  ℹ️  No scored Supabase snapshot rows yet — will grow daily")
                return None

            records = []
            bad_snap_rows = 0
            for r in rows:
                snap_raw = r.get("atm_snapshot")
                # Normalize to a dict. Supabase may hand back a list or null
                # for legacy rows — skip the JSONB merge for those instead of
                # crashing the whole load (which loses all 273 forecast rows).
                if isinstance(snap_raw, str):
                    try:
                        snap = json.loads(snap_raw)
                    except Exception:
                        snap = None
                else:
                    snap = snap_raw
                if not isinstance(snap, dict):
                    bad_snap_rows += 1
                    snap = {}
                record = {"target_date": str(r["target_date"])[:10]}
                record["actual_high"] = r.get("ml_actual_high")
                # All other training features come from atm_snapshot JSONB.
                # This includes nws_last, accu_last, nws_first, rolling_bias_*,
                # and all the Synoptic/atmospheric obs features.
                for k, v in snap.items():
                    if k not in record:
                        record[k] = v
                # Overlay top-level BL cols so they survive the merge into
                # features_df even if the JSONB lacks the source inputs.
                for bl_col in ("entrainment_temp_diff", "marine_containment", "inland_strength"):
                    bl_val = r.get(bl_col)
                    if bl_val is not None:
                        record[bl_col] = bl_val
                records.append(record)
            if bad_snap_rows:
                print(f"  ⚠️ Skipped JSONB on {bad_snap_rows} row(s) with non-dict atm_snapshot")

            df = pd.DataFrame(records)
            df["target_date"] = df["target_date"].astype(str)
            print(f"  📦 Supabase snapshots: {len(df)} scored rows with atm features "
                  f"(features-at-prediction-time → gold-standard training data)")
            return df

        except Exception as e:
            print(f"  ⚠️ Supabase snapshot load failed: {e}")
            return None

    def _merge_supabase_snapshots(self, snap_df: pd.DataFrame) -> None:
        """
        Merge Supabase snapshot rows into self.features_df.

        Strategy:
          • For dates already in features_df: override atmospheric columns
            with the snapshot values (forecast-at-inference > archive-actual).
          • For dates only in snap_df (not yet in CSV backfill): append as
            new training rows so they count immediately.
        """
        if snap_df is None or snap_df.empty or self.features_df is None:
            return

        atm_cols = [c for c in snap_df.columns
                    if c not in ("target_date", "actual_high",
                                 "nws_last", "accu_last", "nws_first",
                                 "nws_mean", "nws_spread",
                                 "rolling_bias_7d", "rolling_bias_21d")]

        existing_dates = set(self.features_df["target_date"].astype(str))

        # 1. Override atmospheric features for dates we already have
        override_df = snap_df[snap_df["target_date"].isin(existing_dates)].copy()
        if not override_df.empty:
            for col in atm_cols:
                if col not in self.features_df.columns:
                    self.features_df[col] = np.nan
            self.features_df = self.features_df.merge(
                override_df[["target_date"] + atm_cols].rename(
                    columns={c: f"_sb_{c}" for c in atm_cols}),
                on="target_date", how="left",
            )
            for col in atm_cols:
                sb_col = f"_sb_{col}"
                if sb_col in self.features_df.columns:
                    mask = self.features_df[sb_col].notna()
                    self.features_df.loc[mask, col] = self.features_df.loc[mask, sb_col]
                    self.features_df.drop(columns=[sb_col], inplace=True)
            print(f"  ↑ Overrode atmospheric features for {len(override_df)} existing dates "
                  f"with Supabase forecast-time snapshots")

        # 2. Append net-new dates (not yet in the CSV backfill)
        new_df = snap_df[~snap_df["target_date"].isin(existing_dates)].copy()
        if not new_df.empty:
            # Ensure all columns align
            for col in self.features_df.columns:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            self.features_df = pd.concat(
                [self.features_df, new_df[self.features_df.columns]],
                ignore_index=True,
            ).sort_values("target_date").reset_index(drop=True)
            print(f"  + Appended {len(new_df)} net-new training rows from Supabase snapshots")

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

    def _load_high_timing_features(self) -> pd.DataFrame | None:
        """Load high-timing features CSV (from backfill_high_timing_features)."""
        prefix = self.model_prefix
        csv = f"{prefix}high_timing_data.csv"
        try:
            df = pd.read_csv(csv)
            overnight = int(df["obs_is_overnight_high"].sum()) if "obs_is_overnight_high" in df.columns else 0
            print(f"  Loaded {len(df)} high-timing rows from {csv} "
                  f"({overnight} overnight highs, "
                  f"{overnight/len(df)*100:.1f}%)")
            return df
        except FileNotFoundError:
            print(f"  ⚠️ No high-timing data: {csv}")
            print(f"     Run: python prediction_writer.py --city {self.city_key} backfill_high_timing")
            return None

    def _merge_high_timing_features(self, ht_df: pd.DataFrame) -> None:
        """Merge high-timing features into self.features_df."""
        from model_config import HIGH_TIMING_COLS
        if ht_df is None or self.features_df is None:
            return
        ht_df = ht_df.copy()
        ht_df["target_date"] = ht_df["target_date"].astype(str)
        # Deduplicate
        ht_df = ht_df.drop_duplicates(subset="target_date", keep="last")
        ht_indexed = ht_df.set_index("target_date")
        cols_to_merge = [c for c in HIGH_TIMING_COLS if c in ht_df.columns]
        dates_updated = 0
        for date_str in ht_indexed.index:
            mask = self.features_df["target_date"] == date_str
            if mask.sum() == 0:
                continue
            for col in cols_to_merge:
                val = ht_indexed.loc[date_str, col]
                if pd.notna(val):
                    self.features_df.loc[mask, col] = val
            dates_updated += 1
        print(f"  Merged high-timing features for {dates_updated} dates "
              f"({', '.join(cols_to_merge)})")

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
        values (obs_vs_intra_forecast=actual delta). Also merges regional obs
        cols (JFK/LGA temps, metro spread, etc.) that IEM backfill provides.
        """
        from model_config import OBSERVATION_COLS, REGIONAL_OBS_COLS

        if obs_df is None or self.features_df is None:
            return

        obs_df["target_date"] = obs_df["target_date"].astype(str)

        # Safety check: deduplicate by target_date (keep last occurrence)
        dupes = obs_df["target_date"].duplicated(keep="last")
        if dupes.any():
            n_dupes = dupes.sum()
            print(f"  ⚠️ Removed {n_dupes} duplicate target_dates from observation_data.csv")
            obs_df = obs_df[~dupes].copy()

        # Merge BOTH OBSERVATION_COLS and REGIONAL_OBS_COLS from the CSV
        all_obs_cols = OBSERVATION_COLS + REGIONAL_OBS_COLS
        obs_cols_in_csv = [c for c in obs_df.columns if c in all_obs_cols]
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

    def _load_intraday_snapshots(self) -> pd.DataFrame | None:
        """Load historical_intraday_snapshots.csv if it exists.

        These are per-hour synthetic obs snapshots for every historical
        date, giving the trainer ~10x more rows per day so v14 blind-spot
        features (which gate on obs_latest_hour) get signal across the
        full IEM history rather than just the live-system days.
        """
        prefix = self.model_prefix
        intraday_csv = f"{prefix}historical_intraday_snapshots.csv"
        try:
            df = pd.read_csv(intraday_csv)
            df["target_date"] = df["target_date"].astype(str)
            print(f"  Loaded {len(df)} intraday snapshot rows from {intraday_csv}")
            return df
        except FileNotFoundError:
            print(f"  ⚠️ No intraday snapshot file: {intraday_csv} (skipping)")
            return None

    def _merge_intraday_snapshots(self, intraday_df: pd.DataFrame) -> None:
        """Append intraday snapshot rows as additional rows in features_df.

        Each intraday row clones the daily row for the same target_date,
        then overwrites obs_* columns with the intraday-specific values
        and sets lead_used to a synthetic sentinel. All non-obs features
        (mm_*, atm_*, nws_last, actual_high) come from the daily row so
        v13/v15 group-by-cli_date logic still works.
        """
        from model_config import OBSERVATION_COLS, REGIONAL_OBS_COLS

        if intraday_df is None or intraday_df.empty:
            return
        if self.features_df is None or self.features_df.empty:
            return

        df = self.features_df.copy()
        df["target_date"] = df["target_date"].astype(str)

        # Index daily rows by target_date — keep first row per date as the donor.
        # If multiple already exist (rare), use the one with the most non-null cols.
        daily_donor = (
            df.groupby("target_date", as_index=False)
              .first()
              .set_index("target_date")
        )

        all_obs_cols = set(OBSERVATION_COLS) | set(REGIONAL_OBS_COLS)
        intraday_obs_cols = [
            c for c in intraday_df.columns
            if c in all_obs_cols and c != "target_date"
        ]
        if not intraday_obs_cols:
            print(f"  ⚠️ No obs columns in intraday snapshot CSV — skipping merge")
            return

        new_rows = []
        skipped_no_donor = 0
        for _, snap in intraday_df.iterrows():
            date_str = str(snap["target_date"])
            if date_str not in daily_donor.index:
                skipped_no_donor += 1
                continue
            # Clone the daily row
            base = daily_donor.loc[date_str].to_dict()
            base["target_date"] = date_str
            # Overwrite obs_* with intraday snapshot values
            for col in intraday_obs_cols:
                v = snap[col]
                if pd.notna(v):
                    base[col] = v
            # Mark these rows as synthetic so they don't contaminate the
            # v13/v15 lead-aware logic that branches on lead_used values.
            H = int(snap["hour_checkpoint"]) if pd.notna(snap.get("hour_checkpoint")) else -1
            base["lead_used"] = f"historical_synthetic_h{H}"
            new_rows.append(base)

        if not new_rows:
            print(f"  ⚠️ Intraday merge produced 0 rows (skipped {skipped_no_donor} with no donor)")
            return

        intraday_rows_df = pd.DataFrame(new_rows)
        # Ensure column alignment with self.features_df
        for col in df.columns:
            if col not in intraday_rows_df.columns:
                intraday_rows_df[col] = np.nan
        intraday_rows_df = intraday_rows_df[df.columns.tolist() + [
            c for c in intraday_rows_df.columns if c not in df.columns
        ]]

        before = len(df)
        merged = pd.concat([df, intraday_rows_df], ignore_index=True)
        merged = merged.sort_values("target_date").reset_index(drop=True)
        self.features_df = merged
        print(f"  Intraday snapshots: appended {len(new_rows)} synthetic rows "
              f"(features_df: {before} → {len(merged)}; "
              f"skipped {skipped_no_donor} no-donor)")

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

    def _compute_model_vs_nws_features(self) -> None:
        """
        Derive v11 model-vs-NWS divergence columns from existing features.

        mm_hrrr_vs_nws  = mm_hrrr_max - nws_last
        mm_nbm_vs_nws   = mm_nbm_max  - nws_last
        mm_mean_vs_nws  = mm_mean     - nws_last

        Called after all merges so both mm_* and nws_last are in features_df.
        Rows without nws_last (multi-year atmospheric-only rows) get NaN —
        HistGradientBoosting handles NaN natively, so no data is lost.
        """
        df = self.features_df
        # Per-target-date aggregation (same fix pattern as v13/v15):
        # the training matrix has SPLIT rows per target_date — the forecast
        # row holds nws_last and the multi-year/atm row holds mm_*. Row-by-row
        # subtraction gives only ~4/273 coverage. Aggregate by target_date so
        # nws_last from the forecast row pairs with mm_* from any other row
        # for the same date.
        mm_cols = [c for c in ["mm_hrrr_max", "mm_nbm_max", "mm_mean"] if c in df.columns]
        agg_cols = mm_cols + (["nws_last"] if "nws_last" in df.columns else [])
        if "target_date" in df.columns and agg_cols:
            _tmp = df[["target_date", *agg_cols]].copy()
            _tmp["target_date"] = pd.to_datetime(_tmp["target_date"], errors="coerce")
            for _c in agg_cols:
                _tmp[_c] = pd.to_numeric(_tmp[_c], errors="coerce")
            mm_per_date = (
                _tmp.dropna(subset=["target_date"])
                .groupby("target_date", as_index=False)[agg_cols]
                .max()
            )
        else:
            mm_per_date = pd.DataFrame(columns=["target_date"])

        def _mm_lookup(value_series):
            if mm_per_date.empty:
                return {}
            return {pd.Timestamp(d): v for d, v in zip(mm_per_date["target_date"], value_series)
                    if pd.notna(v)}

        df_dates = pd.to_datetime(df["target_date"], errors="coerce") if "target_date" in df.columns else None

        if not mm_per_date.empty and "nws_last" in mm_per_date.columns:
            nws_pd = mm_per_date["nws_last"]
            for src, out in [("mm_hrrr_max", "mm_hrrr_vs_nws"),
                             ("mm_nbm_max",  "mm_nbm_vs_nws"),
                             ("mm_mean",     "mm_mean_vs_nws")]:
                if src in mm_per_date.columns:
                    diff_pd = (mm_per_date[src] - nws_pd).round(1)
                    lookup = _mm_lookup(diff_pd)
                    if df_dates is not None and lookup:
                        df[out] = df_dates.map(lookup)
                    else:
                        df[out] = np.nan
                else:
                    df[out] = np.nan
        else:
            for out in ["mm_hrrr_vs_nws", "mm_nbm_vs_nws", "mm_mean_vs_nws"]:
                df[out] = np.nan

        n_hrrr = df["mm_hrrr_vs_nws"].notna().sum() if "mm_hrrr_vs_nws" in df.columns else 0
        n_nbm  = df["mm_nbm_vs_nws"].notna().sum() if "mm_nbm_vs_nws" in df.columns else 0
        n_mean = df["mm_mean_vs_nws"].notna().sum() if "mm_mean_vs_nws" in df.columns else 0
        print(f"  v11 divergence features: mm_hrrr_vs_nws={n_hrrr} rows, mm_nbm_vs_nws={n_nbm} rows")
        _record_coverage(self.model_prefix, "v11", {
            "mm_hrrr_vs_nws": n_hrrr, "mm_nbm_vs_nws": n_nbm, "mm_mean_vs_nws": n_mean,
        })
        self.features_df = df

    def _compute_bl_safeguard_features(self) -> None:
        """
        Derive v13 BL safeguard columns from existing features.

        entrainment_temp_diff = atm_925mb_temp_mean - obs_latest_temp
          → Negative = cool aloft (potential mixing), positive/near-zero = neutral

        marine_containment = obs_kjfk_vs_knyc / atm_bl_height_max
          → Ratio of coastal gradient to BL depth; captures how contained ocean air is

        inland_strength = mean(obs_kteb_temp, obs_kcdw_temp, obs_ksmq_temp) - mm_mean
          → Inland actual vs forecast consensus; positive = inland beating forecast

        Called after all merges so all input columns are populated.
        Rows with missing inputs get NaN — HistGradientBoosting handles NaN natively.
        """
        df = self.features_df

        # Preserve any pre-populated values (e.g. from Supabase top-level cols
        # loaded via _load_supabase_snapshot_features). We compute fresh values
        # below and combine: computed takes priority where valid, else fall
        # back to the pre-existing value.
        def _coalesce(col_name, computed):
            existing = df[col_name] if col_name in df.columns else None
            if existing is None:
                df[col_name] = computed
            else:
                df[col_name] = computed.where(computed.notna(), existing)

        # The training matrix splits forecast/observation rows from multi-year
        # atmospheric rows by cli_date. Row-by-row computation produced
        # 10–18 rows of coverage out of 1577 days because the inputs sit on
        # different rows for the same cli_date. We resolve via per-cli_date
        # max-aggregation, then map back. (Same fix as v15 — see PR #28.)
        bl_input_cols = [c for c in [
            "atm_925mb_temp_mean", "obs_latest_temp",
            "obs_kjfk_vs_knyc", "atm_bl_height_max",
            "obs_kteb_temp", "obs_kcdw_temp", "obs_ksmq_temp", "mm_mean",
        ] if c in df.columns]
        # Note: training matrix uses `target_date` (not `cli_date`). cli_date
        # exists only on raw nws_df, not on the merged features_df we operate on.
        if "target_date" in df.columns and bl_input_cols:
            _bl_tmp = df[["target_date", *bl_input_cols]].copy()
            _bl_tmp["target_date"] = pd.to_datetime(_bl_tmp["target_date"], errors="coerce")
            for _c in bl_input_cols:
                _bl_tmp[_c] = pd.to_numeric(_bl_tmp[_c], errors="coerce")
            bl_per_date = (
                _bl_tmp.dropna(subset=["target_date"])
                .groupby("target_date", as_index=False)[bl_input_cols]
                .max()
            )
        else:
            bl_per_date = pd.DataFrame(columns=["target_date"])

        def _bl_lookup_map(value_series):
            """Build {Timestamp(date): value} skipping NaN."""
            if bl_per_date.empty:
                return {}
            return {
                pd.Timestamp(d): v
                for d, v in zip(bl_per_date["target_date"], value_series)
                if pd.notna(v)
            }

        df_dates = pd.to_datetime(df["target_date"], errors="coerce") if "target_date" in df.columns else None

        # 1) entrainment_temp_diff = 925mb - obs_latest_temp (per-date)
        if (
            not bl_per_date.empty
            and "atm_925mb_temp_mean" in bl_per_date.columns
            and "obs_latest_temp" in bl_per_date.columns
        ):
            _entr_per_date = (bl_per_date["atm_925mb_temp_mean"] - bl_per_date["obs_latest_temp"]).round(1)
            _entr_lookup = _bl_lookup_map(_entr_per_date)
            _coalesce("entrainment_temp_diff",
                      pd.Series(df_dates.map(_entr_lookup), index=df.index))
        elif "entrainment_temp_diff" not in df.columns:
            df["entrainment_temp_diff"] = np.nan

        # 2) marine_containment = obs_kjfk_vs_knyc / atm_bl_height_max (per-date)
        if (
            not bl_per_date.empty
            and "obs_kjfk_vs_knyc" in bl_per_date.columns
            and "atm_bl_height_max" in bl_per_date.columns
        ):
            _bl_h = bl_per_date["atm_bl_height_max"]
            _mc_per_date = pd.Series(
                np.where(
                    _bl_h.notna() & (_bl_h > 0),
                    (bl_per_date["obs_kjfk_vs_knyc"] / _bl_h).round(6),
                    np.nan,
                ),
                index=bl_per_date.index,
            )
            _mc_lookup = _bl_lookup_map(_mc_per_date)
            _coalesce("marine_containment",
                      pd.Series(df_dates.map(_mc_lookup), index=df.index))
        elif "marine_containment" not in df.columns:
            df["marine_containment"] = np.nan

        # 3) inland_strength = mean(kteb, kcdw, ksmq) - mm_mean (per-date)
        if not bl_per_date.empty and "mm_mean" in bl_per_date.columns:
            _inland_cols = [c for c in ("obs_kteb_temp", "obs_kcdw_temp", "obs_ksmq_temp")
                            if c in bl_per_date.columns]
            if _inland_cols:
                _inland_mean = bl_per_date[_inland_cols].mean(axis=1, skipna=True)
                _is_per_date = (_inland_mean - bl_per_date["mm_mean"]).round(1)
                _is_lookup = _bl_lookup_map(_is_per_date)
                _coalesce("inland_strength",
                          pd.Series(df_dates.map(_is_lookup), index=df.index))
            elif "inland_strength" not in df.columns:
                df["inland_strength"] = np.nan
        elif "inland_strength" not in df.columns:
            df["inland_strength"] = np.nan

        # Log coverage
        n_entrainment = df["entrainment_temp_diff"].notna().sum()
        n_marine = df["marine_containment"].notna().sum()
        n_inland = df["inland_strength"].notna().sum()
        print(f"  v13 BL safeguard features: entrainment={n_entrainment} rows, "
              f"marine_containment={n_marine} rows, inland_strength={n_inland} rows")
        _record_coverage(self.model_prefix, "v13", {
            "entrainment_temp_diff": n_entrainment,
            "marine_containment": n_marine,
            "inland_strength": n_inland,
        })
        self.features_df = df


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
        # Pass the FULL features_df — including multi-year rows with no NWS/AccuWeather.
        # build_classification_dataset() already handles the priority chain:
        #   AccuWeather > NWS > _persistence_forecast (yesterday's actual high).
        # Multi-year rows use persistence as center and a wider 13-candidate window.
        # Real-forecast rows are upweighted 5:1 so the model stays shaped toward the
        # production path (real forecasts always available at inference).
        # Previously this was filtered to forecast-only days (~265 rows), which starved
        # the classifier — especially for LAX (only 39 forecast days).
        has_forecast = self.features_df["nws_last"].notna() | self.features_df["accu_last"].notna()
        n_forecast_rows = has_forecast.sum()
        n_persistence_rows = (~has_forecast & self.features_df["_persistence_forecast"].notna()).sum()
        print(f"\n  Classifier training on ALL {len(self.features_df)} days "
              f"({n_forecast_rows} real-forecast + {n_persistence_rows} persistence rows)")
        classifier = BucketClassifier()
        classifier.train(self.features_df.copy().reset_index(drop=True),
                         feature_cols=FEATURE_COLS_V2,
                         residual_std=residual_std_v2,
                         forecast_weight=5.0)
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

        # Append per-hour synthetic intraday snapshots (~10x rows/day for blind-spot features)
        intraday_df = self._load_intraday_snapshots()
        if intraday_df is not None:
            self._merge_intraday_snapshots(intraday_df)

        # Supabase snapshot override (forecast-at-inference-time > archive actuals)
        sb_snap_df = self._load_supabase_snapshot_features()
        if sb_snap_df is not None:
            self._merge_supabase_snapshots(sb_snap_df)

        # Ensure all v4 columns exist (fill missing with NaN)
        available_v4_cols = [c for c in FEATURE_COLS_V4 if c in self.features_df.columns]
        missing_v4 = [c for c in FEATURE_COLS_V4 if c not in self.features_df.columns]
        if missing_v4:
            print(f"  Missing v4 columns (will be NaN): {missing_v4[:5]}{'...' if len(missing_v4) > 5 else ''}")
            for col in missing_v4:
                self.features_df[col] = np.nan

        # Check observation feature coverage (OBSERVATION_COLS + REGIONAL_OBS_COLS)
        from model_config import REGIONAL_OBS_COLS as _REG_COLS
        all_obs_check = OBSERVATION_COLS + _REG_COLS
        obs_populated = 0
        for col in all_obs_check:
            if col in self.features_df.columns:
                n = self.features_df[col].notna().sum()
                if n > 0:
                    obs_populated += 1
        print(f"  Observation features with data: {obs_populated}/{len(all_obs_check)} "
              f"({len(OBSERVATION_COLS)} base + {len(_REG_COLS)} regional)")
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
        # Same expansion as v2: pass full features_df so persistence rows contribute.
        # Real-forecast rows upweighted 5:1 over persistence rows.
        has_forecast_v4 = self.features_df["nws_last"].notna() | self.features_df["accu_last"].notna()
        n_fc_v4 = has_forecast_v4.sum()
        n_ps_v4 = (~has_forecast_v4 & self.features_df["_persistence_forecast"].notna()).sum()
        print(f"\n  v4 Classifier training on ALL {len(self.features_df)} days "
              f"({n_fc_v4} real-forecast + {n_ps_v4} persistence rows)")
        classifier = BucketClassifier()
        classifier.train(self.features_df.copy().reset_index(drop=True),
                         feature_cols=FEATURE_COLS_V4,
                         residual_std=residual_std_v4,
                         forecast_weight=5.0)
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

    def train_v5(self) -> None:
        """
        Train v5 model: v4 + HIGH_TIMING_COLS (3 features = 122 total).

        Adds obs_high_peak_hour, obs_is_overnight_high, obs_temp_falling_hrs.
        These capture the three meteorological regimes that the 2pm/3pm clock
        cutoffs miss:
          1. Overnight/pre-dawn highs (warm-front passage, high at 1-3am)
          2. Late afternoon/evening highs (sea-breeze collapse, high at 4-6pm)
          3. Normal solar peak (1-3pm) — these new features add nothing but don't hurt

        Requires high_timing_data.csv from:
          python prediction_writer.py --city nyc backfill_high_timing
        """
        from model_config import FEATURE_COLS_V5, HIGH_TIMING_COLS
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v5 Training: v4 + High-Timing Features ({len(FEATURE_COLS_V5)} features)")
        print(f"{'═'*60}")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data. Run train_v4() first.")
            return

        # Load and merge high-timing features
        ht_df = self._load_high_timing_features()
        if ht_df is not None:
            self._merge_high_timing_features(ht_df)

        # Also ensure obs features are merged (in case train_v4 wasn't called)
        obs_df = self._load_observation_features()
        if obs_df is not None:
            self._merge_observation_features(obs_df)

        # Append per-hour synthetic intraday snapshots if not already merged
        if "lead_used" not in self.features_df.columns or \
           not self.features_df["lead_used"].astype(str).str.startswith("historical_synthetic_h").any():
            intraday_df = self._load_intraday_snapshots()
            if intraday_df is not None:
                self._merge_intraday_snapshots(intraday_df)

        # ── Supabase snapshot override ─────────────────────────────────────
        # Load features-at-prediction-time from Supabase prediction_logs.
        # These override the Open-Meteo archive values for the same dates
        # (forecast data > archive actuals for training/inference consistency),
        # and add net-new rows for dates not yet in the CSV backfill.
        sb_snap_df = self._load_supabase_snapshot_features()
        if sb_snap_df is not None:
            self._merge_supabase_snapshots(sb_snap_df)

        # Ensure all v5 columns exist
        missing_v5 = [c for c in FEATURE_COLS_V5 if c not in self.features_df.columns]
        if missing_v5:
            print(f"  Missing v5 cols (NaN): {missing_v5}")
            for col in missing_v5:
                self.features_df[col] = np.nan

        # High-timing coverage report
        for col in HIGH_TIMING_COLS:
            n = self.features_df[col].notna().sum() if col in self.features_df.columns else 0
            print(f"  {col}: {n} non-null rows")

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v5 = forecast_df[FEATURE_COLS_V5]
        y_actual = forecast_df["actual_high"]
        nws_last = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]
        base = accu_last.copy()
        base[base.isna()] = nws_last[base.isna()]
        y_bias = y_actual - base

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, bucket_acc, all_residuals = [], [], []

        for tr, te in tscv.split(X_v5):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X_v5.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X_v5.iloc[te])
            pred_temp = base.iloc[te].values + pred_bias
            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())
            pred_buckets   = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc.append(correct / len(actual_buckets))

        residual_std = float(np.std(all_residuals))
        print(f"  v5 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v5 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v5 Residual Std:  {residual_std:.2f}°F")

        # Train final model on all data
        v5_regressor = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        v5_regressor.fit(X_v5, y_bias)

        # Feature importance for high-timing cols
        try:
            fi = dict(zip(FEATURE_COLS_V5, v5_regressor.feature_importances_))
            print("  High-timing feature importances:")
            for col in HIGH_TIMING_COLS:
                print(f"    {col}: {fi.get(col, 0):.4f}")
        except Exception:
            pass

        # Bucket classifier — train on ALL rows (forecast + persistence) with
        # forecast_weight=5.0, matching the v4 approach that unlocked 1,295 extra rows.
        # HIGH_TIMING_COLS will be NaN for persistence rows (no obs data) — HistGBT handles natively.
        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V5,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        # Save models
        prefix = self.model_prefix
        import pickle
        with open(f"{prefix}bcp_v5_regressor.pkl", "wb") as f:
            pickle.dump(v5_regressor, f)
        with open(f"{prefix}bcp_v5_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v5_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V5), f)

        v5_meta = {
            "v5_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V5),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v5": list(FEATURE_COLS_V5),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v5.json", "w") as f:
            _json.dump(v5_meta, f, indent=2)

        print(f"\n  ✅ Saved v5 models: bcp_v5_regressor.pkl, bcp_v5_classifier.pkl")
        print(f"  v5 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        # Compare v4 vs v5
        try:
            import json as _j
            with open(f"{prefix}model_metadata_v4.json") as f:
                v4_meta = _j.load(f)
            v4_mae = v4_meta.get("v4_regression", {}).get("cv_mae")
            v4_bkt = v4_meta.get("v4_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v4 vs v5")
            if v4_mae: print(f"  MAE:        v4={v4_mae:.2f}°F → v5={np.mean(mae_scores):.2f}°F")
            if v4_bkt: print(f"  Bucket Acc: v4={v4_bkt:.1%} → v5={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v6(self) -> None:
        """
        Train v6 model: v5 + NBM, GEM HRDPS, HRRR-specific 925mb, OKX radiosonde (138 features).

        New features address the April 12, 2026 cap-miss case:
          - mm_nbm_max / mm_gem_hrdps_max: top-accuracy models per wethr.net rankings
          - mm_nbm_hrrr_diff: disagreement between #1 and #3 accuracy models
          - atm_925mb_hrrr_*: HRRR 3km boundary layer vs GFS 13km
          - atm_925mb_gfs_hrrr_diff: when large, GFS is missing the cap
          - raob_*: OKX upper-air balloon soundings (actual observed 925mb/850mb)
          - raob_925mb_gfs_diff / raob_925mb_hrrr_diff: forecast vs observed cap signal

        These features are NaN for historical rows (archive has no HRRR-specific 925mb,
        no NBM, no radiosonde backfill yet). HistGradientBoosting handles NaN natively;
        as live rows accumulate, the model will learn to weight them appropriately.

        Requires the v5 training data to be loaded first.
        """
        from model_config import FEATURE_COLS_V6, HRRR_PRESSURE_COLS, RADIOSONDE_COLS
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v6 Training: v5 + NBM/GEM-HRDPS/HRRR-925mb/Radiosonde ({len(FEATURE_COLS_V6)} features)")
        print(f"{'═'*60}")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data. Run train_v5() first.")
            return

        # Ensure all v6 columns exist (NaN for historical rows without these features)
        missing_v6 = [c for c in FEATURE_COLS_V6 if c not in self.features_df.columns]
        if missing_v6:
            print(f"  Missing v6 cols (NaN for historical rows): {len(missing_v6)} columns")
            for col in missing_v6:
                self.features_df[col] = np.nan

        # Coverage report for new v6 features
        new_v6_cols = (
            ["mm_nbm_max", "mm_gem_hrdps_max", "mm_nbm_hrrr_diff"]
            + HRRR_PRESSURE_COLS
            + RADIOSONDE_COLS
        )
        print("  New v6 feature coverage (live rows only):")
        for col in new_v6_cols:
            n = self.features_df[col].notna().sum() if col in self.features_df.columns else 0
            print(f"    {col}: {n} non-null rows")

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v6 = forecast_df[FEATURE_COLS_V6]
        y_actual = forecast_df["actual_high"]
        nws_last = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]
        base = accu_last.copy()
        base[base.isna()] = nws_last[base.isna()]
        y_bias = y_actual - base

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, bucket_acc, all_residuals = [], [], []

        for tr, te in tscv.split(X_v6):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X_v6.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X_v6.iloc[te])
            pred_temp = base.iloc[te].values + pred_bias
            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())
            pred_buckets   = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc.append(correct / len(actual_buckets))

        residual_std = float(np.std(all_residuals))
        print(f"  v6 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v6 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v6 Residual Std:  {residual_std:.2f}°F")

        # Train final model on all data
        v6_regressor = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        v6_regressor.fit(X_v6, y_bias)

        # Feature importance for new v6 cols
        try:
            fi = dict(zip(FEATURE_COLS_V6, v6_regressor.feature_importances_))
            print("  New v6 feature importances:")
            for col in new_v6_cols:
                print(f"    {col}: {fi.get(col, 0):.4f}")
        except Exception:
            pass

        # Bucket classifier
        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V6,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        # Save models
        prefix = self.model_prefix
        with open(f"{prefix}bcp_v6_regressor.pkl", "wb") as f:
            pickle.dump(v6_regressor, f)
        with open(f"{prefix}bcp_v6_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v6_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V6), f)

        v6_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v6_nbm_hrdps_hrrr925_radiosonde",
            "v6_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V6),
                "n_training_rows": n_forecast,
            },
            "new_features_v6": new_v6_cols,
            "feature_columns_v6": list(FEATURE_COLS_V6),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v6.json", "w") as f:
            _json.dump(v6_meta, f, indent=2)

        print(f"\n  ✅ Saved v6 models: bcp_v6_regressor.pkl, bcp_v6_classifier.pkl")
        print(f"  v6 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        # Compare v5 vs v6
        try:
            import json as _j
            with open(f"{prefix}model_metadata_v5.json") as f:
                v5_meta = _j.load(f)
            v5_mae = v5_meta.get("v5_regression", {}).get("cv_mae")
            v5_bkt = v5_meta.get("v5_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v5 vs v6")
            if v5_mae: print(f"  MAE:        v5={v5_mae:.2f}°F → v6={np.mean(mae_scores):.2f}°F")
            if v5_bkt: print(f"  Bucket Acc: v5={v5_bkt:.1%} → v6={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v7(self) -> None:
        """
        Train v7 model: same 138 features as v6 but with HRRR > NBM > AccuWeather > NWS base.

        ARCHITECTURAL CHANGE — v7 trains y_bias = actual - HRRR_max (when HRRR is available),
        falling back to NBM, then AccuWeather, then NWS for historical rows where HRRR/NBM
        are NaN.  This makes the regressor learn "how far off is HRRR?" rather than
        "how far off is AccuWeather?" — a fundamental improvement since HRRR is the
        highest-accuracy short-range model (#1 on wethr.net vs AccuWeather #11-16).

        Cap-day example (April 12, 2026):
          v6:  base = AccuWeather 58°F, bias ≈ -1 → predicted 57°F (WRONG)
          v7:  base = HRRR 54°F, bias ≈ -1 → predicted 53°F (correct ≤55 bucket)

        At inference time (prediction_writer.py), the priority also flips:
          v7 active + HRRR available → center = HRRR_max + v7_regressor_bias
          v7 active + NBM available  → center = NBM_max + v7_regressor_bias
          fallback                   → atm_predicted_high or AccuWeather/NWS + bias

        Requires v6 training data to be loaded first.
        """
        from model_config import FEATURE_COLS_V7, HRRR_PRESSURE_COLS, RADIOSONDE_COLS
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v7 Training: HRRR-anchored base + {len(FEATURE_COLS_V7)} features")
        print(f"{'═'*60}")
        print(f"  Key change: y_bias = actual - HRRR_max (not AccuWeather)")
        print(f"  Fallback cascade: HRRR > NBM > AccuWeather > NWS")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data. Run train_v6() first.")
            return

        # Ensure all v7 columns exist (NaN for historical rows)
        missing_v7 = [c for c in FEATURE_COLS_V7 if c not in self.features_df.columns]
        if missing_v7:
            print(f"  Missing v7 cols (NaN for historical rows): {len(missing_v7)} columns")
            for col in missing_v7:
                self.features_df[col] = np.nan

        # Coverage report — how many rows have the HRRR/NBM base available?
        new_cols = (
            ["mm_hrrr_max", "mm_nbm_max", "mm_gem_hrdps_max"]
            + HRRR_PRESSURE_COLS
            + RADIOSONDE_COLS
        )
        print("  Base cascade coverage (live rows only):")
        for col in new_cols:
            n = self.features_df[col].notna().sum() if col in self.features_df.columns else 0
            print(f"    {col}: {n} non-null rows")

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v7 = forecast_df[FEATURE_COLS_V7]
        y_actual = forecast_df["actual_high"]
        nws_last = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]

        # ── v7 HRRR > NBM > AccuWeather > NWS base cascade ───────────────
        # For rows where HRRR is available (recent live rows): use HRRR as base.
        # For older historical rows (HRRR NaN): fall through to NBM → AccuWeather → NWS.
        # HistGradientBoostingRegressor handles mixed-base training natively via NaN.
        hrrr_max = forecast_df.get("mm_hrrr_max", pd.Series(np.nan, index=forecast_df.index))
        nbm_max  = forecast_df.get("mm_nbm_max",  pd.Series(np.nan, index=forecast_df.index))

        base = hrrr_max.copy()                                 # prefer HRRR
        base[base.isna()] = nbm_max[base.isna()].values       # fallback to NBM
        base[base.isna()] = accu_last[base.isna()].values     # fallback to AccuWeather
        base[base.isna()] = nws_last[base.isna()].values      # last resort: NWS

        # Log base source breakdown
        n_hrrr  = hrrr_max.notna().sum()
        n_nbm   = (hrrr_max.isna() & nbm_max.notna()).sum()
        n_accu  = (hrrr_max.isna() & nbm_max.isna() & accu_last.notna()).sum()
        n_nws   = (hrrr_max.isna() & nbm_max.isna() & accu_last.isna() & nws_last.notna()).sum()
        print(f"\n  Base source breakdown: HRRR={n_hrrr}, NBM={n_nbm}, "
              f"AccuWeather={n_accu}, NWS={n_nws}")

        y_bias = y_actual - base

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, bucket_acc, all_residuals = [], [], []

        for tr, te in tscv.split(X_v7):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X_v7.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X_v7.iloc[te])
            pred_temp = base.iloc[te].values + pred_bias
            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())
            pred_buckets   = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc.append(correct / len(actual_buckets))

        residual_std = float(np.std(all_residuals))
        print(f"  v7 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v7 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v7 Residual Std:  {residual_std:.2f}°F")

        # Train final model on all data
        v7_regressor = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        v7_regressor.fit(X_v7, y_bias)

        # Feature importance for key v7 features
        try:
            fi = dict(zip(FEATURE_COLS_V7, v7_regressor.feature_importances_))
            top_cols = (
                ["mm_hrrr_max", "mm_nbm_max", "mm_gem_hrdps_max",
                 "mm_hrrr_gfs_diff", "mm_nbm_hrrr_diff", "atm_925mb_gfs_hrrr_diff"]
                + HRRR_PRESSURE_COLS
                + RADIOSONDE_COLS
            )
            print("  Key v7 feature importances (HRRR/cap-related):")
            for col in top_cols:
                if col in fi:
                    print(f"    {col}: {fi[col]:.4f}")
        except Exception:
            pass

        # Bucket classifier
        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V7,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        # Save models
        prefix = self.model_prefix
        with open(f"{prefix}bcp_v7_regressor.pkl", "wb") as f:
            pickle.dump(v7_regressor, f)
        with open(f"{prefix}bcp_v7_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v7_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V7), f)

        v7_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v7_hrrr_anchored_base",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "v7_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V7),
                "n_training_rows": n_forecast,
                "base_sources": {
                    "hrrr": int(n_hrrr),
                    "nbm": int(n_nbm),
                    "accuweather": int(n_accu),
                    "nws": int(n_nws),
                },
            },
            "feature_columns_v7": list(FEATURE_COLS_V7),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v7.json", "w") as f:
            _json.dump(v7_meta, f, indent=2)

        print(f"\n  ✅ Saved v7 models: bcp_v7_regressor.pkl, bcp_v7_classifier.pkl")
        print(f"  v7 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        # Compare v6 vs v7
        try:
            import json as _j
            with open(f"{prefix}model_metadata_v6.json") as f:
                v6_meta = _j.load(f)
            v6_mae = v6_meta.get("v6_regression", {}).get("cv_mae")
            v6_bkt = v6_meta.get("v6_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v6 vs v7 (key change: HRRR-anchored base)")
            if v6_mae: print(f"  MAE:        v6={v6_mae:.2f}°F → v7={np.mean(mae_scores):.2f}°F")
            if v6_bkt: print(f"  Bucket Acc: v6={v6_bkt:.1%} → v7={np.mean(bucket_acc):.1%}")
            print(f"  Note: CV uses mixed base (mostly historical AccuWeather rows).")
            print(f"  v7 advantage grows as HRRR/NBM live rows accumulate over time.")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v8(self) -> None:
        """
        Train v8 model: v7 (HRRR-anchored base, 138 features) + obs_heating_rate_delta.

        THE STALL SIGNAL:
          obs_heating_rate_delta = recent_slope - early_slope (°F/hr).
          Negative = warming rate is decelerating = cap is holding.
          This is the automated version of "watching the stations plateau for 2 hours."

          April 12 example:
            7-9am: +1.8°F/hr → 10am-noon: +0.2°F/hr → delta = -1.6°F/hr

        Combined with HRRR-anchored base (v7), HRRR vs NWS gap, radiosonde, this
        is the full signal stack that would have kept the model at ≤55 on April 12.

        Requires v7 training data to be loaded first.
        """
        from model_config import FEATURE_COLS_V8
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v8 Training: v7 + obs_heating_rate_delta stall signal ({len(FEATURE_COLS_V8)} features)")
        print(f"{'═'*60}")
        print(f"  New: obs_heating_rate_delta — deceleration = cap fingerprint")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data. Run train_v7() first.")
            return

        missing_v8 = [c for c in FEATURE_COLS_V8 if c not in self.features_df.columns]
        if missing_v8:
            print(f"  Missing v8 cols (NaN for historical rows): {missing_v8}")
            for col in missing_v8:
                self.features_df[col] = np.nan

        # Coverage report
        n_stall = self.features_df["obs_heating_rate_delta"].notna().sum() \
                  if "obs_heating_rate_delta" in self.features_df.columns else 0
        print(f"  obs_heating_rate_delta: {n_stall} non-null rows (live only)")

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v8 = forecast_df[FEATURE_COLS_V8]
        y_actual = forecast_df["actual_high"]
        nws_last  = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]

        # Same HRRR > NBM > AccuWeather > NWS base cascade as v7
        hrrr_max = forecast_df.get("mm_hrrr_max", pd.Series(np.nan, index=forecast_df.index))
        nbm_max  = forecast_df.get("mm_nbm_max",  pd.Series(np.nan, index=forecast_df.index))
        base = hrrr_max.copy()
        base[base.isna()] = nbm_max[base.isna()].values
        base[base.isna()] = accu_last[base.isna()].values
        base[base.isna()] = nws_last[base.isna()].values
        y_bias = y_actual - base

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, bucket_acc, all_residuals = [], [], []

        for tr, te in tscv.split(X_v8):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X_v8.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X_v8.iloc[te])
            pred_temp = base.iloc[te].values + pred_bias
            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())
            pred_buckets   = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc.append(correct / len(actual_buckets))

        residual_std = float(np.std(all_residuals))
        print(f"  v8 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v8 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v8 Residual Std:  {residual_std:.2f}°F")

        v8_regressor = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        v8_regressor.fit(X_v8, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V8, v8_regressor.feature_importances_))
            print(f"  obs_heating_rate importance:       {fi.get('obs_heating_rate', 0):.4f}")
            print(f"  obs_heating_rate_delta importance: {fi.get('obs_heating_rate_delta', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V8,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        prefix = self.model_prefix
        with open(f"{prefix}bcp_v8_regressor.pkl", "wb") as f:
            pickle.dump(v8_regressor, f)
        with open(f"{prefix}bcp_v8_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v8_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V8), f)

        v8_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v8_hrrr_anchored_stall_signal",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_feature": "obs_heating_rate_delta (stall = cap fingerprint)",
            "v8_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V8),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v8": list(FEATURE_COLS_V8),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v8.json", "w") as f:
            _json.dump(v8_meta, f, indent=2)

        print(f"\n  ✅ Saved v8 models: bcp_v8_regressor.pkl, bcp_v8_classifier.pkl")
        print(f"  v8 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v7.json") as f:
                v7_meta = _j.load(f)
            v7_mae = v7_meta.get("v7_regression", {}).get("cv_mae")
            v7_bkt = v7_meta.get("v7_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v7 vs v8 (added: stall signal)")
            if v7_mae: print(f"  MAE:        v7={v7_mae:.2f}°F → v8={np.mean(mae_scores):.2f}°F")
            if v7_bkt: print(f"  Bucket Acc: v7={v7_bkt:.1%} → v8={np.mean(bucket_acc):.1%}")
            print(f"  stall signal is NaN for most historical rows — advantage grows with live data.")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v9(self) -> None:
        """
        Train v9 model: v8 (139 features) + named ASOS station features (10 features = 149 total).

        THE MARINE CAP FINGERPRINT:
          obs_kjfk_temp        — JFK Airport (coastal Queens/Jamaica Bay)
          obs_klga_temp        — LaGuardia Airport
          obs_kewr_temp        — Newark Airport (inland)
          obs_kteb_temp        — Teterboro Airport (far inland)
          obs_knyc_temp        — Central Park (Synoptic direct)
          obs_kjfk_vs_knyc     — JFK minus Central Park (negative = sea breeze inland)
          obs_klga_vs_knyc     — LGA minus Central Park
          obs_kewr_vs_knyc     — EWR minus Central Park
          obs_airport_spread   — max minus min across all airports
          obs_coastal_vs_inland— mean(JFK,LGA) minus mean(EWR,TEB)
                                  negative = marine air mass boundary confirmed

          April 12, 2026: coastal_vs_inland = -4°F, kjfk_vs_knyc = -5°F
          → model should have locked in ≤55 and ignored the 56-57 flip signal.

        Requires Synoptic backfill to have run (backfill_synoptic.py) so these
        columns exist in atm_snapshot. Rows without station data get NaN (handled
        by HistGradientBoosting natively). Skips if fewer than 30 rows have KJFK data.
        """
        from model_config import FEATURE_COLS_V9
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v9 Training: v8 + named station marine cap features ({len(FEATURE_COLS_V9)} total features)")
        print(f"{'═'*60}")
        print(f"  New: KJFK/KLGA/KEWR/KTEB temps, coastal-vs-inland, JFK-KNYC diff")
        print(f"  Motivation: April 12 marine cap day — these signals were visible but not wired in")

        if self.features_df is None or self.features_df.empty:
            print("  ⚠️ No feature data. Run train_v8() first.")
            return

        # Guard: skip if we don't have meaningful station data yet
        n_kjfk = self.features_df["obs_kjfk_temp"].notna().sum() \
                 if "obs_kjfk_temp" in self.features_df.columns else 0
        print(f"  obs_kjfk_temp: {n_kjfk} non-null rows")
        if n_kjfk < 5:
            print(f"  ⚠️ Only {n_kjfk} rows with KJFK data (need 5+). Run backfill_synoptic.py first.")
            print(f"     Skipping v9 training — will retry tomorrow after nightly backfill.")
            return

        missing_v9 = [c for c in FEATURE_COLS_V9 if c not in self.features_df.columns]
        if missing_v9:
            print(f"  Missing v9 cols (NaN for historical rows): {missing_v9}")
            for col in missing_v9:
                self.features_df[col] = np.nan

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts ({n_kjfk} have KJFK station data)")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v9 = forecast_df[FEATURE_COLS_V9]
        y_actual = forecast_df["actual_high"]
        nws_last  = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]

        # Same HRRR > NBM > AccuWeather > NWS base cascade as v7/v8
        hrrr_max = forecast_df.get("mm_hrrr_max", pd.Series(np.nan, index=forecast_df.index))
        nbm_max  = forecast_df.get("mm_nbm_max",  pd.Series(np.nan, index=forecast_df.index))
        base = hrrr_max.copy()
        base[base.isna()] = nbm_max[base.isna()].values
        base[base.isna()] = accu_last[base.isna()].values
        base[base.isna()] = nws_last[base.isna()].values
        y_bias = y_actual - base

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, bucket_acc, all_residuals = [], [], []

        for tr, te in tscv.split(X_v9):
            model = HistGradientBoostingRegressor(
                max_iter=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=20, l2_regularization=1.0,
                max_leaf_nodes=15, random_state=42,
            )
            model.fit(X_v9.iloc[tr], y_bias.iloc[tr])
            pred_bias = model.predict(X_v9.iloc[te])
            pred_temp = base.iloc[te].values + pred_bias
            mae_scores.append(mean_absolute_error(y_actual.iloc[te], pred_temp))
            all_residuals.extend((y_actual.iloc[te].values - pred_temp).tolist())
            pred_buckets   = [f"{int(p)}-{int(p)+1}" for p in pred_temp]
            actual_buckets = [f"{int(a)}-{int(a)+1}" for a in y_actual.iloc[te]]
            correct = sum(1 for pb, ab in zip(pred_buckets, actual_buckets) if pb == ab)
            bucket_acc.append(correct / len(actual_buckets))

        residual_std = float(np.std(all_residuals))
        print(f"  v9 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v9 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v9 Residual Std:  {residual_std:.2f}°F")

        v9_regressor = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=1.0,
            max_leaf_nodes=15, random_state=42,
        )
        v9_regressor.fit(X_v9, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V9, v9_regressor.feature_importances_))
            print(f"  obs_coastal_vs_inland importance: {fi.get('obs_coastal_vs_inland', 0):.4f}")
            print(f"  obs_kjfk_vs_knyc importance:     {fi.get('obs_kjfk_vs_knyc', 0):.4f}")
            print(f"  obs_kjfk_temp importance:         {fi.get('obs_kjfk_temp', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V9,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        prefix = self.model_prefix
        with open(f"{prefix}bcp_v9_regressor.pkl", "wb") as f:
            pickle.dump(v9_regressor, f)
        with open(f"{prefix}bcp_v9_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v9_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V9), f)

        v9_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v9_marine_cap_stations",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": "KJFK/KLGA/KEWR/KTEB temps + coastal-vs-inland gradient + JFK-KNYC diff",
            "motivation": "April 12 2026: model flipped to 56-57 on marine cap day — station signals now wired in",
            "n_kjfk_rows": int(n_kjfk),
            "v9_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V9),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v9": list(FEATURE_COLS_V9),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v9.json", "w") as f:
            _json.dump(v9_meta, f, indent=2)

        print(f"\n  ✅ Saved v9 models: bcp_v9_regressor.pkl, bcp_v9_classifier.pkl")
        print(f"  v9 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v8.json") as f:
                v8_meta = _j.load(f)
            v8_mae = v8_meta.get("v8_regression", {}).get("cv_mae")
            v8_bkt = v8_meta.get("v8_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v8 vs v9 (added: named station marine cap features)")
            if v8_mae: print(f"  MAE:        v8={v8_mae:.2f}°F → v9={np.mean(mae_scores):.2f}°F")
            if v8_bkt: print(f"  Bucket Acc: v8={v8_bkt:.1%} → v9={np.mean(bucket_acc):.1%}")
            print(f"  Station features are NaN for pre-backfill rows — accuracy improves as data accumulates.")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v10(self) -> None:
        """
        Train v10 model: v9 (149 features) + Manhattan Mesonet MANH (2 features = 151 total).

        MANH (NY Mesonet station near Columbia, ~125th St) updates every 5 minutes —
        the only sub-hourly near-Central Park station in our Synoptic pull.
        It fills the gap between KNYC's hourly :51 reports.

        obs_manh_temp:    raw temperature (°F) at 5-min resolution
        obs_manh_vs_knyc: MANH - KNYC: negative = sea breeze at 125th St before KNYC catches up

        Skips if fewer than 30 rows have obs_manh_temp (needs Synoptic backfill first).
        """
        from model_config import FEATURE_COLS_V10
        from train_classifier import BucketClassifier

        print(f"\n{'═'*60}")
        print(f"v10 Training: v9 + Manhattan Mesonet 5-min fill-in ({len(FEATURE_COLS_V10)} total features)")
        print(f"{'═'*60}")
        print(f"  New: obs_manh_temp, obs_manh_vs_knyc (MANH 5-min vs KNYC hourly)")

        if self.features_df is None or len(self.features_df) == 0:
            print("  ⚠️ No feature data — skipping v10.")
            return

        # Guard: skip if MANH data isn't backfilled yet
        n_manh = self.features_df["obs_manh_temp"].notna().sum() \
                 if "obs_manh_temp" in self.features_df.columns else 0
        print(f"  obs_manh_temp: {n_manh} non-null rows")
        if n_manh < 30:
            print(f"  ⚠️ Only {n_manh} rows with MANH data (need 30+). Run backfill after MANH goes live.")
            print(f"     Skipping v10 training — will retry tomorrow after nightly backfill.")
            return

        missing_v10 = [c for c in FEATURE_COLS_V10 if c not in self.features_df.columns]
        if missing_v10:
            print(f"  Missing v10 cols (NaN for historical rows): {missing_v10}")
            for col in missing_v10:
                self.features_df[col] = float("nan")

        has_forecast_mask = self.features_df["nws_last"].notna()
        forecast_df = self.features_df[has_forecast_mask].copy()
        n_forecast = len(forecast_df)
        print(f"\n  Regression: {n_forecast} rows with forecasts ({n_manh} have MANH station data)")

        if n_forecast < MIN_DAYS_FOR_TRAINING:
            print(f"  ⚠️ Need {MIN_DAYS_FOR_TRAINING} rows, have {n_forecast}. Skipping.")
            return

        X_v10 = forecast_df[FEATURE_COLS_V10]
        y_actual = forecast_df["actual_high"]
        nws_last  = forecast_df["nws_last"]
        accu_last = forecast_df["accu_last"]

        y_bias = y_actual - nws_last

        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        v10_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores   = -cross_val_score(v10_regressor, X_v10, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc   = cross_val_score(
            v10_regressor, X_v10, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1)),
        )
        residual_std = float(np.std(y_bias))

        print(f"  CV MAE: {np.mean(mae_scores):.2f}°F  |  CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        v10_regressor.fit(X_v10, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V10, v10_regressor.feature_importances_))
            print(f"  obs_manh_temp importance:      {fi.get('obs_manh_temp', 0):.4f}")
            print(f"  obs_manh_vs_knyc importance:   {fi.get('obs_manh_vs_knyc', 0):.4f}")
            print(f"  obs_coastal_vs_inland:         {fi.get('obs_coastal_vs_inland', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V10,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        prefix = self.model_prefix
        with open(f"{prefix}bcp_v10_regressor.pkl", "wb") as f:
            pickle.dump(v10_regressor, f)
        with open(f"{prefix}bcp_v10_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v10_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V10), f)

        v10_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v10_manh_mesonet",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": "MANH (Manhattan Mesonet 5-min) temp + MANH-KNYC diff",
            "motivation": "KNYC reports hourly; MANH fills the 59-min blind spot at 5-min resolution",
            "n_manh_rows": int(n_manh),
            "v10_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V10),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v10": list(FEATURE_COLS_V10),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v10.json", "w") as f:
            _json.dump(v10_meta, f, indent=2)

        print(f"\n  ✅ Saved v10 models: bcp_v10_regressor.pkl, bcp_v10_classifier.pkl")
        print(f"  v10 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v9.json") as f:
                v9_meta = _j.load(f)
            v9_mae = v9_meta.get("v9_regression", {}).get("cv_mae")
            v9_bkt = v9_meta.get("v9_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v9 vs v10 (added: MANH Manhattan Mesonet 5-min)")
            if v9_mae: print(f"  MAE:        v9={v9_mae:.2f}°F → v10={np.mean(mae_scores):.2f}°F")
            if v9_bkt: print(f"  Bucket Acc: v9={v9_bkt:.1%} → v10={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v11(self) -> None:
        """
        Train v11 model: v10 (151 features) + model-vs-NWS divergence (3 features = 154 total).

        New features:
          mm_hrrr_vs_nws  — HRRR - nws_last: how far #1-accuracy fast model is above/below NWS
          mm_nbm_vs_nws   — NBM  - nws_last: how far 50-model blend is above/below NWS
          mm_mean_vs_nws  — 7-model consensus - nws_last: unanimous fast-model divergence signal

        Motivation: NWS is ranked #11-16 by 90-day accuracy yet the model currently infers
        the HRRR-vs-NWS gap implicitly from separate mm_hrrr_max and nws_last columns.
        Explicit divergence features give the model a direct "how stale is NWS?" signal.
        Example: HRRR +5.6°F above NWS tonight → mm_hrrr_vs_nws = 5.6 → model confidently
        raises prediction above NWS without having to discover that relationship indirectly.
        """
        from model_config import FEATURE_COLS_V11
        from train_classifier import BucketClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        print(f"\n{'═'*60}")
        print(f"v11 Training: v10 + model-vs-NWS divergence ({len(FEATURE_COLS_V11)} total features)")
        print(f"{'═'*60}")

        # Compute the three derived divergence columns
        self._compute_model_vs_nws_features()

        if self.features_df.empty:
            print("  ⚠️ No feature data — skipping v11.")
            return

        prefix = self.model_prefix
        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()
        n_forecast = len(forecast_df)
        if n_forecast < 30:
            print(f"  ⚠️ Only {n_forecast} forecast rows (need 30+). Skipping v11.")
            return

        # Check how many rows have the new divergence features populated
        n_div = forecast_df["mm_hrrr_vs_nws"].notna().sum() if "mm_hrrr_vs_nws" in forecast_df.columns else 0
        print(f"  mm_hrrr_vs_nws: {n_div} non-null rows (of {n_forecast} forecast rows)")

        missing_v11 = [c for c in FEATURE_COLS_V11 if c not in self.features_df.columns]
        if missing_v11:
            print(f"  Missing v11 cols (NaN for historical rows): {missing_v11}")
            for col in missing_v11:
                self.features_df[col] = np.nan

        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()

        # Airtight gate: v11's model-vs-NWS divergence features need
        # populated multi-model data (HRRR/NBM/etc.). Sparse coverage
        # of mm_*_vs_nws drives the same overfit pattern as v13/v15.
        # v11 features come from prediction_logs.atm_snapshot, populated only
        # since live-multimodel capture began (~144 days). Lower threshold to
        # 100 — populated-and-real on 144 rows is far better than NaN-on-95%.
        # Cap will lift naturally as calendar fills, or via NOMADS HRRR archive
        # backfill (deferred — heavy multi-day project).
        gated = self._gate_and_filter_for_version(
            "v11",
            ["mm_hrrr_vs_nws", "mm_mean_vs_nws"],
            forecast_df,
            min_rows=100,
        )
        if gated is None:
            return
        forecast_df = gated

        X_v11    = forecast_df[FEATURE_COLS_V11]
        nws_last = forecast_df["nws_last"]
        y_actual = forecast_df["actual_high"]
        y_bias   = y_actual - nws_last   # train residual vs NWS (same target as v7-v10)

        v11_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores = -cross_val_score(v11_regressor, X_v11, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc = cross_val_score(
            v11_regressor, X_v11, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(
                np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1
            )),
        )
        residual_std = float(np.std(y_bias))

        print(f"  v11 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v11 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v11 Residual Std:  {residual_std:.2f}°F")
        v11_regressor.fit(X_v11, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V11, v11_regressor.feature_importances_))
            print(f"  mm_hrrr_vs_nws importance: {fi.get('mm_hrrr_vs_nws', 0):.4f}")
            print(f"  mm_nbm_vs_nws importance:  {fi.get('mm_nbm_vs_nws', 0):.4f}")
            print(f"  mm_mean_vs_nws importance: {fi.get('mm_mean_vs_nws', 0):.4f}")
            print(f"  mm_hrrr_max importance:    {fi.get('mm_hrrr_max', 0):.4f}")
            print(f"  nws_last importance:       {fi.get('nws_last', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V11,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        with open(f"{prefix}bcp_v11_regressor.pkl", "wb") as f:
            pickle.dump(v11_regressor, f)
        with open(f"{prefix}bcp_v11_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v11_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V11), f)

        v11_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v11_model_vs_nws_divergence",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": "mm_hrrr_vs_nws, mm_nbm_vs_nws, mm_mean_vs_nws",
            "motivation": "NWS is #11-16 accuracy; explicit model-vs-NWS divergence replaces "
                          "implicit inference from separate mm_hrrr_max + nws_last columns",
            "n_divergence_rows": int(n_div),
            "v11_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V11),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v11": list(FEATURE_COLS_V11),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v11.json", "w") as f:
            _json.dump(v11_meta, f, indent=2)

        print(f"\n  ✅ Saved v11 models: bcp_v11_regressor.pkl, bcp_v11_classifier.pkl")
        print(f"  v11 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v10.json") as f:
                v10_meta = _j.load(f)
            v10_mae = v10_meta.get("v10_regression", {}).get("cv_mae")
            v10_bkt = v10_meta.get("v10_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v10 vs v11 (added: model-vs-NWS divergence features)")
            if v10_mae: print(f"  MAE:        v10={v10_mae:.2f}°F → v11={np.mean(mae_scores):.2f}°F")
            if v10_bkt: print(f"  Bucket Acc: v10={v10_bkt:.1%} → v11={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v12(self) -> None:
        """
        Train v12 model: v11 (154 features) + deep NNJ inland stations (3 features = 157 total).

        New features:
          obs_kcdw_temp       — Caldwell NJ ASOS (~25mi inland from JFK)
          obs_ksmq_temp       — Somerville NJ ASOS (~35mi inland from JFK, deepest reference)
          obs_inland_gradient — KSMQ - KJFK: full 35-mile coastal-to-inland temperature spread

        Motivation: v9 uses EWR (~15mi) and TEB (~20mi) as the inland anchor for
        obs_coastal_vs_inland.  KCDW and KSMQ push 25-35mi into NJ interior where the sea
        breeze NEVER reaches even on cap days.  On a strong marine cap day the gradient
        (JFK=68°F, KSMQ=84°F) is +16°F — an unambiguous fingerprint that is completely
        invisible to v11.  This feature alone can distinguish "coast capped, inland free"
        from "uniform cap across all stations".
        """
        from model_config import FEATURE_COLS_V12
        from train_classifier import BucketClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        # Load Supabase snapshot data to ensure obs_ksmq_temp, obs_kcdw_temp are available
        # (written by backfill_synoptic.py into the atm_snapshot JSONB column)
        sb_snap_df = self._load_supabase_snapshot_features()
        if sb_snap_df is not None:
            self._merge_supabase_snapshots(sb_snap_df)

        print(f"\n{'═'*60}")
        print(f"v12 Training: v11 + deep NNJ inland stations ({len(FEATURE_COLS_V12)} total features)")
        print(f"{'═'*60}")

        if self.features_df.empty:
            print("  ⚠️ No feature data — skipping v12.")
            return

        prefix = self.model_prefix
        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()
        n_forecast = len(forecast_df)
        if n_forecast < 30:
            print(f"  ⚠️ Only {n_forecast} forecast rows (need 30+). Skipping v12.")
            return

        # Check how many rows have the new inland station data
        n_ksmq = forecast_df["obs_ksmq_temp"].notna().sum() if "obs_ksmq_temp" in forecast_df.columns else 0
        n_kcdw = forecast_df["obs_kcdw_temp"].notna().sum() if "obs_kcdw_temp" in forecast_df.columns else 0
        print(f"  obs_ksmq_temp: {n_ksmq} non-null rows (of {n_forecast} forecast rows)")
        print(f"  obs_kcdw_temp: {n_kcdw} non-null rows")
        if n_ksmq < 14:
            print(f"  ⚠️ Only {n_ksmq} rows with KSMQ data (need 14+) — "
                  f"run backfill_synoptic.py to populate history. Skipping v12.")
            return

        missing_v12 = [c for c in FEATURE_COLS_V12 if c not in self.features_df.columns]
        if missing_v12:
            print(f"  Missing v12 cols (NaN for historical rows): {missing_v12}")
            for col in missing_v12:
                self.features_df[col] = np.nan

        # Also derive the v11 divergence features (needed since we build on v11)
        self._compute_model_vs_nws_features()

        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()

        # Airtight gate: v12's deep-NJ inland features (KCDW/KSMQ).
        # Post-IEM backfill: 192 canonical rows have both. Threshold 100 lets
        # v12 deploy now; KCDW/KSMQ accumulate ~1 row/day going forward.
        gated = self._gate_and_filter_for_version(
            "v12",
            ["obs_kcdw_temp", "obs_ksmq_temp"],
            forecast_df,
            min_rows=100,
        )
        if gated is None:
            return
        forecast_df = gated

        X_v12    = forecast_df[FEATURE_COLS_V12]
        nws_last = forecast_df["nws_last"]
        y_actual = forecast_df["actual_high"]
        y_bias   = y_actual - nws_last   # train residual vs NWS (same target as v7-v11)

        v12_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores = -cross_val_score(v12_regressor, X_v12, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc = cross_val_score(
            v12_regressor, X_v12, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(
                np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1
            )),
        )
        residual_std = float(np.std(y_bias))

        print(f"  v12 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v12 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v12 Residual Std:  {residual_std:.2f}°F")
        v12_regressor.fit(X_v12, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V12, v12_regressor.feature_importances_))
            print(f"  obs_inland_gradient importance: {fi.get('obs_inland_gradient', 0):.4f}")
            print(f"  obs_ksmq_temp importance:       {fi.get('obs_ksmq_temp', 0):.4f}")
            print(f"  obs_kcdw_temp importance:       {fi.get('obs_kcdw_temp', 0):.4f}")
            print(f"  obs_coastal_vs_inland (v9):     {fi.get('obs_coastal_vs_inland', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V12,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        with open(f"{prefix}bcp_v12_regressor.pkl", "wb") as f:
            pickle.dump(v12_regressor, f)
        with open(f"{prefix}bcp_v12_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v12_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V12), f)

        v12_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v12_deep_nj_inland_stations",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": "obs_kcdw_temp, obs_ksmq_temp, obs_inland_gradient (KSMQ-JFK)",
            "motivation": (
                "v11's obs_coastal_vs_inland uses EWR/TEB as the inland anchor (~20mi). "
                "KCDW (~25mi) and KSMQ (~35mi) push the reference into NJ interior where "
                "the sea breeze never reaches, making the full coast-to-inland gradient "
                "visible for the first time. On strong cap days the KSMQ-JFK spread is "
                "+15-20°F — a signal completely hidden from v11."
            ),
            "n_ksmq_rows": int(n_ksmq),
            "n_kcdw_rows": int(n_kcdw),
            "v12_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V12),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v12": list(FEATURE_COLS_V12),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v12.json", "w") as f:
            _json.dump(v12_meta, f, indent=2)

        print(f"\n  ✅ Saved v12 models: bcp_v12_regressor.pkl, bcp_v12_classifier.pkl")
        print(f"  v12 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v11.json") as f:
                v11_meta = _j.load(f)
            v11_mae = v11_meta.get("v11_regression", {}).get("cv_mae")
            v11_bkt = v11_meta.get("v11_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v11 vs v12 (added: KCDW + KSMQ + inland_gradient)")
            if v11_mae: print(f"  MAE:        v11={v11_mae:.2f}°F → v12={np.mean(mae_scores):.2f}°F")
            if v11_bkt: print(f"  Bucket Acc: v11={v11_bkt:.1%} → v12={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v13(self) -> None:
        """
        Train v13 model: v12 (157 features) + BL safeguard features (3 features = 160 total).

        New features (computed derived features):
          entrainment_temp_diff  = atm_925mb_temp_mean - obs_latest_temp
          marine_containment     = obs_kjfk_vs_knyc / atm_bl_height_max
          inland_strength        = mean(obs_kteb_temp, obs_kcdw_temp, obs_ksmq_temp) - mm_mean

        Motivation (April 15, 2026):
          v12 BL height spike (+951m at 1:55 PM EDT) triggered downward revision (89-90 → 87-88)
          even though actual high reached 90°F. Root cause: BL trigger was conditional on other
          atmospheric state that wasn't being explicitly modeled.

          entrainment_temp_diff detects whether cool aloft air is actively mixing (new signal).
          marine_containment shows whether ocean air is penetrating inland or contained at coast.
          inland_strength verifies whether inland stations are beating the forecast (upside signal).

          With these 3 features, the model learns: "BL increase matters only if OTHER conditions
          align. Don't reduce forecast if entrainment is weak, marine is contained, inland is
          tracking well." This prevents the April 15 miss while still catching actual cap days.
        """
        from model_config import FEATURE_COLS_V13
        from train_classifier import BucketClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        print(f"\n{'═'*60}")
        print(f"v13 Training: v12 + BL safeguard features ({len(FEATURE_COLS_V13)} total features)")
        print(f"{'═'*60}")

        # Compute the three derived BL safeguard columns
        self._compute_bl_safeguard_features()

        if self.features_df.empty:
            print("  ⚠️ No feature data — skipping v13.")
            return

        prefix = self.model_prefix
        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()
        n_forecast = len(forecast_df)
        if n_forecast < 30:
            print(f"  ⚠️ Only {n_forecast} forecast rows (need 30+). Skipping v13.")
            return

        # Check how many rows have the new BL safeguard features populated
        n_entrainment = forecast_df["entrainment_temp_diff"].notna().sum() if "entrainment_temp_diff" in forecast_df.columns else 0
        n_marine = forecast_df["marine_containment"].notna().sum() if "marine_containment" in forecast_df.columns else 0
        n_inland = forecast_df["inland_strength"].notna().sum() if "inland_strength" in forecast_df.columns else 0
        print(f"  entrainment_temp_diff: {n_entrainment} non-null rows (of {n_forecast} forecast rows)")
        print(f"  marine_containment:    {n_marine} non-null rows")
        print(f"  inland_strength:       {n_inland} non-null rows")

        missing_v13 = [c for c in FEATURE_COLS_V13 if c not in self.features_df.columns]
        if missing_v13:
            print(f"  Missing v13 cols (NaN for historical rows): {missing_v13}")
            for col in missing_v13:
                self.features_df[col] = np.nan

        # Also derive the v12 (and earlier) features to ensure they're available
        self._compute_model_vs_nws_features()

        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()

        # Airtight gate: v13's BL safeguard features. Post-IEM+925mb backfill
        # the canonical-row coverage is still ~132 (training filter excludes
        # historical archive rows where 925mb data wasn't computed at canonical
        # write time). Threshold 100 — populated-and-real beats NaN-on-95%.
        # Coverage will lift naturally as canonical 925mb fetches accumulate.
        gated = self._gate_and_filter_for_version(
            "v13",
            ["entrainment_temp_diff", "marine_containment", "inland_strength"],
            forecast_df,
            min_rows=100,
        )
        if gated is None:
            return
        forecast_df = gated

        X_v13    = forecast_df[FEATURE_COLS_V13]
        nws_last = forecast_df["nws_last"]
        y_actual = forecast_df["actual_high"]
        y_bias   = y_actual - nws_last   # train residual vs NWS (same target as v7-v12)

        v13_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores = -cross_val_score(v13_regressor, X_v13, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc = cross_val_score(
            v13_regressor, X_v13, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(
                np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1
            )),
        )
        residual_std = float(np.std(y_bias))

        print(f"  v13 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v13 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v13 Residual Std:  {residual_std:.2f}°F")
        v13_regressor.fit(X_v13, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V13, v13_regressor.feature_importances_))
            print(f"  entrainment_temp_diff importance: {fi.get('entrainment_temp_diff', 0):.4f}")
            print(f"  marine_containment importance:    {fi.get('marine_containment', 0):.4f}")
            print(f"  inland_strength importance:       {fi.get('inland_strength', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V13,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        with open(f"{prefix}bcp_v13_regressor.pkl", "wb") as f:
            pickle.dump(v13_regressor, f)
        with open(f"{prefix}bcp_v13_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v13_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V13), f)

        v13_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v13_bl_safeguard_features",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": "entrainment_temp_diff, marine_containment, inland_strength",
            "motivation": (
                "April 15, 2026: v12 BL spike triggered downward revision (89-90 → 87-88) "
                "even though actual high was 90°F. Root cause: BL height increase was "
                "conditional on other atmospheric state not being explicitly modeled. "
                "These 3 features encode the guard rails directly, letting the model learn: "
                "'BL trigger matters only if entrainment is cooling, marine penetrates inland, "
                "and inland underperforms forecast. Otherwise, don't reduce.' This prevents "
                "the April 15 miss while still catching actual cap days."
            ),
            "n_entrainment_rows": int(n_entrainment),
            "n_marine_rows": int(n_marine),
            "n_inland_rows": int(n_inland),
            "v13_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V13),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v13": list(FEATURE_COLS_V13),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v13.json", "w") as f:
            _json.dump(v13_meta, f, indent=2)

        print(f"\n  ✅ Saved v13 models: bcp_v13_regressor.pkl, bcp_v13_classifier.pkl")
        print(f"  v13 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v12.json") as f:
                v12_meta = _j.load(f)
            v12_mae = v12_meta.get("v12_regression", {}).get("cv_mae")
            v12_bkt = v12_meta.get("v12_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v12 vs v13 (added: entrainment + marine + inland)")
            if v12_mae: print(f"  MAE:        v12={v12_mae:.2f}°F → v13={np.mean(mae_scores):.2f}°F")
            if v12_bkt: print(f"  Bucket Acc: v12={v12_bkt:.1%} → v13={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def _compute_blind_spot_features(self) -> None:
        """
        Derive v14 blind-spot interaction features from existing columns.

        These encode regime conjunctions the model can't easily learn from
        raw inputs alone given sparse training data (~281 rows, mostly
        morning-run snapshots). By exposing the conjunctions as direct
        features, a single tree split captures the pattern.

        hours_to_heating_close = max(0, 16 - obs_latest_hour)
        peak_to_hrrr_gap       = mm_hrrr_max - obs_max_so_far
        late_obs_below_pred    = (obs_latest_hour>=13) * max(0, peak_to_hrrr_gap)
        late_falling_signal    = (obs_latest_hour>=13) * obs_temp_falling_hrs
        mm_spread_late         = mm_spread * (obs_latest_hour>=12)

        All NaN-safe — when source columns are missing, derived columns are NaN
        and HistGradientBoosting handles them natively.
        """
        df = self.features_df

        def _coalesce(col_name, computed):
            existing = df[col_name] if col_name in df.columns else None
            if existing is None:
                df[col_name] = computed
            else:
                df[col_name] = computed.where(computed.notna(), existing)

        # 1) hours_to_heating_close
        if "obs_latest_hour" in df.columns:
            hr = pd.to_numeric(df["obs_latest_hour"], errors="coerce")
            computed = (16 - hr).clip(lower=0)
            _coalesce("hours_to_heating_close", computed)
        elif "hours_to_heating_close" not in df.columns:
            df["hours_to_heating_close"] = np.nan

        # 2) peak_to_hrrr_gap
        if "mm_hrrr_max" in df.columns and "obs_max_so_far" in df.columns:
            hrrr = pd.to_numeric(df["mm_hrrr_max"], errors="coerce")
            peak = pd.to_numeric(df["obs_max_so_far"], errors="coerce")
            _coalesce("peak_to_hrrr_gap", (hrrr - peak).round(1))
        elif "peak_to_hrrr_gap" not in df.columns:
            df["peak_to_hrrr_gap"] = np.nan

        # 3) late_obs_below_pred = (obs_latest_hour>=13) * max(0, peak_to_hrrr_gap)
        if "obs_latest_hour" in df.columns and "peak_to_hrrr_gap" in df.columns:
            hr = pd.to_numeric(df["obs_latest_hour"], errors="coerce")
            gap = pd.to_numeric(df["peak_to_hrrr_gap"], errors="coerce")
            late = (hr >= 13).astype(float)
            # NaN where either source is NaN — preserve missingness
            mask = hr.notna() & gap.notna()
            computed = pd.Series(np.nan, index=df.index)
            computed.loc[mask] = (late.loc[mask] * gap.loc[mask].clip(lower=0)).round(1)
            _coalesce("late_obs_below_pred", computed)
        elif "late_obs_below_pred" not in df.columns:
            df["late_obs_below_pred"] = np.nan

        # 4) late_falling_signal = (obs_latest_hour>=13) * obs_temp_falling_hrs
        if "obs_latest_hour" in df.columns and "obs_temp_falling_hrs" in df.columns:
            hr = pd.to_numeric(df["obs_latest_hour"], errors="coerce")
            falling = pd.to_numeric(df["obs_temp_falling_hrs"], errors="coerce")
            late = (hr >= 13).astype(float)
            mask = hr.notna() & falling.notna()
            computed = pd.Series(np.nan, index=df.index)
            computed.loc[mask] = (late.loc[mask] * falling.loc[mask]).round(1)
            _coalesce("late_falling_signal", computed)
        elif "late_falling_signal" not in df.columns:
            df["late_falling_signal"] = np.nan

        # 5) mm_spread_late = mm_spread * (obs_latest_hour>=12)
        if "obs_latest_hour" in df.columns and "mm_spread" in df.columns:
            hr = pd.to_numeric(df["obs_latest_hour"], errors="coerce")
            sp = pd.to_numeric(df["mm_spread"], errors="coerce")
            late = (hr >= 12).astype(float)
            mask = hr.notna() & sp.notna()
            computed = pd.Series(np.nan, index=df.index)
            computed.loc[mask] = (late.loc[mask] * sp.loc[mask]).round(2)
            _coalesce("mm_spread_late", computed)
        elif "mm_spread_late" not in df.columns:
            df["mm_spread_late"] = np.nan

        n_h2c = df["hours_to_heating_close"].notna().sum()
        n_pg = df["peak_to_hrrr_gap"].notna().sum()
        n_lobp = df["late_obs_below_pred"].notna().sum()
        n_lfs = df["late_falling_signal"].notna().sum()
        n_msl = df["mm_spread_late"].notna().sum()
        print(f"  v14 blind-spot coverage: h2c={n_h2c}, pg={n_pg}, lobp={n_lobp}, lfs={n_lfs}, msl={n_msl}")
        _record_coverage(self.model_prefix, "v14", {
            "hours_to_close": n_h2c, "peak_to_hrrr_gap": n_pg,
            "late_obs_below_pred": n_lobp, "late_falling_signal": n_lfs,
            "morning_signal_late": n_msl,
        })
        self.features_df = df

    def _compute_v15_features(self) -> None:
        """
        Derive v15 morning-applicable + autoregressive features.

        These features are NOT obs-gated, so they fire at the 7am canonical
        write — fixing v14's morning blind spot (v14 features all gate on
        hour >= 12/13).

        forecast_revision    = nws_last - nws_first (within forecast lifecycle)
        cap_violation_925    = max(0, nws_last - atm_925mb_temp_mean - 14)
                               Adiabatic lapse rate ceiling (~14°F per 1km
                               between surface and 925mb in dry air).
        yesterday_signed_miss = (prev day's) actual_high - nws_last
                               Autoregressive bias lag-1.
        rolling_3day_bias    = mean(signed_miss) over prior 3 calendar days
                               Smoothed short-horizon systematic miss.
        today_realized_error = NaN at training time (only relevant at
                               inference for cross-day tomorrow predictions).
                               Set to NaN in training matrix.

        All NaN-safe — HistGradientBoosting handles missing values natively.
        """
        df = self.features_df

        def _coalesce(col_name, computed):
            existing = df[col_name] if col_name in df.columns else None
            if existing is None:
                df[col_name] = computed
            else:
                df[col_name] = computed.where(computed.notna(), existing)

        # 1) forecast_revision = nws_last - nws_first
        if "nws_last" in df.columns and "nws_first" in df.columns:
            last = pd.to_numeric(df["nws_last"], errors="coerce")
            first = pd.to_numeric(df["nws_first"], errors="coerce")
            _coalesce("forecast_revision", (last - first).round(1))
        elif "forecast_revision" not in df.columns:
            df["forecast_revision"] = np.nan

        # The training matrix has multiple rows per cli_date — typically a
        # forecast row (with nws_first/nws_last but no actual_high) and a
        # multi-year atmospheric/actuals row (with actual_high + 925mb but
        # no nws_last). Earlier row-by-row computation produced ~0 coverage
        # because the cells we need are split across rows. We resolve this
        # by building a per-cli_date aggregate (max ignores NaN, and these
        # values shouldn't conflict within a date), then mapping back.
        # Note: training matrix uses `target_date` (not `cli_date`). cli_date
        # exists only on raw nws_df, not on the merged features_df we operate on.
        if "target_date" in df.columns:
            agg_cols = [c for c in [
                "nws_last", "actual_high",
                "atm_925mb_temp_mean", "atm_925mb_hrrr_mean",
            ] if c in df.columns]
            if agg_cols:
                _tmp = df[["target_date", *agg_cols]].copy()
                _tmp["target_date"] = pd.to_datetime(_tmp["target_date"], errors="coerce")
                for c in agg_cols:
                    _tmp[c] = pd.to_numeric(_tmp[c], errors="coerce")
                per_date = (
                    _tmp.dropna(subset=["target_date"])
                    .groupby("target_date", as_index=False)[agg_cols]
                    .max()
                    .sort_values("target_date")
                    .reset_index(drop=True)
                )
            else:
                per_date = pd.DataFrame(columns=["target_date"])
        else:
            per_date = pd.DataFrame(columns=["target_date"])

        # 2) cap_violation_925 = max(0, nws_last - 925mb_temp - 14)
        # Use atm_925mb_temp_mean (GFS Open-Meteo) primary, fall back to
        # atm_925mb_hrrr_mean (HRRR 3km) when GFS is missing.
        # Compute per-date so nws_last (forecast row) and 925mb (multi-year
        # row) line up via cli_date even when they live in different rows.
        if (
            not per_date.empty
            and "nws_last" in per_date.columns
            and ("atm_925mb_temp_mean" in per_date.columns or "atm_925mb_hrrr_mean" in per_date.columns)
        ):
            last_pd = per_date["nws_last"]
            t925_gfs = per_date["atm_925mb_temp_mean"] if "atm_925mb_temp_mean" in per_date.columns else pd.Series(np.nan, index=per_date.index)
            t925_hrrr = per_date["atm_925mb_hrrr_mean"] if "atm_925mb_hrrr_mean" in per_date.columns else pd.Series(np.nan, index=per_date.index)
            t925 = t925_gfs.where(t925_gfs.notna(), t925_hrrr)
            cap_per_date = (last_pd - t925 - 14.0).clip(lower=0).round(1)
            cap_lookup = {pd.Timestamp(d): v for d, v in zip(per_date["target_date"], cap_per_date) if pd.notna(v)}
            df_dates = pd.to_datetime(df["target_date"], errors="coerce")
            cap_vals = df_dates.map(cap_lookup)
            _coalesce("cap_violation_925", pd.Series(cap_vals, index=df.index))
        elif "cap_violation_925" not in df.columns:
            df["cap_violation_925"] = np.nan

        # 3+4) Autoregressive bias features — date-sorted lag.
        # signed_miss[d] = actual_high[d] - nws_last[d]
        # yesterday_signed_miss[d] = signed_miss[d-1]
        # rolling_3day_bias[d]     = mean(signed_miss[d-1..d-3])
        # Compute per-date so actual_high (multi-year row) and nws_last
        # (forecast row) for the SAME cli_date can produce signed_miss
        # even though they live in different rows of the training matrix.
        if (
            not per_date.empty
            and "actual_high" in per_date.columns
            and "nws_last" in per_date.columns
        ):
            daily = per_date[["target_date", "actual_high", "nws_last"]].copy()
            daily["signed_miss"] = daily["actual_high"] - daily["nws_last"]
            daily = daily.dropna(subset=["signed_miss"]).sort_values("target_date").reset_index(drop=True)
            if not daily.empty:
                daily["yesterday"] = daily["signed_miss"].shift(1)
                daily["roll3"] = daily["signed_miss"].shift(1).rolling(window=3, min_periods=2).mean()
                lookup_y = {pd.Timestamp(d): v for d, v in zip(daily["target_date"], daily["yesterday"])}
                lookup_r = {pd.Timestamp(d): v for d, v in zip(daily["target_date"], daily["roll3"])}
                df_dates = pd.to_datetime(df["target_date"], errors="coerce")
                yesterday_vals = df_dates.map(lookup_y)
                roll3_vals = df_dates.map(lookup_r)
                _coalesce("yesterday_signed_miss", pd.Series(yesterday_vals, index=df.index).round(1))
                _coalesce("rolling_3day_bias", pd.Series(roll3_vals, index=df.index).round(2))
            else:
                if "yesterday_signed_miss" not in df.columns:
                    df["yesterday_signed_miss"] = np.nan
                if "rolling_3day_bias" not in df.columns:
                    df["rolling_3day_bias"] = np.nan
        else:
            if "yesterday_signed_miss" not in df.columns:
                df["yesterday_signed_miss"] = np.nan
            if "rolling_3day_bias" not in df.columns:
                df["rolling_3day_bias"] = np.nan

        # 5) today_realized_error — only meaningful for cross-day tomorrow
        # predictions at inference. NaN at training time (we're training on
        # same-day prediction targets). HistGB handles NaN; keeping the
        # column ensures schema parity between train and inference.
        if "today_realized_error" not in df.columns:
            df["today_realized_error"] = np.nan

        n_fr = df["forecast_revision"].notna().sum()
        n_cap = df["cap_violation_925"].notna().sum()
        n_ysm = df["yesterday_signed_miss"].notna().sum()
        n_r3 = df["rolling_3day_bias"].notna().sum()
        print(f"  v15 morning/autoreg coverage: revision={n_fr}, cap={n_cap}, "
              f"yesterday={n_ysm}, rolling3={n_r3} (today_realized_error: inference-only)")
        _record_coverage(self.model_prefix, "v15", {
            "forecast_revision": n_fr, "cap_violation_925": n_cap,
            "yesterday_signed_miss": n_ysm, "rolling_3day_bias": n_r3,
        })
        self.features_df = df

    def train_v15(self) -> None:
        """
        Train v15 model: v14 (176 features) + 5 morning/autoreg features.

        Motivation:
          v14's blind-spot interactions all gate on obs_latest_hour >= 12/13,
          so they're zero at the 7am canonical write — but that's exactly
          when the bet recommendation gets locked in. v15 adds 5 features
          that fire at canonical time:
            - forecast_revision: NWS overnight revision direction
            - cap_violation_925: physical ceiling violation flag
            - yesterday_signed_miss: autoregressive lag-1 of NWS bias
            - rolling_3day_bias: smoothed recent systematic miss
            - today_realized_error: cross-day prior for tomorrow predictions

          These work alongside v14's late-day features. At canonical time,
          v15 features carry the load; at intraday override, v14 features
          dominate.
        """
        from model_config import FEATURE_COLS_V15
        from train_classifier import BucketClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        print(f"\n{'═'*60}")
        print(f"v15 Training: v14 + morning/autoreg features ({len(FEATURE_COLS_V15)} total features)")
        print(f"{'═'*60}")

        # Ensure all upstream feature derivations have run
        self._compute_bl_safeguard_features()
        self._compute_model_vs_nws_features()
        self._compute_blind_spot_features()
        self._compute_v15_features()

        if self.features_df.empty:
            print("  ⚠️ No feature data — skipping v15.")
            return

        prefix = self.model_prefix
        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()
        n_forecast = len(forecast_df)
        if n_forecast < 30:
            print(f"  ⚠️ Only {n_forecast} forecast rows (need 30+). Skipping v15.")
            return

        missing_v15 = [c for c in FEATURE_COLS_V15 if c not in self.features_df.columns]
        if missing_v15:
            print(f"  Missing v15 cols (NaN for historical rows): {missing_v15}")
            for col in missing_v15:
                self.features_df[col] = np.nan

        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()

        # Airtight gate: v15's autoreg features must have enough populated rows.
        # cap_violation_925 was previously the binding constraint (only 144 rows
        # — needs 925mb GFS coverage which we only started capturing recently).
        # Dropped from the gate: HistGradientBoosting handles its NaN natively,
        # and the other 3 autoreg features carry the load (3300+ rows each).
        # When 925mb data accumulates, cap_violation_925 starts firing organically.
        gated = self._gate_and_filter_for_version(
            "v15",
            ["forecast_revision", "yesterday_signed_miss", "rolling_3day_bias"],
            forecast_df,
            min_rows=500,
        )
        if gated is None:
            return
        forecast_df = gated

        X_v15    = forecast_df[FEATURE_COLS_V15]
        nws_last = forecast_df["nws_last"]
        y_actual = forecast_df["actual_high"]
        y_bias   = y_actual - nws_last

        v15_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores = -cross_val_score(v15_regressor, X_v15, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc = cross_val_score(
            v15_regressor, X_v15, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(
                np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1
            )),
        )
        residual_std = float(np.std(y_bias))

        print(f"  v15 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v15 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v15 Residual Std:  {residual_std:.2f}°F")
        v15_regressor.fit(X_v15, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V15, v15_regressor.feature_importances_))
            print(f"  forecast_revision importance:    {fi.get('forecast_revision', 0):.4f}")
            print(f"  cap_violation_925 importance:    {fi.get('cap_violation_925', 0):.4f}")
            print(f"  yesterday_signed_miss imp:       {fi.get('yesterday_signed_miss', 0):.4f}")
            print(f"  rolling_3day_bias importance:    {fi.get('rolling_3day_bias', 0):.4f}")
            print(f"  late_obs_below_pred (v14) imp:   {fi.get('late_obs_below_pred', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V15,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        with open(f"{prefix}bcp_v15_regressor.pkl", "wb") as f:
            pickle.dump(v15_regressor, f)
        with open(f"{prefix}bcp_v15_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v15_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V15), f)

        v15_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v15_morning_autoreg",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": ", ".join([
                "forecast_revision", "cap_violation_925",
                "yesterday_signed_miss", "rolling_3day_bias", "today_realized_error",
            ]),
            "motivation": (
                "v14 fixed late-day overrides but didn't help the 7am canonical "
                "write (where the bet gets locked). v15 adds morning-applicable + "
                "autoregressive features that fire at canonical time: NWS revision "
                "direction, adiabatic ceiling violation, yesterday's signed miss, "
                "rolling 3-day bias, and cross-day realized HRRR error for "
                "tomorrow predictions."
            ),
            "v15_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V15),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v15": list(FEATURE_COLS_V15),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v15.json", "w") as f:
            _json.dump(v15_meta, f, indent=2)

        print(f"\n  ✅ Saved v15 models: bcp_v15_regressor.pkl, bcp_v15_classifier.pkl")
        print(f"  v15 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v14.json") as f:
                v14_meta = _j.load(f)
            v14_mae = v14_meta.get("v14_regression", {}).get("cv_mae")
            v14_bkt = v14_meta.get("v14_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v14 vs v15 (added: 5 morning/autoreg features)")
            if v14_mae: print(f"  MAE:        v14={v14_mae:.2f}°F → v15={np.mean(mae_scores):.2f}°F")
            if v14_bkt: print(f"  Bucket Acc: v14={v14_bkt:.1%} → v15={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def train_v14(self) -> None:
        """
        Train v14 model: v13 (171 features) + 5 blind-spot interaction features.

        Motivation (April 28, 2026):
          v13 predicted "67° or more · STRONG BET 82%" at 3:36 PM with peak
          observed = 65°F and current = 62°F (already declining). The model
          had every raw signal it needed (obs_max_so_far, obs_latest_hour,
          obs_temp_falling_hrs, mm_hrrr_max) but no explicit interaction term
          tying them together. With ~281 sparse training rows, gradient
          boosting can't reliably learn the conjunction from raw signals.

          v14 makes the conjunction explicit:
            late_obs_below_pred = (hour>=13) * max(0, mm_hrrr_max - obs_max)
          This single feature would have flagged Apr 28 at hour=15 with
          gap=3 → late_obs_below_pred=3.0, a strong downshift signal that
          a single tree split captures cleanly.

          Plus 4 supporting interaction features (hours_to_heating_close,
          peak_to_hrrr_gap, late_falling_signal, mm_spread_late).
        """
        from model_config import FEATURE_COLS_V14
        from train_classifier import BucketClassifier
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        import numpy as np

        print(f"\n{'═'*60}")
        print(f"v14 Training: v13 + blind-spot features ({len(FEATURE_COLS_V14)} total features)")
        print(f"{'═'*60}")

        # Ensure v13's safeguards + v11's divergence + v14's blind-spots are derived
        self._compute_bl_safeguard_features()
        self._compute_model_vs_nws_features()
        self._compute_blind_spot_features()

        if self.features_df.empty:
            print("  ⚠️ No feature data — skipping v14.")
            return

        prefix = self.model_prefix
        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()
        n_forecast = len(forecast_df)
        if n_forecast < 30:
            print(f"  ⚠️ Only {n_forecast} forecast rows (need 30+). Skipping v14.")
            return

        missing_v14 = [c for c in FEATURE_COLS_V14 if c not in self.features_df.columns]
        if missing_v14:
            print(f"  Missing v14 cols (NaN for historical rows): {missing_v14}")
            for col in missing_v14:
                self.features_df[col] = np.nan

        forecast_df = self.features_df[self.features_df["nws_last"].notna()].copy()

        # Airtight gate: v14's blind-spot features come from DISJOINT row pools.
        # peak_to_hrrr_gap + late_obs_below_pred require mm_hrrr_max (canonical
        # writes, ~144 rows). late_falling_signal is computed on intraday
        # snapshots (~3000 rows). AND-intersecting all three gives 0 rows.
        # Gate only on the canonical-pool pair (both ~144 rows, same pool).
        # late_falling_signal stays in FEATURE_COLS_V14 — HistGB handles its
        # NaN natively on the canonical training rows. Threshold 100 since
        # peak_to_hrrr_gap fills naturally as we capture more days.
        gated = self._gate_and_filter_for_version(
            "v14",
            ["peak_to_hrrr_gap", "late_obs_below_pred"],
            forecast_df,
            min_rows=100,
        )
        if gated is None:
            return
        forecast_df = gated

        X_v14    = forecast_df[FEATURE_COLS_V14]
        nws_last = forecast_df["nws_last"]
        y_actual = forecast_df["actual_high"]
        y_bias   = y_actual - nws_last

        v14_regressor = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=4,
            min_samples_leaf=6, l2_regularization=0.3,
            random_state=42,
        )

        mae_scores = -cross_val_score(v14_regressor, X_v14, y_bias, cv=5, scoring="neg_mean_absolute_error")
        bucket_acc = cross_val_score(
            v14_regressor, X_v14, y_bias, cv=5,
            scoring=lambda est, X, y: float(np.mean(
                np.abs((est.predict(X) + nws_last.iloc[:len(X)]) - y_actual.iloc[:len(X)]) <= 1
            )),
        )
        residual_std = float(np.std(y_bias))

        print(f"  v14 CV MAE:        {np.mean(mae_scores):.2f}°F")
        print(f"  v14 CV Bucket Acc: {np.mean(bucket_acc):.1%}")
        print(f"  v14 Residual Std:  {residual_std:.2f}°F")
        v14_regressor.fit(X_v14, y_bias)

        try:
            fi = dict(zip(FEATURE_COLS_V14, v14_regressor.feature_importances_))
            print(f"  late_obs_below_pred importance:    {fi.get('late_obs_below_pred', 0):.4f}")
            print(f"  peak_to_hrrr_gap importance:       {fi.get('peak_to_hrrr_gap', 0):.4f}")
            print(f"  hours_to_heating_close importance: {fi.get('hours_to_heating_close', 0):.4f}")
            print(f"  late_falling_signal importance:    {fi.get('late_falling_signal', 0):.4f}")
            print(f"  mm_spread_late importance:         {fi.get('mm_spread_late', 0):.4f}")
        except Exception:
            pass

        classifier = BucketClassifier()
        classifier.train(
            self.features_df.copy().reset_index(drop=True),
            feature_cols=FEATURE_COLS_V14,
            residual_std=residual_std,
            forecast_weight=5.0,
        )

        with open(f"{prefix}bcp_v14_regressor.pkl", "wb") as f:
            pickle.dump(v14_regressor, f)
        with open(f"{prefix}bcp_v14_classifier.pkl", "wb") as f:
            pickle.dump(classifier, f)
        with open(f"{prefix}bcp_v14_feature_cols.pkl", "wb") as f:
            pickle.dump(list(FEATURE_COLS_V14), f)

        v14_meta = {
            "trained_on": datetime.now().isoformat(),
            "version": "v14_blind_spot_features",
            "base_cascade": "HRRR > NBM > AccuWeather > NWS",
            "new_features": ", ".join([
                "hours_to_heating_close", "peak_to_hrrr_gap",
                "late_obs_below_pred", "late_falling_signal", "mm_spread_late",
            ]),
            "motivation": (
                "April 28, 2026: v13 predicted '67° or more · STRONG BET' at 3:36 PM "
                "with peak observed = 65°F and current = 62°F. Model had raw signals "
                "but no explicit interaction terms tying late-hour + observed-deficit. "
                "v14 adds 5 blind-spot interaction features so a single tree split "
                "captures the conjunction. late_obs_below_pred is the headline feature: "
                "(obs_latest_hour>=13) * max(0, mm_hrrr_max - obs_max_so_far)."
            ),
            "v14_regression": {
                "cv_mae": float(np.mean(mae_scores)),
                "cv_bucket_accuracy": float(np.mean(bucket_acc)),
                "residual_std": residual_std,
                "n_features": len(FEATURE_COLS_V14),
                "n_training_rows": n_forecast,
            },
            "feature_columns_v14": list(FEATURE_COLS_V14),
        }
        import json as _json
        with open(f"{prefix}model_metadata_v14.json", "w") as f:
            _json.dump(v14_meta, f, indent=2)

        print(f"\n  ✅ Saved v14 models: bcp_v14_regressor.pkl, bcp_v14_classifier.pkl")
        print(f"  v14 Classifier Bucket Acc: {classifier.cv_bucket_accuracy:.1%}")

        try:
            import json as _j
            with open(f"{prefix}model_metadata_v13.json") as f:
                v13_meta = _j.load(f)
            v13_mae = v13_meta.get("v13_regression", {}).get("cv_mae")
            v13_bkt = v13_meta.get("v13_regression", {}).get("cv_bucket_accuracy")
            print(f"\n{'─'*50}")
            print(f"COMPARISON: v13 vs v14 (added: 5 blind-spot interaction features)")
            if v13_mae: print(f"  MAE:        v13={v13_mae:.2f}°F → v14={np.mean(mae_scores):.2f}°F")
            if v13_bkt: print(f"  Bucket Acc: v13={v13_bkt:.1%} → v14={np.mean(bucket_acc):.1%}")
            print(f"{'─'*50}")
        except Exception:
            pass

    def run(self, v2: bool = False, v4: bool = False, v5: bool = False, v6: bool = False, v7: bool = False, v8: bool = False, v9: bool = False, v10: bool = False, v11: bool = False, v12: bool = False, v13: bool = False, v14: bool = False, v15: bool = False):
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
                self.train_v2()
                self.train_v3()
            self.train_v4()

        if v5:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            self.train_v5()

        if v6:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            self.train_v6()

        if v7:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            self.train_v7()

        if v8:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            self.train_v8()

        if v9:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            self.train_v9()

        if v10:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            self.train_v10()

        if v11:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            self.train_v11()

        if v12:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            if not v11:
                self.train_v11()
            self.train_v12()

        if v13:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            if not v10:
                self.train_v10()
            if not v11:
                self.train_v11()
            if not v12:
                self.train_v12()
            self.train_v13()

        if v14:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            if not v10:
                self.train_v10()
            if not v11:
                self.train_v11()
            if not v12:
                self.train_v12()
            if not v13:
                self.train_v13()
            self.train_v14()

        if v15:
            if not v2:
                self.train_v2()
                self.train_v3()
            if not v4:
                self.train_v4()
            if not v5:
                self.train_v5()
            if not v6:
                self.train_v6()
            if not v7:
                self.train_v7()
            if not v8:
                self.train_v8()
            if not v9:
                self.train_v9()
            if not v10:
                self.train_v10()
            if not v11:
                self.train_v11()
            if not v12:
                self.train_v12()
            if not v13:
                self.train_v13()
            if not v14:
                self.train_v14()
            self.train_v15()

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
    parser.add_argument("--v5", action="store_true",
                        help="Train v5 (v4 + high-timing features: overnight/late-day high detection)")
    parser.add_argument("--v6", action="store_true",
                        help="Train v6 (v5 + NBM, GEM HRDPS, HRRR 925mb, OKX radiosonde soundings)")
    parser.add_argument("--v7", action="store_true",
                        help="Train v7 (v6 features + HRRR-anchored base: HRRR > NBM > AccuWeather > NWS)")
    parser.add_argument("--v8", action="store_true",
                        help="Train v8 (v7 + obs_heating_rate_delta stall signal: deceleration = cap fingerprint)")
    parser.add_argument("--v9", action="store_true",
                        help="Train v9 (v8 + named ASOS station features: KJFK/KLGA/KEWR/KTEB temps, "
                             "coastal-vs-inland gradient, JFK-KNYC diff — the April 12 marine cap fix)")
    parser.add_argument("--v10", action="store_true",
                        help="Train v10 (v9 + Manhattan Mesonet MANH: 5-min fill-in between KNYC hourly reports)")
    parser.add_argument("--v11", action="store_true",
                        help="Train v11 (v10 + model-vs-NWS divergence: mm_hrrr_vs_nws, mm_nbm_vs_nws, mm_mean_vs_nws)")
    parser.add_argument("--v12", action="store_true",
                        help="Train v12 (v11 + deep NNJ inland stations: KCDW ~25mi, KSMQ ~35mi, obs_inland_gradient)")
    parser.add_argument("--v13", action="store_true",
                        help="Train v13 (v12 + BL safeguard features: entrainment_temp_diff, marine_containment, inland_strength)")
    parser.add_argument("--v14", action="store_true",
                        help="Train v14 (v13 + blind-spot interaction features: late_obs_below_pred, peak_to_hrrr_gap, "
                             "hours_to_heating_close, late_falling_signal, mm_spread_late — explicit regime conjunctions)")
    parser.add_argument("--v15", action="store_true",
                        help="Train v15 (v14 + morning/autoreg features: forecast_revision, cap_violation_925, "
                             "yesterday_signed_miss, rolling_3day_bias, today_realized_error — fires at canonical time)")
    args = parser.parse_args()

    if args.all:
        from city_config import CITIES
        for city_key in CITIES:
            print(f"\n{'#' * 60}")
            print(f"# Training: {city_key}")
            print(f"{'#' * 60}\n")
            trainer = NYCTemperatureModelTrainer(city_key=city_key)
            trainer.run(
                v2=args.v2, v4=args.v4, v5=args.v5,
                v6=getattr(args, "v6", False),
                v7=getattr(args, "v7", False),
                v8=getattr(args, "v8", False),
                v9=getattr(args, "v9", False),
                v10=getattr(args, "v10", False),
                v11=getattr(args, "v11", False),
                v12=getattr(args, "v12", False),
                v13=getattr(args, "v13", False),
                v14=getattr(args, "v14", False),
                v15=getattr(args, "v15", False),
            )
    else:
        trainer = NYCTemperatureModelTrainer(city_key=args.city)
        trainer.run(
            v2=args.v2, v4=args.v4, v5=args.v5,
            v6=getattr(args, "v6", False),
            v7=getattr(args, "v7", False),
            v8=getattr(args, "v8", False),
            v9=getattr(args, "v9", False),
            v10=getattr(args, "v10", False),
            v11=getattr(args, "v11", False),
            v12=getattr(args, "v12", False),
            v13=getattr(args, "v13", False),
            v14=getattr(args, "v14", False),
            v15=getattr(args, "v15", False),
        )
