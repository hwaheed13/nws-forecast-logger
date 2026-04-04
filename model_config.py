# model_config.py — single source of truth for ML model feature columns
# Imported by train_models.py, predict.py, prediction_writer.py, and api.py

import math

# ═══════════════════════════════════════════════════════════════════════
# v1 features (30 columns) — backward compatible, used by existing models
# ═══════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # NWS forecast statistics
    "nws_first", "nws_last", "nws_max", "nws_min", "nws_mean",
    "nws_spread", "nws_std", "nws_trend", "nws_count",
    "forecast_velocity", "forecast_acceleration",
    # AccuWeather forecast statistics (parity with NWS)
    "accu_first", "accu_last", "accu_max", "accu_min", "accu_mean",
    "accu_spread", "accu_std", "accu_trend", "accu_count",
    # Cross-source features
    "nws_accu_spread", "nws_accu_mean_diff",
    # Temporal features (cyclical + categorical)
    "day_of_year_sin", "day_of_year_cos",
    "month", "is_summer", "is_winter",
    # Rolling bias from prior completed days
    "rolling_bias_7d", "rolling_bias_21d",
    # Rolling ML model self-error (how wrong has the ML been recently)
    "rolling_ml_error_7d",
    # Data availability flag
    "has_accu_data",
]

# AccuWeather columns that fall back to NWS equivalents when missing
ACCU_NWS_FALLBACK = {
    "accu_first": "nws_first",
    "accu_last": "nws_last",
    "accu_max": "nws_max",
    "accu_min": "nws_min",
    "accu_mean": "nws_mean",
}

# ═══════════════════════════════════════════════════════════════════════
# v2 atmospheric features (27 columns) — from Open-Meteo APIs + NWS MOS
# ═══════════════════════════════════════════════════════════════════════

# Observed atmospheric conditions (17 features)
# Source: Open-Meteo archive (historical) or forecast API (live)
ATMOSPHERIC_COLS = [
    "atm_wind_max",           # Max wind speed (mph) — high wind = temp moderation
    "atm_wind_mean",          # Mean wind speed (mph)
    "atm_wind_dir_sin",       # Wind direction sin — onshore vs offshore
    "atm_wind_dir_cos",       # Wind direction cos
    "atm_humidity_mean",      # Mean relative humidity (%) — dry air = volatile temps
    "atm_humidity_min",       # Min humidity (%) — afternoon dryness
    "atm_dewpoint_mean",      # Mean dewpoint (°F) — moisture content
    "atm_pressure_mean",      # Mean surface pressure (hPa)
    "atm_pressure_change",    # Pressure change over day (hPa) — falling = approaching system
    "atm_cloud_cover_mean",   # Mean cloud cover (%) — clouds cap heating
    "atm_cloud_cover_max",    # Max cloud cover (%)
    "atm_precip_total",       # Total precipitation (inches) — rain caps daytime high
    "atm_temp_range",         # Daily temp range (°F) — volatility proxy
    "atm_overnight_min",      # Overnight minimum (midnight-8am) — baseline
    "atm_morning_temp_6am",   # Temperature at 6am — starting point
    "atm_850mb_temp_max",     # Max 850mb temperature (daytime 10am-6pm) — warm air advection aloft
    "atm_850mb_temp_mean",    # Mean 850mb temperature (daytime 10am-6pm)
]

# Ensemble uncertainty features (5 features)
# Source: Open-Meteo ECMWF 51-member ensemble (live forecast only, NaN for historical)
ENSEMBLE_COLS = [
    "ens_spread",             # Max - min daily high across 51 members
    "ens_std",                # Std dev across members — uncertainty width
    "ens_iqr",                # IQR (p75 - p25) — robust spread
    "ens_mean",               # Mean of ensemble members
    "ens_skew",               # Skewness — asymmetric risk
]

# Multi-model cross-comparison features (4 features)
# Source: Open-Meteo ECMWF, GFS, ICON, GEM models (live forecast only, NaN for historical)
MULTIMODEL_COLS = [
    "mm_spread",              # Max model - min model daily high
    "mm_std",                 # Std dev across models
    "mm_mean",                # Multi-model mean consensus
    "mm_ecmwf_gfs_diff",      # ECMWF - GFS difference — persistent model bias
]

# Intraday temperature curve features (10 features)
# Source: Open-Meteo archive hourly temperature_2m (historical) or NWS observations (live)
# These capture the SHAPE of the daily heating curve — crucial for predicting
# whether the actual high will overshoot or undershoot the forecast.
INTRADAY_CURVE_COLS = [
    "intra_temp_9am",             # Temperature at 9am — morning baseline after sunrise heating
    "intra_temp_noon",            # Temperature at noon — midday check
    "intra_temp_3pm",             # Temperature at 3pm — near typical peak
    "intra_temp_5pm",             # Temperature at 5pm — late afternoon (late push detection)
    "intra_heating_rate_am",      # (noon - 9am) / 3 = °F/hr morning heating rate
    "intra_heating_rate_pm",      # (3pm - noon) / 3 = °F/hr afternoon heating rate
    "intra_peak_hour",            # Hour when max temperature occurred (0-23)
    "intra_late_heating",         # 5pm - 3pm: positive = still warming late (midnight push signal)
    "intra_rise_from_overnight",  # 9am temp - overnight min: morning warmup magnitude
    "intra_high_vs_noon",         # actual daily max - noon temp: how much heating after noon
]

# Overnight carryover detection features (3 features)
# Helps the model handle days where the CLI actual high is an overnight
# carryover from the previous warm day, not the daytime peak.
OVERNIGHT_CARRYOVER_COLS = [
    "prev_day_high",          # Yesterday's actual high (°F) — from CSV or Open-Meteo
    "prev_day_temp_drop",     # prev_day_high - nws_last: large positive = potential carryover
    "midnight_temp",          # Temperature at midnight (12am) from Open-Meteo hourly
]

# Atmospheric predictor output features (2 features)
# Source: First-stage ML model trained on 1,278 historical days
# Learns: atmospheric_conditions + season → actual daily high
# The classifier gets these as features so it knows what the atmosphere
# "expects" vs what the forecast says — when they diverge, the forecast
# is more likely to be wrong.
ATM_PREDICTOR_COLS = [
    "atm_predicted_high",       # Atmospheric model's predicted daily high (°F)
    "atm_vs_forecast_diff",     # nws_last - atm_predicted_high: positive = NWS higher than atmosphere
]

# NWS MOS (Model Output Statistics) features (1 feature)
# Source: NWS MEX product (live inference only, NaN for historical/backfill)
# MOS provides an independent statistical post-processing of GFS model output.
MOS_COLS = [
    "mos_max_temp",             # MOS predicted max temperature for the target date
]

# NWS real-time observation features (12 features)
# Source: NWS station observations API (live), Open-Meteo archive hourly (training proxy)
# These provide GROUND TRUTH during live inference — the delta between forecasted
# and observed conditions is where the strongest signal lives.
# For D1 (tomorrow) predictions: all obs_* features are NaN except obs_temp_vs_forecast_max.
OBSERVATION_COLS = [
    "obs_latest_temp",          # Most recent observed temp (°F) from NWS station
    "obs_latest_hour",          # Hour (0-23) of latest observation (local time)
    "obs_max_so_far",           # Running daily max from hourly observations
    "obs_6hr_max",              # NWS official 6-hour max (reported at :51 every 6 hrs)
    "obs_vs_intra_forecast",    # obs_latest_temp - Open-Meteo forecasted temp at same hour
    "obs_wind_speed",           # Observed wind speed (mph)
    "obs_wind_gust",            # Observed wind gust (mph)
    "obs_wind_dir_sin",         # Wind direction circular encoding (sin component)
    "obs_wind_dir_cos",         # Wind direction circular encoding (cos component)
    "obs_cloud_cover",          # Mapped from NWS textDescription (0.0=Clear → 1.0=Overcast)
    "obs_heating_rate",         # Observed heating trajectory (°F/hr over last 3 hours)
    "obs_temp_vs_forecast_max", # obs_max_so_far - nws_last: how reality compares to forecast
]

# Features used as INPUT to the atmospheric predictor (first-stage model)
ATM_PREDICTOR_INPUT_COLS = ATMOSPHERIC_COLS + INTRADAY_CURVE_COLS + [
    "day_of_year_sin", "day_of_year_cos", "month", "is_summer", "is_winter",
    "midnight_temp",
]

# Combined v2 feature list (72 total: 30 + 17 + 5 + 4 + 10 + 3 + 2 + 1)
FEATURE_COLS_V2 = FEATURE_COLS + ATMOSPHERIC_COLS + ENSEMBLE_COLS + MULTIMODEL_COLS + INTRADAY_CURVE_COLS + OVERNIGHT_CARRYOVER_COLS + ATM_PREDICTOR_COLS + MOS_COLS

# v3 unified feature list (72 total — same features as v2)
# The difference is architectural: v3 trains a SINGLE regression model on ALL
# data (1,540+ days) predicting actual_high directly, instead of separate
# regression + classifier.  HistGradientBoosting handles NaN forecast features
# natively for multi-year rows.
FEATURE_COLS_V3 = FEATURE_COLS_V2

# v4 feature list (84 total: 72 + 12 observation features)
# Adds real-time NWS observation features that give the model ground-truth
# temperature/wind/sky data to compare against forecasts during live inference.
FEATURE_COLS_V4 = FEATURE_COLS_V3 + OBSERVATION_COLS

# Additional features added per-candidate-bucket during classification (4)
# These are NOT in FEATURE_COLS_V2 because they vary per candidate bucket, not per day
BUCKET_POSITION_COLS = [
    "bucket_center",          # Center temperature of candidate bucket
    "dist_from_prediction",   # Distance from regression prediction to bucket center
    "dist_from_accu",         # Distance from AccuWeather forecast to bucket center
    "dist_from_nws",          # Distance from NWS forecast to bucket center
]


# ═══════════════════════════════════════════════════════════════════════
# Probability functions
# ═══════════════════════════════════════════════════════════════════════

def norm_cdf(x, mu, sigma):
    """Gaussian CDF using math.erf (no scipy dependency)."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def derive_bucket_probabilities(predicted_temp, residual_std, spread=8):
    """
    Derive Kalshi bucket probabilities from a Gaussian centered on predicted_temp.

    Each bucket is a 1-degree range [low, low+1). Returns dict like {"42-43": 0.27, ...}
    with probabilities > 0.001, covering ±spread degrees around the prediction.
    """
    center = int(round(predicted_temp))
    buckets = {}
    for low in range(center - spread, center + spread + 1):
        high = low + 1
        p = norm_cdf(high, predicted_temp, residual_std) - \
            norm_cdf(low, predicted_temp, residual_std)
        if p > 0.001:
            buckets[f"{low}-{high}"] = round(p, 4)
    return buckets


def temp_to_bucket_label(temp_f: float) -> str:
    """Convert temperature to Kalshi bucket label like '48-49'."""
    low = int(math.floor(temp_f))
    return f"{low}-{low + 1}"


def get_candidate_buckets(center_temp: float, n_neighbors: int = 3) -> list[str]:
    """Return list of bucket labels around center_temp (2*n+1 buckets)."""
    center = int(round(center_temp))
    return [f"{b}-{b + 1}" for b in range(center - n_neighbors, center + n_neighbors + 1)]
