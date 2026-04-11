# model_config.py — single source of truth for ML model feature columns
# Imported by train_models.py, predict.py, prediction_writer.py, and api.py

import math

# ═══════════════════════════════════════════════════════════════════════
# v1 features (31 columns) — backward compatible, used by existing models
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
# v2 atmospheric features — from Open-Meteo APIs + NWS MOS
# ═══════════════════════════════════════════════════════════════════════

# Observed atmospheric conditions (23 features)
ATMOSPHERIC_COLS = [
    "atm_wind_max",           # Max wind speed (mph)
    "atm_wind_mean",          # Mean wind speed (mph)
    "atm_wind_dir_sin",       # Wind direction sin — onshore vs offshore
    "atm_wind_dir_cos",       # Wind direction cos
    "atm_humidity_mean",      # Mean relative humidity (%)
    "atm_humidity_min",       # Min humidity (%) — afternoon dryness
    "atm_dewpoint_mean",      # Mean dewpoint (°F)
    "atm_pressure_mean",      # Mean surface pressure (hPa)
    "atm_pressure_change",    # Pressure change over day (hPa)
    "atm_cloud_cover_mean",   # Mean cloud cover (%)
    "atm_cloud_cover_max",    # Max cloud cover (%)
    "atm_precip_total",       # Total precipitation (inches)
    "atm_temp_range",         # Daily temp range (°F)
    "atm_overnight_min",      # Overnight minimum (midnight-8am)
    "atm_morning_temp_6am",   # Temperature at 6am
    "atm_850mb_temp_max",     # Max 850mb temperature (daytime 10am-6pm)
    "atm_850mb_temp_mean",    # Mean 850mb temperature (daytime 10am-6pm)
    "atm_925mb_temp_max",     # Max 925mb temperature — surface-adjacent warm advection
    "atm_925mb_temp_mean",    # Mean 925mb temperature
    "atm_solar_radiation_peak",  # Peak solar irradiance midday (W/m²)
    "atm_solar_radiation_mean",  # Mean solar irradiance midday (W/m²)
    "atm_bl_height_max",      # Max planetary boundary layer height (m) 10am-4pm
    "atm_bl_height_mean",     # Mean PBL height during peak heating 10am-4pm
]

# Ensemble uncertainty features (5 features)
ENSEMBLE_COLS = [
    "ens_spread",             # Max - min daily high across 51 members
    "ens_std",                # Std dev across members
    "ens_iqr",                # IQR (p75 - p25)
    "ens_mean",               # Mean of ensemble members
    "ens_skew",               # Skewness — asymmetric risk
]

# Multi-model cross-comparison features (11 features)
# ECMWF, GFS, ICON (German DWD), GEM (Canadian CMC), HRRR (NCEP mesoscale)
# All 5 models fetched via Open-Meteo. Previously ICON and GEM were fetched
# but dropped — now exposed as individual features.
MULTIMODEL_COLS = [
    "mm_spread",              # Max model - min model daily high (all 5 models)
    "mm_std",                 # Std dev across models
    "mm_mean",                # Multi-model consensus mean
    "mm_ecmwf_gfs_diff",      # ECMWF - GFS difference
    "mm_hrrr_max",            # HRRR predicted daily max (warm-biased mesoscale)
    "mm_hrrr_ecmwf_diff",     # HRRR - ECMWF: positive = HRRR overmixing
    "mm_hrrr_gfs_diff",       # HRRR - GFS: mesoscale vs synoptic agreement
    "mm_icon_max",            # ICON (German DWD) predicted daily max
    "mm_gem_max",             # GEM (Canadian CMC) predicted daily max
    "mm_icon_gfs_diff",       # ICON - GFS: European vs American disagreement
    "mm_gem_ecmwf_diff",      # GEM - ECMWF: Canadian vs European disagreement
]

# Intraday temperature curve features (10 features)
INTRADAY_CURVE_COLS = [
    "intra_temp_9am",
    "intra_temp_noon",
    "intra_temp_3pm",
    "intra_temp_5pm",
    "intra_heating_rate_am",
    "intra_heating_rate_pm",
    "intra_peak_hour",
    "intra_late_heating",
    "intra_rise_from_overnight",
    "intra_high_vs_noon",
]

# Overnight carryover detection features (3 features)
OVERNIGHT_CARRYOVER_COLS = [
    "prev_day_high",
    "prev_day_temp_drop",
    "midnight_temp",
]

# Atmospheric predictor output features (2 features)
ATM_PREDICTOR_COLS = [
    "atm_predicted_high",
    "atm_vs_forecast_diff",
]

# NWS MOS features (1 feature)
MOS_COLS = [
    "mos_max_temp",
]

# Intraday forecast revision features (2 features)
FORECAST_REVISION_COLS = [
    "nws_post_9am_delta",
    "accu_post_9am_delta",
]

# NWS real-time observation features — KNYC primary station (12 features)
OBSERVATION_COLS = [
    "obs_latest_temp",          # Most recent observed temp (°F) from NWS KNYC
    "obs_latest_hour",          # Hour (0-23) of latest observation (local time)
    "obs_max_so_far",           # Running daily max from hourly observations
    "obs_6hr_max",              # NWS official 6-hour max
    "obs_vs_intra_forecast",    # obs_latest_temp - Open-Meteo forecasted temp at same hour
    "obs_wind_speed",           # Observed wind speed (mph)
    "obs_wind_gust",            # Observed wind gust (mph)
    "obs_wind_dir_sin",         # Wind direction circular encoding (sin)
    "obs_wind_dir_cos",         # Wind direction circular encoding (cos)
    "obs_cloud_cover",          # Sky condition mapped to 0.0=Clear → 1.0=Overcast
    "obs_heating_rate",         # °F/hr over last 3 hours
    "obs_temp_vs_forecast_max", # obs_max_so_far - nws_last
]

# Regional NYC-metro NWS station features (5 features)
# KJFK + KLGA supplement KNYC to capture mesoscale gradients.
# Sea-breeze days: JFK often 3-5°F colder than Central Park by midday.
REGIONAL_OBS_COLS = [
    "obs_jfk_temp",             # Latest KJFK (JFK Airport) observed temp (°F)
    "obs_lga_temp",             # Latest KLGA (LaGuardia) observed temp (°F)
    "obs_regional_spread",      # max(KNYC,JFK,LGA) - min: mesoscale gradient width
    "obs_regional_mean",        # Mean of all available NYC-area NWS station temps
    "obs_regional_vs_nws",      # obs_regional_mean - nws_last: metro reality vs forecast
]

# NWS forecast sequence features (2 features)
# Captures overnight NWS jumps — when D0 morning diverges from D-1 final.
# April 10 2026: D-1 final=63°F, D0 3am=66°F (+3°F jump), actual=63°F.
NWS_SEQUENCE_COLS = [
    "nws_d1_final",             # Last NWS forecast issued the day BEFORE target date
    "nws_overnight_jump",       # nws_first_d0 - nws_d1_final (positive = NWS warmed overnight)
]

# Weather Underground PWS features (4 features) — requires WU_API_KEY secret
# Hyper-local citizen weather station data near Central Park.
AMBIENT_OBS_COLS = [
    "obs_ambient_temp",         # Mean temp across nearby PWS stations (°F)
    "obs_ambient_vs_nws",       # obs_ambient_temp - nws_last
    "obs_ambient_spread",       # Max - min temp across PWS stations
    "obs_ambient_count",        # Number of stations with valid readings
]

# Synoptic Data (MesoWest) features (6 features) — requires SYNOPTIC_TOKEN secret
# Aggregates 100+ station networks within 5 miles of Central Park:
# ASOS, AWOS, NY Mesonet, maritime, campus sensors. Best multi-network consensus.
SYNOPTIC_OBS_COLS = [
    "obs_synoptic_mean",        # Mean temp across all nearby stations (°F)
    "obs_synoptic_min",         # Coldest station — catches cold advection earliest
    "obs_synoptic_max",         # Warmest station
    "obs_synoptic_spread",      # Max - min across all stations
    "obs_synoptic_vs_nws",      # obs_synoptic_mean - nws_last
    "obs_synoptic_count",       # Number of valid stations
]

# NY State Mesonet features (6 features) — NO API KEY REQUIRED (public CSV)
# SUNY Albany network. NYC borough stations: Manhattan (MANH), Brooklyn (BKLN),
# Queens (QUEE), Bronx (BRON), Staten Island (STAT).
# Captures urban heat island gradient: when BKLN = 59°F and MANH = 64°F while
# NWS says 66°F, that's a strong signal the forecast is too warm.
NYSM_OBS_COLS = [
    "obs_nysm_mean",            # Mean temp across NYC borough NYSM stations (°F)
    "obs_nysm_min",             # Coldest borough
    "obs_nysm_max",             # Warmest borough
    "obs_nysm_spread",          # Borough temp gradient
    "obs_nysm_vs_nws",          # obs_nysm_mean - nws_last
    "obs_nysm_count",           # Number of valid borough stations
]

# High-timing features (3 features)
# Captures WHEN the daily max occurs — critical for non-textbook heating profiles:
#   - Overnight/pre-dawn highs: warm fronts peak at 1–3am, then cold air floods in
#   - Late afternoon/evening highs: summer sea-breeze collapse, convective clearing
#   - Normal solar max: 1–3pm — already well-modelled; these features add the edge cases
# All three are NaN-safe — HistGradientBoosting handles missing values natively.
HIGH_TIMING_COLS = [
    "obs_high_peak_hour",      # Hour (0-23) when obs_max_so_far was recorded today
                               # NaN = no obs yet. Tells model "is the day's max at 1am or 3pm?"
    "obs_is_overnight_high",   # 1 if obs_high_peak_hour < 9, else 0 (NaN if no obs)
                               # Strong prior: if high happened before 9am, bucket is likely lower
    "obs_temp_falling_hrs",    # Consecutive hours the temp has been falling from the peak
                               # 0 = still heating/flat; 3+ = clearly post-peak
]

# Features used as INPUT to the atmospheric predictor (first-stage model)
ATM_PREDICTOR_INPUT_COLS = ATMOSPHERIC_COLS + INTRADAY_CURVE_COLS + [
    "day_of_year_sin", "day_of_year_cos", "month", "is_summer", "is_winter",
    "midnight_temp",
]

# v2 base feature list
FEATURE_COLS_V2 = (FEATURE_COLS + ATMOSPHERIC_COLS + ENSEMBLE_COLS + MULTIMODEL_COLS +
                   INTRADAY_CURVE_COLS + OVERNIGHT_CARRYOVER_COLS + ATM_PREDICTOR_COLS +
                   MOS_COLS + FORECAST_REVISION_COLS)

# v3 = v2 (same features, single unified regression model)
FEATURE_COLS_V3 = FEATURE_COLS_V2

# v4 — full feature set including all real-time observation sources
# Total: v3(84) + obs(12) + regional(5) + seq(2) + pws(4) + synoptic(6) + nysm(6) = 119
# HistGradientBoosting handles NaN natively — all live-only cols degrade gracefully
# for historical training rows.
FEATURE_COLS_V4 = (FEATURE_COLS_V3 + OBSERVATION_COLS + REGIONAL_OBS_COLS +
                   NWS_SEQUENCE_COLS + AMBIENT_OBS_COLS + SYNOPTIC_OBS_COLS + NYSM_OBS_COLS)

# v5 — adds high-timing features for overnight/late-day high detection (3 new features)
# Total: v4(119) + high_timing(3) = 122
FEATURE_COLS_V5 = FEATURE_COLS_V4 + HIGH_TIMING_COLS

# Additional features added per-candidate-bucket during classification (4)
BUCKET_POSITION_COLS = [
    "bucket_center",
    "dist_from_prediction",
    "dist_from_accu",
    "dist_from_nws",
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
    Each bucket is a 1-degree range [low, low+1).
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
