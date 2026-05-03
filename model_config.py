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

# Multi-model cross-comparison features (17 features)
# Models ranked by 90-day accuracy (wethr.net): HRRR(#1), NBM(#2-3), GEM HRDPS(#4-5),
# then ECMWF, GFS, ICON, GEM global. NWS point forecast lags GFS by hours.
# All models fetched via Open-Meteo (free, no API key).
MULTIMODEL_COLS = [
    "mm_spread",              # Max model - min model daily high (all 7 models)
    "mm_std",                 # Std dev across models
    "mm_mean",                # Multi-model consensus mean
    "mm_ecmwf_gfs_diff",      # ECMWF - GFS difference
    "mm_hrrr_max",            # HRRR predicted daily max (#1 accuracy, runs hourly)
    "mm_hrrr_ecmwf_diff",     # HRRR - ECMWF: positive = HRRR overmixing (divergence signal)
    "mm_hrrr_gfs_diff",       # HRRR - GFS: mesoscale vs synoptic agreement
    "mm_icon_max",            # ICON (German DWD) predicted daily max
    "mm_gem_max",             # GEM global (Canadian CMC) predicted daily max
    "mm_icon_gfs_diff",       # ICON - GFS: European vs American disagreement
    "mm_gem_ecmwf_diff",      # GEM - ECMWF: Canadian vs European disagreement
    # ── NEW in v6: top-accuracy models per wethr.net rankings ──────────────
    "mm_nbm_max",             # NBM (National Blend of Models) — blends 50+ models,
                              # faster updates than NWS point forecast, top-3 accuracy
    "mm_nbm_hrrr_diff",       # NBM - HRRR: #3 vs #1 — large disagreement = high uncertainty
    "mm_nbm_gfs_diff",        # NBM - GFS: blend vs raw synoptic baseline
    "mm_nbm_ecmwf_diff",      # NBM - ECMWF: USA blend vs European deterministic
    "mm_gem_hrdps_max",       # GEM HRDPS (Canadian high-res 2.5km) — top-5 accuracy,
                              # superior boundary layer physics for mesoscale events
    "mm_gem_hrdps_hrrr_diff", # GEM HRDPS - HRRR: two high-res models disagreeing
                              # is a strong signal of boundary layer uncertainty
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
    "obs_heating_rate",         # °F/hr over last 3 hours (overall slope)
    "obs_heating_rate_delta",   # Δ°F/hr: recent_half_slope - early_half_slope (stall signal)
                                # Negative = decelerating = cap holding. April 12: ~-1.6°F/hr.
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

# ── NEW in v6: HRRR-specific pressure level features (4 features) ──────────
# The standard atm_925mb_temp_* features use GFS-derived 925mb from Open-Meteo.
# GFS at 13km resolution can miss boundary layer caps that HRRR resolves at 3km.
# Negative atm_925mb_gfs_hrrr_diff = HRRR sees cooler air aloft than GFS (cap stronger).
HRRR_PRESSURE_COLS = [
    "atm_925mb_hrrr_max",       # HRRR 925mb max temp (daytime 10am-6pm) — 3km resolution
    "atm_925mb_hrrr_mean",      # HRRR 925mb mean temp (daytime 10am-6pm)
    "atm_850mb_hrrr_max",       # HRRR 850mb max temp — warm advection aloft detection
    "atm_850mb_hrrr_mean",      # HRRR 850mb mean temp
    "atm_925mb_gfs_hrrr_diff",  # GFS minus HRRR 925mb — when large & positive, GFS is missing
                                # the cap. Critical for cases like April 12 2026 where GFS
                                # showed warm 925mb but cap suppressed actual high below 55°F.
]

# ── NEW in v6: Radiosonde (upper-air balloon) observed soundings (5 features) ──
# Source: Iowa State Mesonet → OKX (Upton, NY) 12Z and 00Z launches
# These are ACTUAL OBSERVED upper-air temperatures, not model output.
# The difference between forecasted and observed 925mb is the cap miss signal.
RADIOSONDE_COLS = [
    "raob_925mb_temp",          # Observed 925mb temperature (°F) from OKX morning sounding
    "raob_850mb_temp",          # Observed 850mb temperature (°F) — synoptic-scale warm advection
    "raob_700mb_temp",          # Observed 700mb temperature (°F) — mid-level cap signal
    "raob_925mb_gfs_diff",      # GFS forecast minus observed 925mb (positive = GFS too warm)
    "raob_925mb_hrrr_diff",     # HRRR forecast minus observed 925mb (positive = HRRR too warm)
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

# v6 — adds high-accuracy models (NBM, GEM HRDPS), HRRR-specific 925mb,
#       GFS-HRRR 925mb diff, and OKX radiosonde upper-air observed soundings.
#
# Motivation (April 12, 2026 case):
#   - Model flipped from <=55 to 56-57 based on stale GFS 925mb (showed warm cap)
#   - HRRR 925mb and actual OKX radiosonde showed cold cap (~48°F)
#   - NBM and GEM HRDPS would have provided faster, higher-accuracy model consensus
#   - These features give the model direct signals to resist NWS/GFS cap misses
#
# Total: v5(122) + multimodel_new(6) + hrrr_pressure(5) + radiosonde(5) = 138
FEATURE_COLS_V6 = (
    FEATURE_COLS_V5
    + ["mm_nbm_max", "mm_nbm_hrrr_diff", "mm_nbm_gfs_diff", "mm_nbm_ecmwf_diff",
       "mm_gem_hrdps_max", "mm_gem_hrdps_hrrr_diff"]
    + HRRR_PRESSURE_COLS
    + RADIOSONDE_COLS
)

# ═══════════════════════════════════════════════════════════════════════
# v7 feature columns — same as v6 but DIFFERENT training base
# ═══════════════════════════════════════════════════════════════════════
#
# ARCHITECTURAL CHANGE (v7):
#   v1-v6: y_bias = actual - AccuWeather (or NWS fallback)
#   v7:    y_bias = actual - HRRR_max (or NBM > AccuWeather > NWS cascade)
#
#   Why this matters:
#     HRRR is ranked #1 accuracy model (wethr.net). AccuWeather/NWS rank #11-16.
#     When y_bias is computed relative to HRRR, the regressor learns:
#       "HRRR is typically X°F off in these conditions" rather than
#       "AccuWeather is typically Y°F off in these conditions."
#     On cap days: HRRR shows 54°F, AccuWeather shows 58°F. v6 anchors at 58
#     and tries to apply a -3 correction. v7 anchors at 54 from the start.
#
#   Inference priority (v7 active):
#     1. HRRR_max + v7_regressor_bias  (when HRRR available — best model)
#     2. NBM_max + v7_regressor_bias   (when HRRR unavailable, NBM available)
#     3. atm_predicted_high            (physics model fallback)
#     4. AccuWeather/NWS + v7_bias     (last resort for historical compatibility)
#
#   Historical rows (no HRRR/NBM): base falls through to AccuWeather/NWS.
#   HistGradientBoostingRegressor handles mixed-base NaN training natively.
#
# Total: same 138 features as v6 — architectural change is in the base only.
FEATURE_COLS_V7 = FEATURE_COLS_V6  # identical feature set; base changes at train/inference

# ═══════════════════════════════════════════════════════════════════════
# v8 feature columns — v7 + obs_heating_rate_delta (stall signal)
# ═══════════════════════════════════════════════════════════════════════
#
# THE STALL SIGNAL:
#   obs_heating_rate captures overall slope (°F/hr over last ~3 hrs).
#   obs_heating_rate_delta = recent_slope - early_slope.
#
#   Negative delta = warming rate is DECELERATING = cap is holding.
#   A human watching the station graph spots this instantly.
#   This gives the model the same information without needing Synoptic.
#
#   April 12 example:
#     Early (7-9am): +1.8°F/hr → recent (10am-noon): +0.2°F/hr
#     delta = -1.6°F/hr  ← automated cap fingerprint
#
#   Combined with HRRR-anchored base (v7) + HRRR vs NWS gap + radiosonde,
#   this is the set of features that would have held the ≤55 bucket on Apr 12.
#
# Also adds 1 new feature to OBSERVATION_COLS (obs_heating_rate_delta).
# Total: v7(138) + 1 = 139 features
FEATURE_COLS_V8 = list(FEATURE_COLS_V7) + ["obs_heating_rate_delta"]

# ═══════════════════════════════════════════════════════════════════════
# v9 feature columns — v8 + named ASOS station features (marine cap signal)
# ═══════════════════════════════════════════════════════════════════════
#
# Individual station readings via Synoptic timeseries API.
# The insight: aggregates (obs_synoptic_mean) lose the directional signal
# that matters most for marine cap detection.
#
# KJFK (coastal Queens/Jamaica Bay) is the first station to feel the sea
# breeze. When KJFK is colder than KNYC (Central Park, inland), the cap
# is coming from the ocean. The stronger the KJFK-KNYC gap, the stronger
# the marine influence and the lower the actual high.
#
# Historical access: Synoptic enterprise tier gives 1-year history via the
# timeseries API. This means we can BACKFILL these features for all
# historical training rows — the model learns from actual cap-day
# fingerprints immediately, not just from future live cycles.
#
# On April 12, 2026 (cap day):
#   KJFK ≈ 50°F, KNYC ≈ 52°F → obs_kjfk_vs_knyc = -2°F
#   KEWR ≈ 54°F → obs_kewr_vs_knyc = +2°F (NJ warmer, not capped)
#   obs_coastal_vs_inland ≈ -4°F (strong marine signal)
#
# Total: v8(139) + named_stations(10) = 149 features
SYNOPTIC_NAMED_STATION_COLS = [
    # Individual ASOS station temperatures (°F)
    "obs_kjfk_temp",         # JFK Airport — coastal Queens/Jamaica Bay
    "obs_klga_temp",         # LaGuardia — north Queens/East River
    "obs_kewr_temp",         # Newark — inland NJ, warmer on marine cap days
    "obs_kteb_temp",         # Teterboro — most inland, warmest on cap days
    "obs_knyc_temp",         # Central Park via Synoptic (cross-check vs NWS KNYC)
    # Cross-station diffs anchored at KNYC — the marine cap signal
    "obs_kjfk_vs_knyc",      # KJFK - KNYC: negative = sea breeze penetrating inland
    "obs_klga_vs_knyc",      # KLGA - KNYC: intermediate signal
    "obs_kewr_vs_knyc",      # KEWR - KNYC: positive = NJ warmer = no marine cap
    # Network-wide marine signal
    "obs_airport_spread",    # max(airports) - min(airports): near-zero = uniform cap
    "obs_coastal_vs_inland", # mean(KJFK,KLGA) - mean(KEWR,KTEB): negative = marine
]

FEATURE_COLS_V9 = list(FEATURE_COLS_V8) + SYNOPTIC_NAMED_STATION_COLS

# ═══════════════════════════════════════════════════════════════════════
# v10 feature columns — v9 + Manhattan Mesonet (MANH) 5-min fill-in
# ═══════════════════════════════════════════════════════════════════════
#
# MANH is the NY State Mesonet station near Columbia University (~125th St),
# ~1.5 miles north of Central Park. Updates every 5 minutes — the only
# sub-hourly station in the Synoptic radius pull near KNYC.
#
# Why it matters: KNYC (ASOS) reports once per hour at :51. Between :51 and
# the next report, a sea breeze intrusion or temperature inversion could
# already be underway with no KNYC signal. MANH closes that 59-min blind
# spot. When MANH is colder than the last KNYC reading, the cap is active
# and tightening — even if KNYC hasn't caught up yet.
#
# obs_manh_vs_knyc: negative = MANH colder than last KNYC = cap deepening
#
# Total: v9(149) + manh(2) = 151 features
MANHATTAN_MESONET_COLS = [
    "obs_manh_temp",        # MANH (Columbia/125th St) — 5-min updates
    "obs_manh_vs_knyc",     # MANH - KNYC: negative = sea breeze reaching park early
]

# ═══════════════════════════════════════════════════════════════════════
# LAX city variant feature columns
# ═══════════════════════════════════════════════════════════════════════
#
# SYNOPTIC_LAX_STATION_COLS: LAX marine layer detection via 5 regional stations
# (replaces NYC's SYNOPTIC_NAMED_STATION_COLS for city="lax")
#
# Marine layer geometry in LAX is OPPOSITE from NYC sea breeze:
#   - Coastal (KLAX/KSMO) are COLDEST when marine layer is active
#   - Inland (KBUR/KVNY) are WARMEST when layer pins coast (heating uncapped)
#
# On a marine layer day:
#   KLAX ≈ 65°F, KSMO ≈ 66°F, KBUR ≈ 78°F, KVNY ≈ 76°F
#   obs_bur_vs_lax = +13°F (strong inland heating signal = layer active)
#
# On a clear day:
#   KLAX ≈ 78°F, KSMO ≈ 79°F, KBUR ≈ 82°F, KVNY ≈ 80°F
#   obs_bur_vs_lax = +4°F (small diff = no layer, all heating together)
#
# Total: v8(139) + lax_named_stations(10) = 149 features (same count as NYC for compatibility)
SYNOPTIC_LAX_STATION_COLS = [
    # Individual LAX regional airport temperatures (°F)
    "obs_lax_temp",               # KLAX — coastal airport, reference point
    "obs_smo_temp",               # KSMO — Santa Monica, coastal
    "obs_bur_temp",               # KBUR — Burbank, inland San Fernando Valley
    "obs_vny_temp",               # KVNY — Van Nuys, inland
    "obs_cqt_temp",               # KCQT — USC Campus Downtown LA, official NWS/Kalshi reference
    # Cross-station diffs — marine layer signal
    "obs_bur_vs_lax",             # KBUR - KLAX: positive = inland warmer = marine layer active
    "obs_coastal_vs_inland_lax",  # mean(KLAX,KSMO) - mean(KBUR,KVNY): negative = marine layer
    # Network-wide marine signal
    "obs_airport_spread_lax",     # max - min across all 4 LAX regional stations
]

# LAX v9 equivalent: NYC's v9 features translated for LAX
# Uses SYNOPTIC_LAX_STATION_COLS instead of SYNOPTIC_NAMED_STATION_COLS
FEATURE_COLS_V9_LAX = list(FEATURE_COLS_V8) + SYNOPTIC_LAX_STATION_COLS

# LAX v10 equivalent: v9_lax + observation columns (no Manhattan Mesonet equivalent yet)
# Manhattan Mesonet is NYC-specific — no comparable high-frequency network in LAX yet.
# Skip MANHATTAN_MESONET_COLS for LAX.
FEATURE_COLS_V10_LAX = list(FEATURE_COLS_V9_LAX)

FEATURE_COLS_V10 = list(FEATURE_COLS_V9) + MANHATTAN_MESONET_COLS

# ═══════════════════════════════════════════════════════════════════════
# v11 feature columns — v10 + model-vs-NWS divergence features
# ═══════════════════════════════════════════════════════════════════════
#
# The core insight: NWS point forecast is ranked #11-16 by 90-day accuracy
# (wethr.net), yet the existing features treat it as the reference benchmark
# (obs_synoptic_vs_nws, rolling_bias_7d, etc.). The model must INFER the
# HRRR-vs-NWS gap from two separate features (mm_hrrr_max, nws_last).
#
# Explicitly computing the divergence gives the model a clean, direct signal:
#   "Right now HRRR is calling +5.6°F above NWS. How should I adjust?"
#
# Three complementary divergence signals:
#   mm_hrrr_vs_nws   — HRRR (#1 accuracy) vs slow NWS. The primary upside/downside flag.
#   mm_nbm_vs_nws    — NBM (#3, blends 50+ models, updates hourly) vs NWS.
#                      NBM agreement with HRRR = high conviction. Disagreement = caution.
#   mm_mean_vs_nws   — Full 7-model consensus vs NWS. When ALL fast models are above
#                      NWS, that's a unanimous signal the official forecast is stale.
#
# All three are derived at train time (features_df has both mm_* and nws_last) and at
# inference time (v2_features assembled before model.predict() is called).
#
# Total: v10(151) + divergence(3) = 154 features
MM_VS_NWS_COLS = [
    "mm_hrrr_vs_nws",     # HRRR - nws_last: fast mesoscale model vs slow official forecast
    "mm_nbm_vs_nws",      # NBM - nws_last: 50-model blend vs slow official forecast
    "mm_mean_vs_nws",     # 7-model consensus mean - nws_last: unanimous fast-model signal
]

# Model-vs-model divergence — anchored on 4+ years of historical multi-model forecasts
# (NWS forecast log only goes back ~10 months, capping mm_*_vs_nws coverage at ~144 rows).
# These swap the NWS anchor for HRRR (already 4yr in multiyear CSV); after backfill of
# GFS/ECMWF/NBM via open-meteo historical-forecast-api, all three diffs have 4+ years
# of populated history.
#   mm_hrrr_vs_gfs    — high-res mesoscale vs synoptic global (real signal)
#   mm_hrrr_vs_ecmwf  — HRRR vs the #1 global deterministic model
#   mm_hrrr_vs_nbm    — HRRR vs the 50-model blend (high conviction when small)
MM_VS_MM_COLS = [
    "mm_hrrr_vs_gfs",
    "mm_hrrr_vs_ecmwf",
    "mm_hrrr_vs_nbm",
]

FEATURE_COLS_V11 = list(FEATURE_COLS_V10) + MM_VS_NWS_COLS + MM_VS_MM_COLS

# v11 LAX equivalent — same divergence features apply
FEATURE_COLS_V11_LAX = list(FEATURE_COLS_V10_LAX) + MM_VS_NWS_COLS + MM_VS_MM_COLS

# ═══════════════════════════════════════════════════════════════════════
# v12 feature columns — v11 + deep NNJ inland stations (KCDW + KSMQ)
# ═══════════════════════════════════════════════════════════════════════
#
# v9 captured EWR (~15mi) and TEB (~20mi) as the inland reference.
# obs_coastal_vs_inland = mean(JFK,LGA) - mean(EWR,TEB) — only 20mi spread.
#
# KCDW (Caldwell NJ, ~25mi) and KSMQ (Somerville NJ, ~35mi) push the
# inland reference much deeper into the NJ interior, where the sea breeze
# NEVER reaches on any day.  When JFK is 68°F and KSMQ is 84°F, the
# gradient (+16°F over 35mi) is an unambiguous marine cap fingerprint.
# That gradient is invisible to v11, which only sees the 20mi EWR/TEB signal.
#
# obs_inland_gradient = KSMQ - KJFK:
#   Positive and large (≥+5°F): warm air mass inland, sea breeze not penetrating far
#   Positive and huge (≥+10°F): strong cap day — JFK pinned, inland heating
#   Near zero:                  uniform conditions, no differential heating
#
# These features are populated by:
#   - Live: prediction_writer.py obs pipeline via synoptic_client.py
#   - Historical backfill: backfill_synoptic.py (KCDW/KSMQ added to BACKFILL_STIDS)
#
# Total: v11(154) + deep_inland(3) = 157 features
DEEP_INLAND_STATION_COLS = [
    "obs_kcdw_temp",        # Caldwell NJ ASOS (~25mi inland) — warmer than TEB on cap days
    "obs_ksmq_temp",        # Somerville NJ ASOS (~35mi inland) — deepest inland reference
    "obs_inland_gradient",  # KSMQ - KJFK: positive = warm inland air mass, negative = uniform cap
]

FEATURE_COLS_V12 = list(FEATURE_COLS_V11) + DEEP_INLAND_STATION_COLS

# ═══════════════════════════════════════════════════════════════════════
# v13 feature columns — v12 + BL safeguard features (entrainment, marine, inland)
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION (April 15, 2026):
#   v12 BL height spike triggered downward revision even though:
#   - Entrainment gradient was established, not actively growing
#   - Marine layer was contained at coast (not penetrating inland)
#   - Inland stations showed forecast was tracking well
#
# These 3 features encode those guard rails directly, letting model learn:
#   "BL trigger is conditional on atmospheric state"
#   "Don't reduce forecast if entrainment, marine, inland signals don't align"
#
# Feature definitions:
#   entrainment_temp_diff = atm_925mb_temp_mean - obs_latest_temp
#     → Negative = cool aloft (potential mixing), positive/near-zero = neutral
#   marine_containment = obs_kjfk_vs_knyc / atm_bl_height_max
#     → Ratio of coastal gradient to BL depth; large negative + deep BL = contained
#   inland_strength = mean(obs_kteb_temp, obs_kcdw_temp, obs_ksmq_temp) - mm_mean
#     → Inland actual vs forecast consensus; positive = inland beating forecast
#
# Total: v12(168) + BL_safeguard(3) = 171 features
BL_SAFEGUARD_COLS = [
    "entrainment_temp_diff",    # 925mb - obs: is aloft air actively cooling surface?
    "marine_containment",        # JFK_vs_KNYC / BL_max: is ocean penetrating inland?
    "inland_strength",           # inland_mean - forecast_mean: who's winning?
]

FEATURE_COLS_V13 = list(FEATURE_COLS_V12) + BL_SAFEGUARD_COLS

# ═══════════════════════════════════════════════════════════════════════
# v14 feature columns — v13 + intraday "blind-spot" interaction features
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION (April 28, 2026):
#   v13 predicted "67° or more · STRONG BET" at 3:36 PM with:
#     obs_max_so_far  = 65.0°F   (peak observed today)
#     obs_latest_temp = 62.1°F   (already declining)
#     mm_hrrr_max     = 68°F     (HRRR's prediction)
#     obs_latest_hour = 15
#
#   v13 has all of these as raw features. But it has NEVER seen an explicit
#   "late afternoon AND obs is well below model prediction" interaction term —
#   so even though the raw signals are there, the gradient boosting tree splits
#   would need to learn the conjunction from sparse training data (281 rows,
#   most morning-run snapshots). With <30 late-day-divergence cases in history,
#   the model can't reliably learn the conjunction from raw features alone.
#
#   v14 makes the conjunction explicit so a single tree split captures it.
#
# Feature definitions:
#   hours_to_heating_close = max(0, 16 - obs_latest_hour)
#     → 0 means heating window over; high means lots of time left to climb
#   peak_to_hrrr_gap = mm_hrrr_max - obs_max_so_far
#     → How much room left between observed peak and HRRR's prediction
#   late_obs_below_pred = (obs_latest_hour >= 13) * max(0, peak_to_hrrr_gap)
#     → THE key signal: "in late afternoon, model is X°F above what we've hit"
#       Apr 28 case: hour=15, gap=68-65=3 → late_obs_below_pred=3.0
#   late_falling_signal = (obs_latest_hour >= 13) * obs_temp_falling_hrs
#     → "we're past peak AND it's late in the day" combined into one feature
#   mm_spread_late = mm_spread * (obs_latest_hour >= 12)
#     → High model uncertainty AFTER noon (early-morning spread is normal noise)
#
# All derived from existing columns — no backfill needed beyond recomputing
# at training time. HistGradientBoosting handles NaN natively for rows where
# any source column is missing.
#
# Total: v13(171) + 5 = 176 features
BLIND_SPOT_COLS = [
    "hours_to_heating_close",   # max(0, 16 - obs_latest_hour)
    "peak_to_hrrr_gap",         # mm_hrrr_max - obs_max_so_far
    "late_obs_below_pred",      # (obs_latest_hour>=13) * max(0, peak_to_hrrr_gap)
    "late_falling_signal",      # (obs_latest_hour>=13) * obs_temp_falling_hrs
    "mm_spread_late",           # mm_spread * (obs_latest_hour>=12)
]

FEATURE_COLS_V14 = list(FEATURE_COLS_V13) + BLIND_SPOT_COLS

# v15 — Morning-applicable + autoregressive features (canonical-time fixes)
# ─────────────────────────────────────────────────────────────────────────
# v14's blind-spot features all gate on obs_latest_hour >= 12/13, so they're
# zero or NaN at the 7am canonical write — exactly the moment the bet
# recommendation gets locked in. v15 adds five features that fire at
# canonical time and across cross-day predictions:
#
#   forecast_revision    = nws_last - nws_first (within forecast lifecycle)
#                          Positive = NWS revised upward overnight = warming
#                          regime; negative = cooling regime.
#
#   cap_violation_925    = max(0, nws_last - atm_925mb_temp_mean - 14)
#                          Adiabatic lapse: surface should be ≤14°F warmer
#                          than 925mb in dry air. Positive value = forecast
#                          exceeds physical ceiling = systemic over-forecast.
#
#   yesterday_signed_miss = prev_day_actual - prev_day_canonical_prediction
#                          Autoregressive bias correction — yesterday's miss
#                          direction is a leading indicator for today's.
#                          Computed at write time via Supabase lookup; NaN
#                          if no canonical row for prior day.
#
#   rolling_3day_bias    = mean(signed_miss) over prior 3 days
#                          Short-horizon systematic bias. Smooths single-day
#                          noise. NaN if <2 of last 3 days have canonicals.
#
#   today_realized_error = obs_max_so_far_today - mm_hrrr_max_today
#                          For tomorrow predictions only: how is HRRR doing
#                          on today's forecast right now? Strong prior for
#                          tomorrow's same-regime error. NaN for today
#                          predictions (collapses to 0 via gating in compute).
#
# Total: v14(176) + 5 = 181 features
MORNING_AUTOREG_COLS = [
    "forecast_revision",
    "cap_violation_925",
    "yesterday_signed_miss",
    "rolling_3day_bias",
    "today_realized_error",
]

FEATURE_COLS_V15 = list(FEATURE_COLS_V14) + MORNING_AUTOREG_COLS

# ═══════════════════════════════════════════════════════════════════════
# v16 — UNIFIED MODEL: ALL features, ALL labeled rows (4yr + live)
# ═══════════════════════════════════════════════════════════════════════
#
# Design intent:
#   ONE model. Trained on the union of every labeled row we have:
#     - 4 years of multiyear atmospheric history (~1577 rows)
#     - 10 months of NWS-log forecast rows (~1258 rows)
#     - Per-hour intraday snapshots (10x per day, populates obs_*)
#   ALL features active. HistGradientBoosting handles NaN per-feature
#   per-row, so missing fields are fine — the model learns which features
#   matter under which conditions.
#
# Target:
#   actual_high — direct regression on tomorrow's high.
#   No bias-from-reference indirection. Eliminates the cascade's "v15
#   trained on a 3300-row pool" sparsity problem.
#
# Why this beats the cascade:
#   The cascade was incremental scaffolding. Each v_n trained on rows
#   where v_n's distinctive features were populated → narrow pools,
#   sparse training, NaN-falls-back-to-prior-version fragility. A
#   single model on the union pool is what predictive ML is supposed
#   to be.
#
# Intraday moat:
#   When obs roll in mid-day, v16 sees the updated obs_max_so_far,
#   obs_heating_rate, obs_heating_rate_delta etc. and re-predicts. The
#   per-hour intraday snapshots (~10x/day in training) teach v16 to
#   recognize early signals of bucket flips before they manifest.
# v16 adds cap_with_clearing — explicit interaction between entrainment
# (cold cap signal) and clearing sky (cloud_cover_max). Captures the
# May 3 2026 pattern: cold morning + clearing afternoon → afternoon sun
# fights the cap → HRRR overshoots LESS than pure stall heuristic suggests.
# All inputs are in 100% of multi-year archive rows so the 4yr corpus
# teaches it. Without this interaction the residual model overcorrected
# HRRR by 4°F (predicted 55.5°F when actual was ~59°F).
V16_INTERACTION_COLS = ["cap_with_clearing"]
FEATURE_COLS_V16 = list(FEATURE_COLS_V15) + V16_INTERACTION_COLS
FEATURE_COLS_V16_LAX = list(FEATURE_COLS_V15) + V16_INTERACTION_COLS  # same superset

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