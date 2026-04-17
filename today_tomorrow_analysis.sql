-- ════════════════════════════════════════════════════════════════════════════
-- TODAY vs TOMORROW: Model Prediction & Data Flow Analysis
-- ════════════════════════════════════════════════════════════════════════════

-- 1. TODAY: What actually happened? (Final state)
SELECT 
  target_date,
  'TODAY (ACTUAL)' as period,
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  ml_f,
  ml_bucket,
  ml_confidence,
  nws_d0 as nws_forecast,
  accuweather as accu_forecast,
  ml_f_canonical,
  ml_bucket_canonical,
  entrainment_temp_diff,
  marine_containment,
  inland_strength,
  -- Extract key observation snapshots
  (atm_snapshot ->> 'obs_snap_max_so_far')::float as actual_high_so_far,
  (atm_snapshot ->> 'obs_snap_temp')::float as current_temp,
  (atm_snapshot ->> 'obs_snap_hour')::float as obs_hour,
  (atm_snapshot ->> 'obs_snap_heating_rate')::float as heating_rate,
  (atm_snapshot ->> 'atm_bl_height_max')::float as mixing_layer_max,
  (atm_snapshot ->> 'atm_cloud_cover_mean')::float as cloud_cover,
  (atm_snapshot ->> 'ens_spread')::float as ensemble_spread,
  (atm_snapshot ->> 'mm_spread')::float as mm_spread,
  (atm_snapshot ->> 'mm_hrrr_max')::float as hrrr_forecast,
  (atm_snapshot ->> 'mm_nbm_max')::float as nbm_forecast
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
ORDER BY timestamp DESC
LIMIT 1;


-- 2. TOMORROW: What is the model predicting?
SELECT 
  target_date,
  'TOMORROW (FORECAST)' as period,
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  ml_f,
  ml_bucket,
  ml_confidence,
  nws_d0 as nws_d1_forecast,
  accuweather as accu_d1_forecast,
  ml_f_canonical,
  ml_bucket_canonical,
  entrainment_temp_diff,
  marine_containment,
  inland_strength,
  -- Extract atmospheric baseline from prediction time
  (atm_snapshot ->> 'obs_snap_temp')::float as temp_at_prediction,
  (atm_snapshot ->> 'atm_bl_height_max')::float as forecast_mixing_layer,
  (atm_snapshot ->> 'atm_cloud_cover_mean')::float as forecast_cloud_cover,
  (atm_snapshot ->> 'atm_solar_radiation_peak')::float as forecast_solar_peak,
  (atm_snapshot ->> 'ens_spread')::float as ensemble_spread,
  (atm_snapshot ->> 'mm_hrrr_max')::float as hrrr_forecast,
  (atm_snapshot ->> 'mm_ecmwf_max')::float as ecmwf_forecast,
  (atm_snapshot ->> 'mm_nbm_max')::float as nbm_forecast
FROM prediction_logs
WHERE target_date = CURRENT_DATE + INTERVAL '1 day'
  AND city = 'nyc'
ORDER BY timestamp DESC
LIMIT 1;


-- 3. INTRADAY CHANGES (how much did forecasts shift during the day?)
SELECT 
  'INTRADAY REVISIONS' as metric,
  ROUND(AVG((nws_post_9am_delta)::float), 1) as avg_nws_revision,
  ROUND(AVG((accu_post_9am_delta)::float), 1) as avg_accu_revision,
  COUNT(*) as total_updates,
  MIN(timestamp AT TIME ZONE 'America/New_York') as first_update,
  MAX(timestamp AT TIME ZONE 'America/New_York') as last_update
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
  AND (nws_post_9am_delta IS NOT NULL OR accu_post_9am_delta IS NOT NULL);


-- 4. MODEL SHIFTS (did the model change its prediction during the day?)
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as update_time,
  ml_f,
  ml_bucket,
  ml_confidence,
  LAG(ml_f) OVER (ORDER BY timestamp) as prev_ml_f,
  LAG(ml_bucket) OVER (ORDER BY timestamp) as prev_ml_bucket,
  CASE 
    WHEN LAG(ml_bucket) OVER (ORDER BY timestamp) IS NOT NULL 
      AND LAG(ml_bucket) OVER (ORDER BY timestamp) != ml_bucket
    THEN '🔀 BUCKET SHIFTED'
    ELSE 'stable'
  END as bucket_status,
  ROUND((ml_f - LAG(ml_f) OVER (ORDER BY timestamp))::numeric, 1) as ml_f_change
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
ORDER BY timestamp;


-- 5. DATA FLOW: Is today feeding tomorrow?
-- Today's actual high should be captured in obs_snap_max_so_far
-- This actual high will be in training data for next retrain
SELECT 
  CURRENT_DATE as today_date,
  (SELECT (atm_snapshot ->> 'obs_snap_max_so_far')::float 
   FROM prediction_logs 
   WHERE target_date = CURRENT_DATE AND city = 'nyc'
   ORDER BY timestamp DESC LIMIT 1) as today_actual_high,
  (SELECT (atm_snapshot ->> 'obs_snap_hour')::float 
   FROM prediction_logs 
   WHERE target_date = CURRENT_DATE AND city = 'nyc'
   ORDER BY timestamp DESC LIMIT 1) as peak_hour,
  CURRENT_DATE + INTERVAL '1 day' as tomorrow_date,
  (SELECT ml_f_canonical 
   FROM prediction_logs 
   WHERE target_date = CURRENT_DATE + INTERVAL '1 day' AND city = 'nyc' AND is_canonical = true) as tomorrow_prediction,
  'Today actual → train.py (2am) → tomorrow weights → D+2 prediction' as data_flow;


-- 6. FEATURE IMPORTANCE: What drove today's prediction?
SELECT 
  'TODAY FEATURES' as analysis,
  ROUND((atm_snapshot ->> 'atm_bl_height_max')::float, 0) as mixing_layer_m,
  ROUND((atm_snapshot ->> 'atm_cloud_cover_mean')::float, 1) as cloud_pct,
  ROUND((atm_snapshot ->> 'atm_wind_max')::float, 1) as wind_mph,
  ROUND((atm_snapshot ->> 'ens_spread')::float, 1) as ens_uncert,
  ROUND((atm_snapshot ->> 'mm_hrrr_ecmwf_diff')::float, 1) as model_divergence,
  ROUND((atm_snapshot ->> 'atm_925mb_temp_mean')::float, 1) as temp_925mb,
  ROUND((atm_snapshot ->> 'atm_850mb_temp_mean')::float, 1) as temp_850mb,
  entrainment_temp_diff as entrainment_signal,
  marine_containment as marine_signal,
  inland_strength as inland_signal
FROM prediction_logs
WHERE target_date = CURRENT_DATE AND city = 'nyc'
ORDER BY timestamp DESC
LIMIT 1;

