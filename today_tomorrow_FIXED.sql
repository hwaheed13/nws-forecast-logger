-- ════════════════════════════════════════════════════════════════════════════
-- TODAY vs TOMORROW: FIXED QUERIES (simpler, no syntax errors)
-- ════════════════════════════════════════════════════════════════════════════

-- 1. TODAY: What rows exist and what got written?
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  is_canonical,
  ml_f,
  ml_bucket,
  ml_confidence,
  nws_d0,
  accuweather,
  CASE 
    WHEN ml_f IS NOT NULL THEN 'HAS PREDICTION'
    ELSE 'NULL - NO PREDICTION'
  END as prediction_status,
  jsonb_object_keys(atm_snapshot) IS NOT NULL as has_snapshot
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
ORDER BY timestamp DESC;


-- 2. TOMORROW: Check if it exists
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  target_date,
  ml_f,
  ml_bucket,
  ml_confidence,
  nws_d0 as nws_d1_forecast,
  accuweather as accu_d1_forecast
FROM prediction_logs
WHERE target_date = CURRENT_DATE + INTERVAL '1 day'
  AND city = 'nyc'
ORDER BY timestamp DESC
LIMIT 5;


-- 3. ACTUAL HIGH: Extract from snapshot (last update of the day)
SELECT 
  target_date,
  'ACTUAL HIGH' as metric,
  (atm_snapshot ->> 'obs_snap_max_so_far')::numeric as actual_high,
  (atm_snapshot ->> 'obs_snap_temp')::numeric as current_temp,
  (atm_snapshot ->> 'obs_snap_hour')::numeric as peak_hour,
  (atm_snapshot ->> 'obs_snap_heating_rate')::numeric as heating_rate_f_per_hr,
  timestamp AT TIME ZONE 'America/New_York' as last_update
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
  AND atm_snapshot IS NOT NULL
ORDER BY timestamp DESC
LIMIT 1;


-- 4. DIAGNOSTIC: Count rows by prediction status
SELECT 
  target_date,
  COUNT(*) as total_rows,
  COUNT(CASE WHEN ml_f IS NOT NULL THEN 1 END) as rows_with_prediction,
  COUNT(CASE WHEN ml_f IS NULL THEN 1 END) as rows_with_null_prediction,
  ROUND(100.0 * COUNT(CASE WHEN ml_f IS NOT NULL THEN 1 END) / COUNT(*), 1) as prediction_write_rate
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date >= CURRENT_DATE - INTERVAL '3 days'
GROUP BY target_date
ORDER BY target_date DESC;


-- 5. MODEL BEHAVIOR: Where did predictions get written?
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  ml_f,
  ml_bucket,
  ml_confidence,
  is_canonical,
  'PREDICTION WRITTEN' as status
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
  AND ml_f IS NOT NULL
ORDER BY timestamp DESC;


-- 6. ATMOSPHERIC DATA: Is snapshot being written?
SELECT 
  target_date,
  COUNT(*) as total_rows,
  COUNT(CASE WHEN atm_snapshot IS NOT NULL THEN 1 END) as rows_with_snapshot,
  COUNT(CASE WHEN atm_snapshot = '{}'::jsonb THEN 1 END) as empty_snapshots,
  ROUND(100.0 * COUNT(CASE WHEN atm_snapshot IS NOT NULL THEN 1 END) / COUNT(*), 1) as snapshot_write_rate
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date = CURRENT_DATE
GROUP BY target_date;

