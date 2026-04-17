-- ════════════════════════════════════════════════════════════════════════════
-- ATM_SNAPSHOT MONITORING QUERIES
-- Use these to watch for failed Open-Meteo fetches and catch fallback cases
-- ════════════════════════════════════════════════════════════════════════════

-- Query 1: Find rows with EMPTY or MOSTLY-NULL atm_snapshot
-- This catches cases where live_atm fetch failed and we're in fallback mode
-- Shows: timestamp, how many keys, how many are non-null
SELECT 
  timestamp,
  target_date,
  ml_bucket,
  (jsonb_object_keys(atm_snapshot))::text[] AS snapshot_keys,
  jsonb_array_length((
    SELECT jsonb_agg(v) FROM jsonb_each(atm_snapshot) 
    WHERE v != 'null'::jsonb
  )) AS valid_non_null_count,
  jsonb_array_length(jsonb_object_keys(atm_snapshot)::jsonb[]) AS total_key_count,
  atm_snapshot
FROM prediction_logs
WHERE target_date = CURRENT_DATE - INTERVAL '1 day'
  AND (
    -- Snapshot is empty
    jsonb_array_length(jsonb_object_keys(atm_snapshot)::jsonb[]) = 0
    -- OR mostly null (< 5 valid keys out of expected 80+)
    OR jsonb_array_length((
      SELECT jsonb_agg(v) FROM jsonb_each(atm_snapshot) 
      WHERE v != 'null'::jsonb
    )) < 5
  )
ORDER BY timestamp DESC;

-- Query 2: Snapshot quality histogram — how many valid keys per day?
-- Shows the distribution: if most rows have 72+ valid keys, fetch is working
-- If many rows have <10 valid keys, fetch is failing
SELECT 
  target_date,
  CASE 
    WHEN valid_count = 0 THEN '0 (empty/failed)'
    WHEN valid_count < 10 THEN '1-9 (mostly null)'
    WHEN valid_count < 40 THEN '10-39 (partial)'
    WHEN valid_count < 70 THEN '40-69 (incomplete)'
    ELSE '70+ (full)'
  END AS snapshot_quality,
  COUNT(*) AS row_count,
  ROUND(100.0 * COUNT(*) / (
    SELECT COUNT(*) FROM prediction_logs WHERE target_date = prediction_logs.target_date
  ), 1) AS percentage
FROM (
  SELECT 
    target_date,
    jsonb_array_length((
      SELECT jsonb_agg(v) FROM jsonb_each(atm_snapshot) 
      WHERE v != 'null'::jsonb
    )) AS valid_count
  FROM prediction_logs
  WHERE target_date >= CURRENT_DATE - INTERVAL '7 days'
    AND atm_snapshot IS NOT NULL
)
GROUP BY target_date, snapshot_quality
ORDER BY target_date DESC, snapshot_quality;

-- Query 3: Real-time monitor — last 10 canonical writes and their snapshot health
-- The key columns are: timestamp (when written), atm_count (atmospheric keys),
-- obs_snap_count (observation keys), valid_count (non-null values)
SELECT 
  timestamp AT TIME ZONE 'America/New_York' AS timestamp_ny,
  target_date,
  is_canonical,
  ml_bucket,
  -- Count atm_* / mm_* / ens_* keys
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^(atm_|mm_|ens_)') AS atm_count,
  -- Count obs_snap_* keys
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^obs_snap_') AS obs_snap_count,
  -- Count non-null values
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) AS valid_count,
  -- Total keys
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot)) AS total_count
FROM prediction_logs
WHERE target_date >= CURRENT_DATE - INTERVAL '1 day'
  AND ml_bucket_canonical IS NOT NULL  -- Only canonical writes
ORDER BY timestamp DESC
LIMIT 10;

-- Query 4: Detect pattern — when did snapshot quality degrade?
-- Shows if there was a specific time or date when Open-Meteo started failing
SELECT 
  DATE_TRUNC('hour', timestamp) AS hour,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN valid_count > 0 THEN 1 END) AS rows_with_data,
  ROUND(
    100.0 * COUNT(CASE WHEN valid_count > 0 THEN 1 END) / COUNT(*),
    1
  ) AS success_rate,
  ROUND(AVG(valid_count), 1) AS avg_valid_count,
  MIN(valid_count) AS min_valid_count,
  MAX(valid_count) AS max_valid_count
FROM (
  SELECT 
    timestamp,
    jsonb_array_length((
      SELECT jsonb_agg(v) FROM jsonb_each(atm_snapshot) 
      WHERE v != 'null'::jsonb
    )) AS valid_count
  FROM prediction_logs
  WHERE target_date >= CURRENT_DATE - INTERVAL '3 days'
    AND atm_snapshot IS NOT NULL
)
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Query 5: Deep dive — specific row analysis (for debugging)
-- Use this to inspect a specific prediction row and see exactly what's in the snapshot
-- Replace the timestamp with one from your logs
SELECT 
  timestamp,
  target_date,
  is_canonical,
  ml_f,
  ml_bucket,
  -- Full snapshot as readable JSON
  jsonb_pretty(atm_snapshot) AS snapshot_formatted,
  -- Count by key prefix
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^atm_') AS atm_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^mm_') AS mm_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^ens_') AS ens_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^nws_') AS nws_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE key ~ '^obs_snap_') AS obs_snap_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value = 'null'::jsonb) AS null_count,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) AS non_null_count
FROM prediction_logs
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '6 hours'
  AND is_canonical = true
ORDER BY timestamp DESC
LIMIT 1;

-- Query 6: Alert query — rows with ZERO valid values (something went very wrong)
-- Run this regularly to catch complete failures
SELECT 
  timestamp AT TIME ZONE 'America/New_York' AS timestamp_ny,
  target_date,
  ml_bucket,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot)) AS total_keys,
  CASE 
    WHEN atm_snapshot = '{}'::jsonb THEN 'EMPTY SNAPSHOT'
    WHEN atm_snapshot IS NULL THEN 'NULL SNAPSHOT'
    ELSE 'POPULATED BUT ALL NULL'
  END AS failure_type
FROM prediction_logs
WHERE target_date >= CURRENT_DATE - INTERVAL '7 days'
  AND (
    jsonb_array_length((
      SELECT jsonb_agg(v) FROM jsonb_each(atm_snapshot) 
      WHERE v != 'null'::jsonb
    )) = 0
    OR atm_snapshot = '{}'::jsonb
    OR atm_snapshot IS NULL
  )
ORDER BY timestamp DESC;

