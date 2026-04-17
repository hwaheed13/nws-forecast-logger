-- ════════════════════════════════════════════════════════════════════════════
-- ATM_SNAPSHOT MONITORING - FIXED QUERIES
-- ════════════════════════════════════════════════════════════════════════════

-- Query 1: Snapshots with mostly-NULL content (fallback cases)
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  target_date,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot)) as total_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) as valid_keys,
  atm_snapshot IS NOT NULL as has_snapshot
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
  AND atm_snapshot IS NOT NULL
ORDER BY timestamp DESC
LIMIT 20;


-- Query 2: Daily snapshot quality (last 7 days)
SELECT 
  target_date,
  COUNT(*) as total_rows,
  ROUND(
    (SELECT AVG(valid_count) FROM (
      SELECT (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) as valid_count
      FROM prediction_logs sub
      WHERE sub.target_date = prediction_logs.target_date AND sub.city = 'nyc'
    ) temp),
    1
  ) as avg_valid_keys_per_row,
  ROUND(100.0 * COUNT(CASE WHEN atm_snapshot IS NOT NULL THEN 1 END) / COUNT(*), 1) as snapshot_coverage_pct
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY target_date
ORDER BY target_date DESC;


-- Query 3: Latest canonical write and snapshot health
SELECT 
  'LATEST CANONICAL' as analysis,
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  target_date,
  ml_f,
  ml_bucket,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot)) as total_snapshot_keys,
  (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) as valid_snapshot_keys
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date = CURRENT_DATE
  AND is_canonical = true
  AND ml_f IS NOT NULL
ORDER BY timestamp DESC
LIMIT 1;


-- Query 4: Snapshot quality timeline (hourly)
SELECT 
  DATE_TRUNC('hour', timestamp) AT TIME ZONE 'America/New_York' as hour,
  COUNT(*) as updates,
  ROUND(
    AVG(
      COALESCE(
        (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb),
        0
      )
    ),
    1
  ) as avg_valid_keys,
  COUNT(CASE WHEN atm_snapshot = '{}'::jsonb THEN 1 END) as empty_snapshots
FROM prediction_logs
WHERE target_date = CURRENT_DATE
  AND city = 'nyc'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;


-- Query 5: Alert — rows with ZERO valid values
SELECT 
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  target_date,
  ml_f,
  atm_snapshot IS NOT NULL as has_snapshot,
  atm_snapshot = '{}'::jsonb as is_empty,
  'ALERT: Zero valid keys' as alert
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date >= CURRENT_DATE - INTERVAL '1 day'
  AND atm_snapshot IS NOT NULL
  AND (SELECT COUNT(*) FROM jsonb_each(atm_snapshot) WHERE value != 'null'::jsonb) = 0
ORDER BY timestamp DESC;

