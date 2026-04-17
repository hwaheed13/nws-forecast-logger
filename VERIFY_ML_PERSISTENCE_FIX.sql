-- Verification Query for ML Persistence Fix
-- Run this after the next write_today_for_tomorrow() cycle to verify ml_f values
-- are now persisting across intraday writes

-- 1. Verify ml_f values exist and are not NULL across all writes for today's +1 day
SELECT
  timestamp AT TIME ZONE 'America/New_York' as timestamp_ny,
  lead_used,
  ml_f,
  ml_bucket,
  ml_f_canonical,
  ml_bucket_canonical,
  is_canonical,
  CASE
    WHEN ml_f IS NOT NULL THEN '✅ ml_f WRITTEN'
    ELSE '❌ ml_f NULL'
  END as ml_f_status,
  CASE
    WHEN ml_f_canonical IS NOT NULL THEN '✅ canonical preserved'
    WHEN ml_f_canonical IS NULL THEN '❌ canonical NULL (data loss)'
  END as canonical_status
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date = (SELECT DATE(now() AT TIME ZONE 'America/New_York') + interval '1 day')
  AND lead_used IN ('today_for_tomorrow')
ORDER BY timestamp ASC;

-- 2. Comparison: revision_log vs prediction_logs (should show all ✅ WRITTEN now)
SELECT
  rev.timestamp AT TIME ZONE 'America/New_York' as computed_time,
  rev.lead_used,
  rev.trigger_reason,
  rev.ml_f as computed_ml_f,
  pred.ml_f as written_ml_f,
  pred.ml_f_canonical,
  CASE
    WHEN rev.ml_f IS NOT NULL AND pred.ml_f IS NOT NULL THEN '✅ WRITTEN'
    WHEN rev.ml_f IS NOT NULL AND pred.ml_f IS NULL THEN '❌ COMPUTED but NOT WRITTEN'
    ELSE '⚠️ OTHER'
  END as status
FROM prediction_revision_log rev
LEFT JOIN prediction_logs pred ON (
  rev.city = pred.city
  AND rev.target_date = pred.target_date
  AND rev.lead_used = pred.lead_used
)
WHERE rev.city = 'nyc'
  AND rev.target_date = (SELECT DATE(now() AT TIME ZONE 'America/New_York') + interval '1 day')
  AND rev.lead_used = 'today_for_tomorrow'
ORDER BY rev.timestamp ASC;

-- 3. Summary statistics: how many intraday writes now have ml_f?
SELECT
  COUNT(*) as total_writes,
  SUM(CASE WHEN ml_f IS NOT NULL THEN 1 ELSE 0 END) as writes_with_ml_f,
  SUM(CASE WHEN ml_f_canonical IS NOT NULL THEN 1 ELSE 0 END) as writes_with_canonical,
  ROUND(100.0 * SUM(CASE WHEN ml_f IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as percent_ml_f_written,
  ROUND(100.0 * SUM(CASE WHEN ml_f_canonical IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as percent_canonical_preserved
FROM prediction_logs
WHERE city = 'nyc'
  AND target_date = (SELECT DATE(now() AT TIME ZONE 'America/New_York') + interval '1 day')
  AND lead_used = 'today_for_tomorrow';

-- Expected Results After Fix:
-- ✅ Query 1: All ml_f values should be NOT NULL, all ml_f_canonical values should be NOT NULL
-- ✅ Query 2: All statuses should show '✅ WRITTEN', no '❌ COMPUTED but NOT WRITTEN' entries
-- ✅ Query 3: percent_ml_f_written should be 100%, percent_canonical_preserved should be 100%

-- If any of these conditions are NOT met, the fix didn't work and we need to investigate further.
