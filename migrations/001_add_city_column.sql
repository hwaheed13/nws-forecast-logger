-- Migration: Add city column to prediction_logs + update forecast_day_summary view
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- ============================================================================

-- 1. Add city column with default 'nyc' (all existing rows become 'nyc')
ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS city TEXT NOT NULL DEFAULT 'nyc';

-- 2. Create index on city for fast filtering
CREATE INDEX IF NOT EXISTS idx_prediction_logs_city
  ON prediction_logs (city);

-- 3. Drop the old unique constraint that doesn't include city
--    (The constraint name may vary — check your DB if this fails.
--     Common names: uq_pred_logs, prediction_logs_target_date_lead_used_issuance_iso_model_name_version_key)
ALTER TABLE prediction_logs
  DROP CONSTRAINT IF EXISTS uq_pred_logs;

ALTER TABLE prediction_logs
  DROP CONSTRAINT IF EXISTS prediction_logs_target_date_lead_used_issuance_iso_model_name_version_key;

-- 4. Create new unique constraint that includes city
--    This allows NYC and LA to each have their own prediction for the same date/lead/model
ALTER TABLE prediction_logs
  ADD CONSTRAINT uq_pred_logs
  UNIQUE (target_date, lead_used, issuance_iso, model_name, version, city);

-- 5. Also ensure idempotency_key remains unique (it already has city in the value
--    for prediction_writer.py, but log-prediction.js needs updating too)
--    The existing unique index on idempotency_key should be fine as-is.

-- 6. Recreate the forecast_day_summary view WITH city column
--    Drop the old view first, then recreate with city grouping
DROP VIEW IF EXISTS forecast_day_summary;

CREATE VIEW forecast_day_summary AS
WITH ranked AS (
  SELECT
    target_date,
    city,
    prediction_value,
    timestamp_et,
    -- Determine if this prediction was made on the previous day (lead=D1/today_for_tomorrow)
    -- or on the same day (lead=D0/today_for_today)
    CASE
      WHEN lead_used IN ('D1', 'today_for_tomorrow') THEN 'prev_day'
      WHEN lead_used IN ('D0', 'today_for_today')    THEN 'same_day'
    END AS day_type,
    ROW_NUMBER() OVER (
      PARTITION BY target_date, city,
        CASE
          WHEN lead_used IN ('D1', 'today_for_tomorrow') THEN 'prev_day'
          WHEN lead_used IN ('D0', 'today_for_today')    THEN 'same_day'
        END
      ORDER BY
        CASE
          -- For prev_day: we want the LATEST prediction (last update before the day)
          WHEN lead_used IN ('D1', 'today_for_tomorrow') THEN 1
          -- For same_day: we want the EARLIEST prediction (first update of the day)
          WHEN lead_used IN ('D0', 'today_for_today')    THEN 2
        END,
        CASE
          WHEN lead_used IN ('D1', 'today_for_tomorrow') THEN timestamp_et
        END DESC,
        CASE
          WHEN lead_used IN ('D0', 'today_for_today') THEN timestamp_et
        END ASC
    ) AS rn
  FROM prediction_logs
  WHERE prediction_value IS NOT NULL
    AND lead_used IN ('D0', 'D1', 'today_for_today', 'today_for_tomorrow')
)
SELECT
  r.target_date,
  r.city,
  MAX(CASE WHEN r.day_type = 'prev_day' AND r.rn = 1 THEN r.prediction_value END) AS prev_day_latest_value,
  MAX(CASE WHEN r.day_type = 'same_day' AND r.rn = 1 THEN r.prediction_value END) AS same_day_earliest_value
FROM ranked r
WHERE r.rn = 1
GROUP BY r.target_date, r.city;

-- 7. Grant anon access to the view (so the frontend can read it)
GRANT SELECT ON forecast_day_summary TO anon;
GRANT SELECT ON forecast_day_summary TO authenticated;

-- 8. Refresh the PostgREST schema cache so the new column is immediately visible
--    (Supabase may do this automatically, but this ensures it)
NOTIFY pgrst, 'reload schema';

-- ============================================================================
-- DONE! After running this SQL:
--   ✅ log-prediction.js — already updated (city field + onConflict include city)
--   ✅ Frontend queries — already updated (.eq('city', currentCity))
--   ✅ prediction_writer.py — already sends city (it was failing before, now it works)
--   ✅ All existing rows default to 'nyc'
--
-- Deploy the code changes, then run this SQL in Supabase SQL Editor.
-- ============================================================================
