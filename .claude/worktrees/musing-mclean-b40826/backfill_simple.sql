-- Simple step-by-step backfill of obs_heating_rate and obs_cloud_cover
-- Execute in Supabase SQL Editor

-- Step 1: Create a temporary table with computed heating rates for NYC
CREATE TEMP TABLE heating_rates_nyc AS
SELECT
  pl.id,
  ROUND(
    (MAX(EXTRACT(EPOCH FROM o.observed_at - MIN(o.observed_at) OVER (PARTITION BY pl.id))) / 3600.0) *
    (MAX(o.temp_f) OVER (PARTITION BY pl.id) - MIN(o.temp_f) OVER (PARTITION BY pl.id)) /
    (MAX(EXTRACT(EPOCH FROM o.observed_at - MIN(o.observed_at) OVER (PARTITION BY pl.id))) / 3600.0)
    , 2
  ) as heating_rate
FROM prediction_logs pl
LEFT JOIN nws_observations o ON
  DATE(o.observed_at) = pl.target_date::date
  AND o.city = pl.city
WHERE pl.city = 'nyc'
  AND o.temp_f IS NOT NULL
GROUP BY pl.id;

-- Step 2: Create a temporary table with cloud cover for NYC
CREATE TEMP TABLE cloud_cover_nyc AS
SELECT DISTINCT ON (pl.id)
  pl.id,
  CASE
    WHEN lower(o.sky_condition) LIKE '%skc%' OR lower(o.sky_condition) LIKE '%clr%' THEN 0
    WHEN lower(o.sky_condition) LIKE '%few%' THEN 25
    WHEN lower(o.sky_condition) LIKE '%sct%' THEN 50
    WHEN lower(o.sky_condition) LIKE '%bkn%' THEN 75
    WHEN lower(o.sky_condition) LIKE '%ovc%' THEN 100
    ELSE NULL
  END as cloud_cover
FROM prediction_logs pl
LEFT JOIN nws_observations o ON
  DATE(o.observed_at) = pl.target_date::date
  AND o.city = pl.city
WHERE pl.city = 'nyc'
  AND o.sky_condition IS NOT NULL
ORDER BY pl.id, o.observed_at DESC;

-- Step 3: Update prediction_logs for NYC
UPDATE prediction_logs pl
SET atm_snapshot = jsonb_set(
  jsonb_set(
    COALESCE(pl.atm_snapshot, '{}'::jsonb),
    '{obs_snap_heating_rate}',
    to_jsonb(hr.heating_rate)
  ),
  '{obs_snap_cloud_cover}',
  to_jsonb(cc.cloud_cover)
)
FROM heating_rates_nyc hr
LEFT JOIN cloud_cover_nyc cc ON hr.id = cc.id
WHERE pl.id = hr.id
  AND pl.city = 'nyc'
  AND (pl.atm_snapshot->>'obs_snap_heating_rate' IS NULL OR pl.atm_snapshot->>'obs_snap_heating_rate' = 'null');

-- Step 4: Repeat for LAX (same structure)
CREATE TEMP TABLE heating_rates_lax AS
SELECT
  pl.id,
  ROUND(
    (MAX(EXTRACT(EPOCH FROM o.observed_at - MIN(o.observed_at) OVER (PARTITION BY pl.id))) / 3600.0) *
    (MAX(o.temp_f) OVER (PARTITION BY pl.id) - MIN(o.temp_f) OVER (PARTITION BY pl.id)) /
    (MAX(EXTRACT(EPOCH FROM o.observed_at - MIN(o.observed_at) OVER (PARTITION BY pl.id))) / 3600.0)
    , 2
  ) as heating_rate
FROM prediction_logs pl
LEFT JOIN nws_observations o ON
  DATE(o.observed_at) = pl.target_date::date
  AND o.city = pl.city
WHERE pl.city = 'lax'
  AND o.temp_f IS NOT NULL
GROUP BY pl.id;

CREATE TEMP TABLE cloud_cover_lax AS
SELECT DISTINCT ON (pl.id)
  pl.id,
  CASE
    WHEN lower(o.sky_condition) LIKE '%skc%' OR lower(o.sky_condition) LIKE '%clr%' THEN 0
    WHEN lower(o.sky_condition) LIKE '%few%' THEN 25
    WHEN lower(o.sky_condition) LIKE '%sct%' THEN 50
    WHEN lower(o.sky_condition) LIKE '%bkn%' THEN 75
    WHEN lower(o.sky_condition) LIKE '%ovc%' THEN 100
    ELSE NULL
  END as cloud_cover
FROM prediction_logs pl
LEFT JOIN nws_observations o ON
  DATE(o.observed_at) = pl.target_date::date
  AND o.city = pl.city
WHERE pl.city = 'lax'
  AND o.sky_condition IS NOT NULL
ORDER BY pl.id, o.observed_at DESC;

UPDATE prediction_logs pl
SET atm_snapshot = jsonb_set(
  jsonb_set(
    COALESCE(pl.atm_snapshot, '{}'::jsonb),
    '{obs_snap_heating_rate}',
    to_jsonb(hr.heating_rate)
  ),
  '{obs_snap_cloud_cover}',
  to_jsonb(cc.cloud_cover)
)
FROM heating_rates_lax hr
LEFT JOIN cloud_cover_lax cc ON hr.id = cc.id
WHERE pl.id = hr.id
  AND pl.city = 'lax'
  AND (pl.atm_snapshot->>'obs_snap_heating_rate' IS NULL OR pl.atm_snapshot->>'obs_snap_heating_rate' = 'null');

-- Step 5: Verify the results
SELECT
  city,
  COUNT(*) as total_rows,
  SUM(CASE WHEN atm_snapshot->>'obs_snap_heating_rate' IS NOT NULL AND atm_snapshot->>'obs_snap_heating_rate' != 'null' THEN 1 ELSE 0 END) as rows_with_heating_rate,
  SUM(CASE WHEN atm_snapshot->>'obs_snap_cloud_cover' IS NOT NULL AND atm_snapshot->>'obs_snap_cloud_cover' != 'null' THEN 1 ELSE 0 END) as rows_with_cloud_cover
FROM prediction_logs
WHERE city IN ('nyc', 'lax')
GROUP BY city
ORDER BY city;
