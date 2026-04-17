-- Backfill obs_heating_rate and obs_cloud_cover into prediction_logs atm_snapshot
-- These queries compute the missing observation features from raw NWS data

-- Helper function: Convert sky condition text to cloud cover percentage
-- This will be used in the UPDATE query below
CREATE OR REPLACE FUNCTION sky_to_cloud_cover(sky_str TEXT) RETURNS FLOAT AS $$
BEGIN
  CASE lower(trim(sky_str))
    WHEN 'skc' THEN RETURN 0;      -- sky clear
    WHEN 'clr' THEN RETURN 0;      -- clear
    WHEN 'few' THEN RETURN 25;     -- 1/8 to 2/8 coverage
    WHEN 'sct' THEN RETURN 50;     -- 3/8 to 4/8 coverage (scattered)
    WHEN 'bkn' THEN RETURN 75;     -- 5/8 to 7/8 coverage (broken)
    WHEN 'ovc' THEN RETURN 100;    -- 8/8 coverage (overcast)
    WHEN 'vv' THEN RETURN NULL;    -- vertical visibility (obscured)
    ELSE RETURN NULL;
  END CASE;
END;
$$ LANGUAGE plpgsql;

-- Backfill NYC: Update prediction_logs with computed obs_heating_rate and obs_cloud_cover
UPDATE prediction_logs pl
SET atm_snapshot =
  CASE
    WHEN atm_snapshot IS NULL THEN jsonb_build_object(
      'obs_snap_heating_rate', computed_heating.rate,
      'obs_snap_cloud_cover', computed_cloud.cover
    )
    ELSE jsonb_set(
      jsonb_set(atm_snapshot, '{obs_snap_heating_rate}', to_jsonb(computed_heating.rate)),
      '{obs_snap_cloud_cover}',
      to_jsonb(computed_cloud.cover)
    )
  END
FROM (
  -- Compute heating rate for each target_date
  SELECT
    pl_inner.id,
    pl_inner.target_date,
    CASE
      WHEN COUNT(DISTINCT o.observed_at) < 2 THEN NULL
      ELSE
        (EXTRACT(EPOCH FROM MAX(o.observed_at) - MIN(o.observed_at)) / 3600.0)
        * (MAX(o.temp_f) - MIN(o.temp_f)) /
        NULLIF(EXTRACT(EPOCH FROM MAX(o.observed_at) - MIN(o.observed_at)), 0)
    END as rate
  FROM prediction_logs pl_inner
  LEFT JOIN nws_observations o ON
    DATE(o.observed_at) = pl_inner.target_date::date
    AND o.city = pl_inner.city
  WHERE pl_inner.city = 'nyc'
    AND (pl_inner.atm_snapshot->>'obs_snap_heating_rate' IS NULL
         OR pl_inner.atm_snapshot->>'obs_snap_heating_rate' = 'null')
  GROUP BY pl_inner.id, pl_inner.target_date
) computed_heating
LEFT JOIN (
  -- Compute cloud cover from latest observation
  SELECT
    pl_inner.id,
    pl_inner.target_date,
    sky_to_cloud_cover(o.sky_condition) as cover
  FROM prediction_logs pl_inner
  LEFT JOIN nws_observations o ON
    DATE(o.observed_at) = pl_inner.target_date::date
    AND o.city = pl_inner.city
  WHERE pl_inner.city = 'nyc'
    AND (pl_inner.atm_snapshot->>'obs_snap_cloud_cover' IS NULL
         OR pl_inner.atm_snapshot->>'obs_snap_cloud_cover' = 'null')
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pl_inner.id ORDER BY o.observed_at DESC) = 1
) computed_cloud ON computed_heating.id = computed_cloud.id
WHERE pl.city = 'nyc'
  AND pl.id = computed_heating.id;

-- Backfill LAX: Update prediction_logs with computed obs_heating_rate and obs_cloud_cover
UPDATE prediction_logs pl
SET atm_snapshot =
  CASE
    WHEN atm_snapshot IS NULL THEN jsonb_build_object(
      'obs_snap_heating_rate', computed_heating.rate,
      'obs_snap_cloud_cover', computed_cloud.cover
    )
    ELSE jsonb_set(
      jsonb_set(atm_snapshot, '{obs_snap_heating_rate}', to_jsonb(computed_heating.rate)),
      '{obs_snap_cloud_cover}',
      to_jsonb(computed_cloud.cover)
    )
  END
FROM (
  -- Compute heating rate for each target_date
  SELECT
    pl_inner.id,
    pl_inner.target_date,
    CASE
      WHEN COUNT(DISTINCT o.observed_at) < 2 THEN NULL
      ELSE
        (EXTRACT(EPOCH FROM MAX(o.observed_at) - MIN(o.observed_at)) / 3600.0)
        * (MAX(o.temp_f) - MIN(o.temp_f)) /
        NULLIF(EXTRACT(EPOCH FROM MAX(o.observed_at) - MIN(o.observed_at)), 0)
    END as rate
  FROM prediction_logs pl_inner
  LEFT JOIN nws_observations o ON
    DATE(o.observed_at) = pl_inner.target_date::date
    AND o.city = pl_inner.city
  WHERE pl_inner.city = 'lax'
    AND (pl_inner.atm_snapshot->>'obs_snap_heating_rate' IS NULL
         OR pl_inner.atm_snapshot->>'obs_snap_heating_rate' = 'null')
  GROUP BY pl_inner.id, pl_inner.target_date
) computed_heating
LEFT JOIN (
  -- Compute cloud cover from latest observation
  SELECT
    pl_inner.id,
    pl_inner.target_date,
    sky_to_cloud_cover(o.sky_condition) as cover
  FROM prediction_logs pl_inner
  LEFT JOIN nws_observations o ON
    DATE(o.observed_at) = pl_inner.target_date::date
    AND o.city = pl_inner.city
  WHERE pl_inner.city = 'lax'
    AND (pl_inner.atm_snapshot->>'obs_snap_cloud_cover' IS NULL
         OR pl_inner.atm_snapshot->>'obs_snap_cloud_cover' = 'null')
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pl_inner.id ORDER BY o.observed_at DESC) = 1
) computed_cloud ON computed_heating.id = computed_cloud.id
WHERE pl.city = 'lax'
  AND pl.id = computed_heating.id;

-- Verify the backfill (check how many rows were updated)
SELECT
  city,
  COUNT(*) as total_rows,
  SUM(CASE WHEN atm_snapshot->>'obs_snap_heating_rate' IS NOT NULL AND atm_snapshot->>'obs_snap_heating_rate' != 'null' THEN 1 ELSE 0 END) as with_heating_rate,
  SUM(CASE WHEN atm_snapshot->>'obs_snap_cloud_cover' IS NOT NULL AND atm_snapshot->>'obs_snap_cloud_cover' != 'null' THEN 1 ELSE 0 END) as with_cloud_cover
FROM prediction_logs
WHERE city IN ('nyc', 'lax')
GROUP BY city;
