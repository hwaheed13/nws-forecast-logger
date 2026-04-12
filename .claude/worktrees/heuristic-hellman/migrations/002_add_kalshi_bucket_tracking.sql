-- Migration: Add bucket tracking columns to prediction_logs
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- ============================================================================

-- 1. Add kalshi_bucket column (the bucket label we picked, e.g. "43° to 44°")
ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS kalshi_bucket TEXT;

-- 2. Add kalshi_confidence column (our probability %, e.g. 72.0)
ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS kalshi_confidence NUMERIC;

-- 3. Add kalshi_clamped column (whether AccuWeather overrode the ensemble)
ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS kalshi_clamped BOOLEAN DEFAULT false;

-- 4. Refresh PostgREST schema cache
NOTIFY pgrst, 'reload schema';

-- ============================================================================
-- DONE! These columns will be populated by the browser's prediction logger
-- going forward. Historical rows will have NULL for these columns.
-- ============================================================================
