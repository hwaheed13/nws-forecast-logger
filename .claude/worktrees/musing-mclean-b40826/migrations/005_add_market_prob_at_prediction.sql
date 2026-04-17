-- Migration 005: Add market_prob_at_prediction column
-- Captures the Kalshi market's live probability for our predicted bucket
-- at canonical write time (not settled end-of-day prices).
-- This is the exact market price we were betting "against" — enables
-- true edge analysis: ml_confidence - market_prob_at_prediction.
-- Run in Supabase SQL Editor before deploying prediction_writer.py.

ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS market_prob_at_prediction NUMERIC;

-- Notify PostgREST to pick up schema changes
NOTIFY pgrst, 'reload schema';
