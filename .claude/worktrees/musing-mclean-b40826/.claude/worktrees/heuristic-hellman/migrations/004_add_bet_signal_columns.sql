-- Migration 004: Add bet signal + market snapshot columns
-- Run in Supabase SQL Editor after deploying code changes

-- Bet signal: STRONG_BET / BET / LEAN / SKIP
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS bet_signal TEXT;

-- Full Kalshi market state at prediction time (for analysis/retraining)
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS kalshi_market_snapshot JSONB;

-- Edge: model confidence minus market probability (positive = model sees value)
ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS ml_edge NUMERIC;

-- Notify PostgREST to pick up schema changes
NOTIFY pgrst, 'reload schema';
