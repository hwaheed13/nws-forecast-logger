-- Migration 007: Add ml_edge_tier column
-- Splits the single bet_signal into two orthogonal dimensions:
--   bet_signal      = conviction (STRONG_BET / BET / LEAN / SKIP) — model-only
--   ml_edge_tier    = market-mispricing tier (STRONG_EDGE / EDGE / NO_EDGE /
--                     MARKET_AHEAD / UNKNOWN) — model vs. market
--
-- A model can be STRONG_BET confident while the market already agrees
-- (NO_EDGE = no trade). The old schema conflated these.

ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS ml_edge_tier TEXT;

NOTIFY pgrst, 'reload schema';
