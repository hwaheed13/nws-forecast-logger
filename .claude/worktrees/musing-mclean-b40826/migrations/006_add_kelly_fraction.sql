-- Migration 006: Add kelly_fraction column
-- Half-Kelly position sizing: 0.5 * (p_model - p_market) / (1 - p_market)
-- Tells you what fraction of your bankroll to bet on this prediction.
-- Only written at canonical write time (alongside market_prob_at_prediction).
-- kelly_fraction = 0 means no edge / skip.
-- kelly_fraction = 0.12 means bet 12% of bankroll.
-- Run in Supabase SQL Editor before deploying prediction_writer.py.

ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS kelly_fraction NUMERIC;

-- Notify PostgREST to pick up schema changes
NOTIFY pgrst, 'reload schema';
