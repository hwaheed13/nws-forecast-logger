-- Migration 003: Add ML v2 columns for bucket classifier predictions
--
-- ml_bucket_probs: JSONB storing top-5 bucket probabilities from the classifier
--   e.g., {"48-49": 0.42, "49-50": 0.31, "50-51": 0.15, ...}
-- ml_version: Text identifying which model version produced the prediction
--   e.g., "v2_atm_classifier" or "v1" (backward compat)

ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS ml_bucket_probs JSONB;

ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS ml_version TEXT;

-- Notify PostgREST to pick up schema changes
NOTIFY pgrst, 'reload schema';
