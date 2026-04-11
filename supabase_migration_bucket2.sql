-- Migration: add bucket-rank outcome tracking columns to prediction_logs
-- Run this once in Supabase → SQL Editor

ALTER TABLE prediction_logs
  ADD COLUMN IF NOT EXISTS ml_bucket_2       TEXT,
  ADD COLUMN IF NOT EXISTS ml_bucket_2_prob  FLOAT,
  ADD COLUMN IF NOT EXISTS bucket_rank_hit   INTEGER;
  -- bucket_rank_hit values:
  --   1 = bucket 1 (top pick) was correct
  --   2 = bucket 1 missed, bucket 2 was correct
  --   0 = both buckets missed
  -- NULL = not yet scored (prediction still in future)

-- Optional: index for fast stat queries
CREATE INDEX IF NOT EXISTS idx_prediction_logs_bucket_rank
  ON prediction_logs (city, bucket_rank_hit)
  WHERE bucket_rank_hit IS NOT NULL;
