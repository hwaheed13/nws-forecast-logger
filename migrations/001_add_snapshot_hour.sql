-- Migration: Add snapshot_hour column for 3-hourly snapshot archival
-- Date: 2026-04-17
-- Purpose: Enable training pipeline to see intraday atmospheric progression

ALTER TABLE prediction_logs
ADD COLUMN snapshot_hour INTEGER DEFAULT NULL;

-- Create index for efficient queries by snapshot_hour
CREATE INDEX IF NOT EXISTS idx_prediction_logs_snapshot_hour
ON prediction_logs(target_date, snapshot_hour)
WHERE snapshot_hour IS NOT NULL;

-- Add comment for clarity
COMMENT ON COLUMN prediction_logs.snapshot_hour IS
'3-hour bucket for snapshot archival (0, 3, 6, 9, 12, 15, 18, 21).
Enables training pipeline to see intraday atmospheric progression.
Values: 0=midnight-3am, 3=3-6am, 6=6-9am, 9=9am-noon, 12=noon-3pm, 15=3-6pm, 18=6-9pm, 21=9pm-midnight';
