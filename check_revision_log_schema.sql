-- Check the actual schema of prediction_revision_log
SELECT 
  column_name,
  data_type,
  is_nullable
FROM information_schema.columns
WHERE table_name = 'prediction_revision_log'
ORDER BY ordinal_position;
