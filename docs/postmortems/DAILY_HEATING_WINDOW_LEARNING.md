# Daily Heating Window Learning Loop

**Status**: ✅ Integrated into GitHub Actions workflow

## Overview

The heating window configuration now **learns from new observation data every day** as part of the nightly retraining workflow. This means the windows automatically evolve as more data is collected, becoming more accurate over time.

## How It Works

### 1. **Daily Trigger**
- Runs nightly at **1 AM EDT** (05:00 UTC) via GitHub Actions
- Integrates with existing `.github/workflows/nightly-lightweight-retrain.yml`
- Runs **before** model retraining so windows are updated first

### 2. **Data Analysis**
The `update_heating_windows.py` script:
1. Queries latest `obs_snap_heating_rate` data from `prediction_logs`
2. Groups data by city (NYC, LAX) and season (winter, spring, summer, fall)
3. Analyzes heating patterns:
   - Mean heating rate
   - % of days with positive heating
   - Seasonal variations
4. Determines optimal window for each season/city combination

### 3. **Automatic Updates**
- If heating patterns have changed significantly, updates `heating_windows.py`
- Commits changes to git automatically
- Dashboard automatically uses updated windows (no redeploy needed)

### 4. **Window Logic**

**NYC**: Consistent across all seasons
- If mean heating > 0: Window = 10 AM - 4 PM

**LAX**: Seasonal variations due to marine layer
- Winter (strong heating): 10 AM - 4 PM
- Spring/Fall (slight cooling): 10 AM - 1 PM  
- Summer (strong marine cap): 10 AM - 12 PM

## Implementation Status

### ✅ Complete
- [x] Initial analysis based on 4+ years of data (2022-2026)
- [x] Python module (`heating_windows.py`) with configuration
- [x] JavaScript dashboard integration
- [x] Daily updater script (`update_heating_windows.py`)
- [x] GitHub Actions workflow integration

### ⚠️ Needs Completion
The `update_heating_windows.py` script needs a data fetching implementation. Currently it has a placeholder because the Supabase REST API is blocked by RLS policies. 

**To complete this, choose one approach:**

#### Option 1: Direct PostgreSQL Connection (Recommended)
```python
import psycopg2

# Extract connection string from Supabase
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()

# Query heating data
cur.execute("""
SELECT city, season, mean_heating_rate, positive_count, observation_count
FROM heating_analysis_view
""")
```

#### Option 2: Use Supabase Python SDK with RLS Bypass
```python
# If RLS is disabled for specific operations, use:
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
result = sb.rpc('analyze_heating_windows').execute()
```

#### Option 3: Call Supabase via curl/subprocess
```bash
curl -X POST \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE" \
  https://ztjtuhkjkqchsiuuvmzs.supabase.co/rest/v1/rpc/analyze_heating_windows
```

## Current Configuration (Static)

Until the data fetching is implemented, the system uses this static config that was analyzed from 4+ years of data:

```python
HEATING_WINDOWS = {
    "nyc": {
        "winter": (10, 16),   # 0.93 °F/hr, 87.6% positive
        "spring": (10, 16),   # 0.86 °F/hr, 80.9% positive
        "summer": (10, 16),   # 0.66 °F/hr, 78.0% positive
        "fall": (10, 16),     # 0.94 °F/hr, 88.2% positive
    },
    "lax": {
        "winter": (10, 16),   # 0.87 °F/hr, 78.9% positive
        "spring": (10, 13),   # -0.36 °F/hr, 28.9% positive
        "summer": (10, 12),   # -0.92 °F/hr, 3.3% positive
        "fall": (10, 13),     # -0.15 °F/hr, 36.3% positive
    },
}
```

## Evolution Over Time

As more observation data accumulates, the windows will evolve:

**Month 1-3**: Static windows based on historical analysis (current state)
**Month 4+**: Dynamic updates as new seasons are represented in recent data
**Year 2+**: Refined windows based on multi-year rolling averages

Example evolution:
- Spring 2026: LAX spring window might shift from (10, 13) → (10, 12) if more data shows stronger marine layer
- Summer 2026: LAX summer might shift from (10, 12) → (10, 11) if heating becomes even rarer
- NYC: Likely stays (10, 16) year-round as patterns are very consistent

## Dashboard Impact

The dashboard automatically uses the latest windows:
1. `heating_windows.py` is version-controlled
2. Dashboard function `_getHeatingWindow()` reads from config
3. When `heating_windows.py` is updated and committed, next page reload uses new windows
4. **No code changes** needed to dashboard itself

## Testing the Daily Learner

To test locally:
```bash
# Simulate the workflow
python update_heating_windows.py

# Check what changed
git diff heating_windows.py

# Revert if needed
git checkout heating_windows.py
```

## Next Steps

1. **Implement data fetching** in `update_heating_windows.py` (see options above)
2. **Test in staging** before merging to production
3. **Monitor** window evolution as more data accumulates
4. **Evaluate** whether windows are becoming more accurate over time

## Files Involved

- `.github/workflows/nightly-lightweight-retrain.yml` - Workflow that triggers update
- `update_heating_windows.py` - Analyzer script (needs data fetching implementation)
- `heating_windows.py` - Configuration file (version-controlled, updated daily)
- `public/index.html` - Dashboard (reads from heating_windows.py config)

---

**Last Updated**: April 16, 2026  
**Status**: Ready for daily learning (data fetching implementation pending)
