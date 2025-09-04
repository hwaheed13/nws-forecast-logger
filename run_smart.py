# run_smart.py
import os
from nws_auto_logger import (
    ensure_csv_header,
    log_forecast,
    log_forecast_for_tomorrow,
    log_actual_today_if_after_6pm_local,
    upsert_yesterday_actual_if_morning_local,
    # NEW:
    write_today_bcp_snapshot_if_after_6pm,
)

# TASK can be set in the workflow "env" or via "workflow_dispatch" input mapping.
TASK = (os.getenv("TASK") or "smart_all").strip().lower()

def main():
    ensure_csv_header()

    if TASK == "forecast_today":
        log_forecast()
    elif TASK == "forecast_tomorrow":
        log_forecast_for_tomorrow()
    elif TASK == "actual_provisional":
        log_actual_today_if_after_6pm_local()          # only effective after 6pm ET
        write_today_bcp_snapshot_if_after_6pm()        # only effective after 6pm ET
    elif TASK == "actual_finalize":
        upsert_yesterday_actual_if_morning_local()     # only effective midnight–noon ET
    elif TASK == "bcp_snapshot":
        write_today_bcp_snapshot_if_after_6pm()        # optional manual hook
    else:
        # smart_all (default): run everything idempotently
        log_forecast()
        log_forecast_for_tomorrow()
        log_actual_today_if_after_6pm_local()
        upsert_yesterday_actual_if_morning_local()
        write_today_bcp_snapshot_if_after_6pm()

if __name__ == "__main__":
    main()
