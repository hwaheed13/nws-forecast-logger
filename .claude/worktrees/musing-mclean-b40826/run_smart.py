# run_smart.py
import os

from city_config import DEFAULT_CITY

# Resolve city from --city arg or CITY env var
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--city", default=os.getenv("CITY", DEFAULT_CITY),
                     help="City key (nyc, lax, etc.)")
_args, _ = _parser.parse_known_args()

from nws_auto_logger import (
    set_city,
    ensure_csv_header,
    log_forecast,
    log_forecast_for_tomorrow,
    log_actual_today_if_after_6pm_local,
    upsert_yesterday_actual_if_morning_local,
    # NEW:
    write_today_bcp_snapshot_if_after_6pm,
)

set_city(_args.city)

# TASK can be set in the workflow "env" or via "workflow_dispatch" input mapping.
TASK = (os.getenv("TASK") or "smart_all").strip().lower()

def main():
    ensure_csv_header()

    if TASK == "forecast_today":
        log_forecast()
    elif TASK == "forecast_tomorrow":
        log_forecast_for_tomorrow()
    elif TASK == "actual_provisional":
        log_actual_today_if_after_6pm_local()          # only effective after 6pm local
        write_today_bcp_snapshot_if_after_6pm()        # only effective after 6pm local
    elif TASK == "actual_finalize":
        upsert_yesterday_actual_if_morning_local()     # only effective midnight–noon local
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
