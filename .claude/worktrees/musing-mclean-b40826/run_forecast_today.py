from nws_auto_logger import ensure_csv_header, log_forecast
if __name__ == "__main__":
    ensure_csv_header()
    log_forecast()  # idempotent; only logs if different/not already present

