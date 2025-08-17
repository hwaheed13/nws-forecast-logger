from nws_auto_logger import ensure_csv_header, log_yesterday_actual
if __name__ == "__main__":
    ensure_csv_header()
    log_yesterday_actual()  # appends once per day

