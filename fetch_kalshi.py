#!/usr/bin/env python3
import json, re, sys, time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc"
OUT = Path("kalshi_winner.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (+Kalshi-scrape for personal dashboard)"
}

def parse_winner(html: str):
    """
    Heuristic:
    - Find candidate ranges like '75° to 76°'
    - For each, look for a percentage like '99%' nearby
    - Pick the one with the highest percentage
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    ranges = list(re.finditer(r"(\d{2,3})\s*°\s*to\s*(\d{2,3})\s*°", text))
    if not ranges:
        return None

    best = None
    for m in ranges:
        start, end = m.span()
        window = text[max(0, start-120): min(len(text), end+120)]
        percents = re.findall(r"(\d{1,3})\s*%", window)
        if not percents:
            continue
        pmax = max(int(p) for p in percents if p.isdigit())
        rng = f"{m.group(1)}° to {m.group(2)}°"
        if (best is None) or (pmax > best["probability"]):
            best = {"range": rng, "probability": pmax}

    return best

def main():
    try:
        r = requests.get(URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        winner = parse_winner(r.text)
        if not winner:
            print("No winner parsed", file=sys.stderr)
            return 2

        payload = {
            "updated_at": int(time.time()),
            "market": "NYC Highest Temperature",
            "winner": {
                "range": winner["range"],
                "probability": winner["probability"]
            }
        }
        OUT.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {OUT} -> {payload}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
