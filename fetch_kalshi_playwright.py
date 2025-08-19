#!/usr/bin/env python3
"""
Fetch the current leading ("winner") range for the Kalshi
NYC highest temperature market by rendering the page with Playwright.

Writes kalshi_winner.json at repo root:
{
  "updated_at": <epoch>,
  "market": "NYC Highest Temperature",
  "winner": {"range": "75° to 76°", "probability": 99}
}
If nothing can be parsed, writes:
{"updated_at": ..., "market": "...", "winner": null}
and exits code 2.
"""
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc"
OUT = Path("kalshi_winner.json")

RANGE_RE = re.compile(r"(\d{2,3})\s*°\s*(?:to|–|-)\s*(\d{2,3})\s*°")
PCT_RE   = re.compile(r"(\d{1,3})\s*%")

def pick_winner_from_text(text: str):
    """
    Look through the full visible text, find all temperature ranges,
    and for each, consider percents that appear near it (±200 chars).
    Return (range_label, percent) for the highest percent, else None.
    """
    best = None
    for m in RANGE_RE.finditer(text):
        start, end = m.span()
        window = text[max(0, start-200):min(len(text), end+200)]
        candidates = [int(p) for p in PCT_RE.findall(window)]
        if candidates:
            pct = max(candidates)
            label = f"{m.group(1)}° to {m.group(2)}°"
            if not best or pct > best[1]:
                best = (label, pct)
    return best

def main() -> int:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Kalshi scrape for personal dashboard)"
            )
            page = ctx.new_page()
            page.goto(URL, timeout=60_000, wait_until="networkidle")
            # Give React/Next a moment to settle after network idle.
            page.wait_for_timeout(2000)
            # Grab fully rendered visible text
            text = page.evaluate("document.body.innerText")
            browser.close()
    except Exception as e:
        print(f"Playwright/navigation error: {e}", file=sys.stderr)
        # Write placeholder so dashboard hides the box gracefully
        OUT.write_text(json.dumps({
            "updated_at": int(time.time()),
            "market": "NYC Highest Temperature",
            "winner": None
        }, indent=2))
        return 1

    winner = pick_winner_from_text(text)

    if not winner:
        print("No winner parsed from rendered page.", file=sys.stderr)
        OUT.write_text(json.dumps({
            "updated_at": int(time.time()),
            "market": "NYC Highest Temperature",
            "winner": None
        }, indent=2))
        return 2

    rng, pct = winner
    payload = {
        "updated_at": int(time.time()),
        "market": "NYC Highest Temperature",
        "winner": {"range": rng, "probability": pct},
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} -> {payload}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
