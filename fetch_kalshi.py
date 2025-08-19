#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

URL = "https://kalshi.com/markets/kxhighny/highest-temperature-in-nyc"
OUT = Path("kalshi_winner.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Kalshi-scrape for personal dashboard)",
    "Accept-Language": "en-US,en;q=0.9",
}

RANGE_RE = re.compile(r"(\d{2,3})\s*°\s*(?:to|–|-)\s*(\d{2,3})\s*°")
PCT_RE = re.compile(r"(\d{1,3})\s*%")

def _normalize_prob(x: Any) -> Optional[int]:
    """
    Return an integer percent 0..100 from common shapes:
    - 0..1 floats
    - 0..100 ints/floats
    - strings like '99%' or '0.87'
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if 0 <= x <= 1:
            return int(round(x * 100))
        if 0 <= x <= 100:
            return int(round(x))
        return None
    if isinstance(x, str):
        m = PCT_RE.search(x)
        if m:
            return int(m.group(1))
        try:
            v = float(x.strip().replace("%", ""))
            return _normalize_prob(v)
        except Exception:
            return None
    return None

def _best_of(candidates: list[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[1], reverse=True)
    return candidates[0]

def parse_from_next_data(data: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """
    Walk the Next.js data looking for contracts/options with a temp range label
    and a probability/price field.
    """
    best: Optional[Tuple[str, int]] = None

    def walk(node: Any):
        nonlocal best
        if isinstance(node, dict):
            # Try to detect a label and probability nearby
            label = None
            prob = None

            # Common label-ish keys
            for k in ("label", "name", "display", "title", "strike_display", "range"):
                v = node.get(k)
                if isinstance(v, str) and RANGE_RE.search(v):
                    label = v
                    break

            # Common probability/price-ish keys
            for k in ("probability", "prob", "price", "last_price", "implied_probability"):
                if k in node:
                    prob = _normalize_prob(node.get(k))
                    if prob is not None:
                        break

            if label and prob is not None:
                # Normalize label to "A° to B°"
                m = RANGE_RE.search(label)
                if m:
                    rng = f"{m.group(1)}° to {m.group(2)}°"
                    cand = (rng, prob)
                    if not best or cand[1] > best[1]:
                        best = cand

            # Continue walking
            for v in node.values():
                walk(v)

        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return best

def parse_from_meta(soup: BeautifulSoup) -> Optional[Tuple[str, int]]:
    tag = soup.find("meta", attrs={"property": "og:description"}) or \
          soup.find("meta", attrs={"name": "description"})
    if not tag:
        return None
    content = tag.get("content") or ""
    m_range = RANGE_RE.search(content)
    m_pct = PCT_RE.search(content)
    if m_range and m_pct:
        rng = f"{m_range.group(1)}° to {m_range.group(2)}°"
        return (rng, int(m_pct.group(1)))
    return None

def parse_from_plain_text(soup: BeautifulSoup) -> Optional[Tuple[str, int]]:
    text = soup.get_text(" ", strip=True)
    ranges = list(RANGE_RE.finditer(text))
    candidates: list[Tuple[str, int]] = []
    for m in ranges:
        start, end = m.span()
        window = text[max(0, start-160): min(len(text), end+160)]
        percents = [int(p) for p in PCT_RE.findall(window)]
        if percents:
            p = max(percents)
            candidates.append((f"{m.group(1)}° to {m.group(2)}°", p))
    return _best_of(candidates)

def main() -> int:
    try:
        r = requests.get(URL, headers=HEADERS, timeout=25)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 1

    soup = BeautifulSoup(html, "html.parser")

    # 1) Try Next.js JSON
    winner: Optional[Tuple[str, int]] = None
    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            data = json.loads(script.string)
            winner = parse_from_next_data(data)
        except Exception as e:
            print(f"NEXT_DATA parse error: {e}", file=sys.stderr)

    # 2) Try meta description
    if not winner:
        winner = parse_from_meta(soup)

    # 3) Fallback heuristic on plain text
    if not winner:
        winner = parse_from_plain_text(soup)

    if not winner:
        print("No winner parsed (site layout likely changed).", file=sys.stderr)
        # Write a placeholder so the dashboard hides the box gracefully
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
