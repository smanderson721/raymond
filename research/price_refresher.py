#!/usr/bin/env python3
"""
Daily Price Refresher — fetches 30-day price history for the top stocks
ranked by (precondition_score + 2 × catalyst_score).

This is intentionally a small universe (~250 tickers) so the precondition
scan can run only weekly while pct_30d stays fresh for BUP scoring.

Output: research_output/price_data.json
    {
        "fetched_at": "2026-05-08T...",
        "universe_size": 250,
        "tickers": {
            "AAPL": {"price": 150.0, "pct_30d": 0.034}
        }
    }
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output",
)
SCAN_FILE = os.path.join(OUTPUT_DIR, "scan_results.json")
SCORES_FILE = os.path.join(OUTPUT_DIR, "catalyst_scores.json")
PRICE_FILE = os.path.join(OUTPUT_DIR, "price_data.json")


def _load_latest_scan() -> dict[str, dict]:
    if not os.path.exists(SCAN_FILE):
        return {}
    with open(SCAN_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    scans = db.get("scans", [])
    if not scans:
        return {}
    return {s["ticker"]: s for s in scans[0].get("stocks", [])}


def _load_latest_catalyst_scores() -> dict[str, dict]:
    if not os.path.exists(SCORES_FILE):
        return {}
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
    sessions = db.get("sessions", [])
    if not sessions:
        return {}
    # Latest session (sessions list grows by appending — last is most recent)
    return sessions[-1].get("scores", {})


def _select_universe(top_n: int = 250) -> list[str]:
    """Top N tickers by precondition_score + 2 × catalyst_score."""
    scan = _load_latest_scan()
    cats = _load_latest_catalyst_scores()

    if not scan and not cats:
        return []

    universe = set(scan.keys()) | set(cats.keys())
    ranked = []
    for t in universe:
        p = (scan.get(t) or {}).get("precondition_score", 0) or 0
        c = (cats.get(t) or {}).get("score", 0) or 0
        combined = p + 2 * c
        if combined > 0:
            ranked.append((t, combined, p, c))

    ranked.sort(key=lambda r: r[1], reverse=True)
    return [r[0] for r in ranked[:top_n]]


def _fetch_pct_30d(ticker: str) -> dict | None:
    """Capture today's close, daily change, 7d change, and 30d rise."""
    import yfinance as yf
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo")
        if len(hist) < 2:
            return None
        closes = hist["Close"].tolist()
        new_close = float(closes[-1])
        prev_close = float(closes[-2])
        low_close = float(min(closes))
        if low_close <= 0 or prev_close <= 0:
            return None
        # 7-day prior close: ~5 trading days back
        seven_close = float(closes[-6]) if len(closes) >= 6 else prev_close
        # Absolute daily dollar move
        change = new_close - prev_close
        # Today's volume (last bar)
        try:
            volume = int(hist["Volume"].iloc[-1])
        except Exception:
            volume = None
        return {
            "price": new_close,
            "prev_close": prev_close,
            "change": change,
            "pct_1d": (new_close - prev_close) / prev_close,
            "pct_7d": (new_close - seven_close) / seven_close if seven_close > 0 else None,
            "pct_30d": (new_close - low_close) / low_close,
            "volume": volume,
        }
    except Exception:
        return None


def refresh_prices(top_n: int = 250) -> dict:
    universe = _select_universe(top_n=top_n)
    if not universe:
        print("  [price_refresher] No scan/catalyst data found — nothing to refresh.")
        return {}

    print(f"\n── Price refresh: top {len(universe)} tickers by P + 2C ──\n", flush=True)

    # Per-call delay for shared/CI IPs (GitHub Actions yfinance rate limits)
    rate_delay = float(os.environ.get("YFINANCE_RATE_DELAY", "0.2") or 0.2)

    results: dict[str, dict] = {}
    failed = 0
    for i, ticker in enumerate(universe, 1):
        data = _fetch_pct_30d(ticker)
        if data is None:
            failed += 1
            print(f"  {i}/{len(universe)} {ticker} — FAILED", flush=True)
        else:
            results[ticker] = data
            d = data["pct_1d"] * 100
            m = data["pct_30d"] * 100
            print(f"  {i}/{len(universe)} {ticker} — 1d {d:+.1f}%  30d {m:+.1f}%", flush=True)
        if rate_delay > 0:
            time.sleep(rate_delay)

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "fetched_count": len(results),
        "failed_count": failed,
        "tickers": results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PRICE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Done. {len(results)} fetched, {failed} failed → {PRICE_FILE}")
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=250)
    args = ap.parse_args()
    refresh_prices(top_n=args.top_n)
