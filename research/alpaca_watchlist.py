#!/usr/bin/env python3
"""
Push the top-N BUP scorers to an Alpaca watchlist.

Reads the same JSON files the pitches.html Market Scan tab reads, applies
the BUP formula, and replaces the contents of a named watchlist on Alpaca.

BUP = (precondition_score + 2 × catalyst_score) − √(max(round(pct_30d × 100), 0))

Required env vars:
    ALPACA_API_KEY_ID
    ALPACA_API_SECRET_KEY
    ALPACA_BASE_URL   (optional — defaults to paper trading endpoint)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output",
)
SCAN_FILE = os.path.join(OUTPUT_DIR, "scan_results.json")
SCORES_FILE = os.path.join(OUTPUT_DIR, "catalyst_scores.json")
PRICE_FILE = os.path.join(OUTPUT_DIR, "price_data.json")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "watchlist_history.json")
COMBINED_FILE = os.path.join(OUTPUT_DIR, "watchlist_combined.json")

DEFAULT_WATCHLIST_NAME = "Raymond Top 100 BUP"
HISTORY_WINDOW_DAYS = 14
PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_scan() -> dict[str, dict]:
    d = _load_json(SCAN_FILE)
    scans = d.get("scans", [])
    if not scans:
        return {}
    return {s["ticker"]: s for s in scans[0].get("stocks", [])}


def _latest_catalyst_scores() -> dict[str, dict]:
    d = _load_json(SCORES_FILE)
    sessions = d.get("sessions", [])
    if not sessions:
        return {}
    return sessions[-1].get("scores", {})


def _latest_prices() -> dict[str, dict]:
    d = _load_json(PRICE_FILE)
    return d.get("tickers", {}) or {}


def _compute_bup_top(top_n: int) -> list[tuple[str, float]]:
    scan = _latest_scan()
    cats = _latest_catalyst_scores()
    prices = _latest_prices()

    universe = set(scan.keys()) | set(cats.keys())
    ranked: list[tuple[str, float]] = []

    for ticker in universe:
        p = (scan.get(ticker) or {}).get("precondition_score", 0) or 0
        c = (cats.get(ticker) or {}).get("score", 0) or 0
        total = p + 2 * c
        pct = (prices.get(ticker) or {}).get("pct_30d")
        if pct is None:
            # No price data — skip if there's also no catalyst. With no penalty
            # info, including high-precondition-only stocks pollutes the list.
            if c == 0:
                continue
            pct = 0.0
        pct_int = max(round(pct * 100), 0)
        penalty = math.sqrt(pct_int)
        bup = total - penalty
        if bup > 0:
            ranked.append((ticker, bup))

    ranked.sort(key=lambda r: r[1], reverse=True)
    return ranked[:top_n]


# ── Alpaca HTTP helpers ──────────────────────────────────────────────

class AlpacaError(Exception):
    pass


def _request(method: str, base: str, path: str, key_id: str, secret: str,
             body: dict | None = None) -> dict | list | None:
    url = f"{base}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("APCA-API-KEY-ID", key_id)
    req.add_header("APCA-API-SECRET-KEY", secret)
    if data is not None:
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise AlpacaError(f"{method} {path} → HTTP {e.code}: {msg}") from e


def _find_watchlist(base: str, key_id: str, secret: str, name: str) -> dict | None:
    try:
        lists = _request("GET", base, "/v2/watchlists", key_id, secret)
    except AlpacaError:
        return None
    if not isinstance(lists, list):
        return None
    for wl in lists:
        if wl.get("name") == name:
            return wl
    return None


def _create_watchlist(base: str, key_id: str, secret: str,
                      name: str, symbols: list[str]) -> dict:
    return _request("POST", base, "/v2/watchlists", key_id, secret,
                    {"name": name, "symbols": symbols})  # type: ignore[return-value]


def _replace_symbols(base: str, key_id: str, secret: str,
                     wl_id: str, name: str, symbols: list[str]) -> dict:
    return _request("PUT", base, f"/v2/watchlists/{wl_id}", key_id, secret,
                    {"name": name, "symbols": symbols})  # type: ignore[return-value]


def _delete_watchlist(base: str, key_id: str, secret: str, wl_id: str) -> None:
    _request("DELETE", base, f"/v2/watchlists/{wl_id}", key_id, secret)


def _add_symbol(base: str, key_id: str, secret: str, wl_id: str, symbol: str) -> None:
    _request("POST", base, f"/v2/watchlists/{wl_id}", key_id, secret,
             {"symbol": symbol})


# ── Main ─────────────────────────────────────────────────────────────

def _write_watchlist_txt(tickers: list[str]) -> None:
    """Write the deduped tickers as a plain-text file for TradingView import."""
    path = os.path.join(OUTPUT_DIR, "watchlist.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tickers:
            f.write(t + "\n")
    print(f"  Wrote {len(tickers)} tickers to {path}")


def _update_history(today_top: list[tuple[str, float]]) -> dict:
    """Append today's top-N to history, trim to HISTORY_WINDOW_DAYS, return history dict."""
    today = datetime.now(timezone.utc).date().isoformat()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=HISTORY_WINDOW_DAYS - 1)).isoformat()

    history = _load_json(HISTORY_FILE) or {"days": []}
    days = [d for d in history.get("days", []) if d.get("date") and d["date"] >= cutoff and d["date"] != today]
    days.append({
        "date": today,
        "top": [{"ticker": t, "bup": round(b, 2)} for t, b in today_top],
    })
    days.sort(key=lambda d: d["date"])
    history["days"] = days

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def _combined_from_history(history: dict) -> list[dict]:
    """Build deduped per-ticker rollup from history days."""
    rollup: dict[str, dict] = {}
    for day in history.get("days", []):
        date = day["date"]
        for entry in day.get("top", []):
            t = entry["ticker"]
            b = entry["bup"]
            if t not in rollup:
                rollup[t] = {
                    "ticker": t,
                    "best_bup": b,
                    "best_bup_date": date,
                    "first_seen": date,
                    "last_seen": date,
                    "days_in_list": 1,
                }
            else:
                r = rollup[t]
                r["days_in_list"] += 1
                r["last_seen"] = max(r["last_seen"], date)
                r["first_seen"] = min(r["first_seen"], date)
                if b > r["best_bup"]:
                    r["best_bup"] = b
                    r["best_bup_date"] = date

    combined = list(rollup.values())
    combined.sort(key=lambda r: (-r["best_bup"], r["ticker"]))
    return combined


def _write_combined(combined: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": HISTORY_WINDOW_DAYS,
        "count": len(combined),
        "tickers": combined,
    }
    with open(COMBINED_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {len(combined)} unique tickers ({HISTORY_WINDOW_DAYS}d window) to {COMBINED_FILE}")


def push_watchlist(
    top_n: int = 100,
    name: str = DEFAULT_WATCHLIST_NAME,
    live: bool = False,
) -> None:
    today_top = _compute_bup_top(top_n)
    if not today_top:
        print("No BUP scorers found — nothing to push.")
        return

    # Roll today's top into the rolling 14-day history and build combined set.
    history = _update_history(today_top)
    combined = _combined_from_history(history)
    symbols = [r["ticker"] for r in combined]

    _write_watchlist_txt(symbols)
    _write_combined(combined)

    key_id = os.environ.get("ALPACA_API_KEY_ID", "")
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
    if not key_id or not secret:
        print("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set — skipping Alpaca push.")
        return

    base = os.environ.get("ALPACA_BASE_URL") or (LIVE_BASE if live else PAPER_BASE)

    print(f"\n── Alpaca watchlist push: {len(symbols)} unique tickers (last {HISTORY_WINDOW_DAYS}d) ──")
    print(f"  Endpoint: {base}")
    print(f"  Watchlist: {name!r}")
    print(f"  History spans {len(history['days'])} day(s), today added {len(today_top)} new scorers")
    for i, r in enumerate(combined[:10], 1):
        print(f"    {i:2d}. {r['ticker']:6s}  best BUP={r['best_bup']:.2f}  on {r['best_bup_date']}  ({r['days_in_list']}d)")
    if len(combined) > 10:
        print(f"    ... and {len(combined) - 10} more")

    existing = _find_watchlist(base, key_id, secret, name)

    # Try bulk update first. Alpaca rejects the whole request if any symbol
    # is invalid, so fall back to per-symbol add-on-failure.
    try:
        if existing:
            wl_id = existing["id"]
            _replace_symbols(base, key_id, secret, wl_id, name, symbols)
            print(f"  Replaced {len(symbols)} symbols in existing watchlist.")
        else:
            wl = _create_watchlist(base, key_id, secret, name, symbols)
            print(f"  Created new watchlist with {len(symbols)} symbols.")
        return
    except AlpacaError as e:
        print(f"  Bulk update failed: {e}")
        print(f"  Falling back to per-symbol add (skipping invalid tickers)...")

    # Fallback: wipe + recreate empty, then add one at a time, swallowing errors
    if existing:
        try:
            _delete_watchlist(base, key_id, secret, existing["id"])
        except AlpacaError as e:
            print(f"  Could not delete existing watchlist: {e}")
            return

    try:
        wl = _create_watchlist(base, key_id, secret, name, [])
    except AlpacaError as e:
        # Empty-symbol create sometimes 400s; try with the first symbol seeded.
        try:
            wl = _create_watchlist(base, key_id, secret, name, symbols[:1])
            symbols = symbols[1:]
        except AlpacaError as e2:
            print(f"  Could not create watchlist: {e2}")
            return

    wl_id = wl["id"]
    added, skipped = 0, []
    for sym in symbols:
        try:
            _add_symbol(base, key_id, secret, wl_id, sym)
            added += 1
        except AlpacaError:
            skipped.append(sym)
        time.sleep(0.05)  # 200 req/min Alpaca rate cap, stay polite

    print(f"  Added {added} symbols. Skipped (invalid/untradeable): {len(skipped)}")
    if skipped:
        print(f"    {', '.join(skipped[:30])}{' …' if len(skipped) > 30 else ''}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--name", type=str, default=DEFAULT_WATCHLIST_NAME)
    ap.add_argument("--live", action="store_true",
                    help="Push to live Alpaca account instead of paper (default: paper)")
    args = ap.parse_args()
    push_watchlist(top_n=args.top_n, name=args.name, live=args.live)
