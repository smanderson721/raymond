"""
Macro economic series + release calendar (FRED).

Pulls key macro series from FRED (Federal Reserve Economic Data) and:
  - records levels in data/live/macro_state.json (alongside macro_regime)
  - logs notable changes since the prior observation (rate cut/hike,
    NFP surprise, inflation print, etc.) into the live feed

Series tracked:
  - DFF        — Effective Federal Funds Rate (daily)
  - DGS10      — 10-Year Treasury Constant Maturity (daily)
  - DGS2       — 2-Year Treasury Constant Maturity (daily)
  - DTWEXBGS   — Trade-Weighted USD Index (daily)
  - VIXCLS     — VIX (daily, already in macro_regime but useful for delta)
  - CPIAUCSL   — Consumer Price Index (monthly)
  - UNRATE     — Unemployment Rate (monthly)
  - PAYEMS     — Total Nonfarm Payrolls (monthly)
  - GDPC1      — Real GDP (quarterly)

We don't replace macro_regime — this scan augments it with macro
data releases (which fire on event, not on a 15-min cadence).
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR

SCAN_NAME = "macro_econ"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
CACHE_FILE = Path(LIVE_DIR) / "_macro_econ.json"

SERIES = {
    "DFF":        ("Fed Funds Rate (effective)",       "%",     0.10),
    "DGS10":      ("10Y Treasury Yield",               "%",     0.10),
    "DGS2":       ("2Y Treasury Yield",                "%",     0.10),
    "DTWEXBGS":   ("Trade-Weighted USD Index",         "",      0.50),
    "VIXCLS":     ("VIX (FRED snapshot)",              "",      2.00),
    "CPIAUCSL":   ("CPI (all items)",                  "index", 0.30),
    "UNRATE":     ("Unemployment Rate",                "%",     0.10),
    "PAYEMS":     ("Nonfarm Payrolls",                 "k",    50.00),
    "GDPC1":      ("Real GDP",                         "B$",   25.00),
}


def _fetch_series(sid: str, api_key: str) -> list[dict] | None:
    try:
        r = requests.get(FRED_BASE, params={
            "series_id": sid, "api_key": api_key, "file_type": "json",
            "sort_order": "desc", "limit": 6,
        }, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        obs = data.get("observations") or []
        # filter out '.' (FRED missing data marker)
        return [o for o in obs if o.get("value") not in (None, "", ".")]
    except Exception:
        return None


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_FILE)


def run() -> dict:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        with Session(SCAN_NAME, note="FRED macro series + releases") as s:
            s.log("FRED_API_KEY not set — skipping", level="info")
            return {"ok": False, "reason": "no_api_key"}

    cache = _load_cache()
    new_cache: dict = {}
    new_releases = 0
    big_moves = 0

    with Session(SCAN_NAME, note="FRED macro series + releases") as s:
        for sid, (label, unit, threshold) in SERIES.items():
            obs = _fetch_series(sid, api_key)
            if not obs:
                s.log(f"{sid}: fetch failed", level="info")
                continue
            latest = obs[0]
            try:
                latest_val = float(latest["value"])
            except (ValueError, TypeError):
                continue
            latest_date = latest["date"]
            new_cache[sid] = {"value": latest_val, "date": latest_date}

            prev_cached = cache.get(sid) or {}
            prev_date = prev_cached.get("date")
            prev_val = prev_cached.get("value")

            # new observation released since last scan?
            if prev_date and prev_date != latest_date:
                new_releases += 1
                delta = latest_val - (prev_val if prev_val is not None else latest_val)
                s.log(
                    f"📅 NEW RELEASE {sid} ({label}) {latest_date}: {latest_val:.2f}{unit} (Δ {delta:+.2f})",
                    level="notable",
                )
                if abs(delta) >= threshold:
                    big_moves += 1
                    direction = "spike" if delta > 0 else "drop"
                    s.log(
                        f"🚨 MACRO {direction.upper()}: {label} moved {delta:+.2f}{unit} "
                        f"({prev_val:.2f} → {latest_val:.2f})",
                        level="hit",
                    )
            elif not prev_date:
                # first time we see this series
                s.log(f"{sid} {latest_date}: {latest_val:.2f}{unit}", level="info")
            # else: same observation as last scan — no log

        _save_cache(new_cache)

        return {
            "ok": True,
            "series_tracked": len(new_cache),
            "new_releases": new_releases,
            "big_moves": big_moves,
        }


if __name__ == "__main__":
    print(run())
