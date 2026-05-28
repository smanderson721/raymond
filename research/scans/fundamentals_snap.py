"""
Finnhub fundamentals snapshot (float, short interest, days-to-cover).

Hits Finnhub /stock/metric for the top-N current watchlist tickers
plus any ticker that's been hit recently. Free tier is 60 req/min, so
we cap at SCAN_TOP_N = 120 per run (≈2 min).

Cached values live in data/live/_fundamentals.json so we can detect
*changes* (e.g. short interest jumped 30%) on subsequent runs.

Signals:
  - short interest % float >= 20%                       -> squeeze candidate
  - short interest % float >= 30%                       -> hot squeeze
  - days to cover >= 7                                  -> sustained squeeze potential
  - SI jumped ≥30% vs prior snapshot (week-over-week)   -> shorts piling in
  - SI dropped ≥30%                                     -> covering wave
  - float < 20M shares with elevated SI                 -> low-float squeeze
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR

SCAN_NAME = "fundamentals_snap"
FINNHUB_BASE = "https://finnhub.io/api/v1"
WATCHLIST_FILE = Path(LIVE_DIR) / "watchlist.json"
CACHE_FILE = Path(LIVE_DIR) / "_fundamentals.json"

SCAN_TOP_N = 120                # top watchlist tickers to refresh per run
REQ_PER_MIN = 55                # below 60/min free-tier cap
SLEEP_BETWEEN = 60.0 / REQ_PER_MIN

# award thresholds
SI_PCT_NOTABLE = 20.0
SI_PCT_HOT = 30.0
SI_PCT_MEGA = 40.0
DTC_NOTABLE = 7.0
DTC_HOT = 12.0
LOW_FLOAT_THRESHOLD_M = 20.0    # float in millions
SI_CHANGE_THRESHOLD = 0.30      # ±30% week-over-week


def _load_watchlist_tickers(n: int) -> list[str]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        wl = json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return []
    out = []
    for r in wl.get("watchlist", [])[:n]:
        tk = r.get("ticker", "").upper()
        if not tk or any(c in tk for c in (".", "-", "/")):
            continue
        if len(tk) > 5:
            continue
        out.append(tk)
    return out


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {"updated_at": None, "tickers": {}}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(CACHE_FILE)


def _fetch_metric(ticker: str, api_key: str) -> dict | None:
    url = f"{FINNHUB_BASE}/stock/metric"
    try:
        r = requests.get(url, params={"symbol": ticker, "metric": "all", "token": api_key}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    m = data.get("metric") or {}
    # extract what we care about
    return {
        "shares_outstanding_m": m.get("shareOutstanding"),  # in millions
        "float_m": m.get("freeFloat") or m.get("shareFloat"),  # millions
        "short_interest_pct_float": m.get("shortInterestSharePercent"),  # %
        "short_interest_shares": m.get("shortInterest"),
        "days_to_cover": m.get("shortRatio"),
        "ttm_eps": m.get("epsTTM"),
        "beta": m.get("beta"),
    }


def _evaluate(ticker: str, snap: dict, prior: dict | None) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    si_pct = snap.get("short_interest_pct_float")
    dtc = snap.get("days_to_cover")
    flt = snap.get("float_m")

    # short interest % float
    if isinstance(si_pct, (int, float)):
        if si_pct >= SI_PCT_MEGA:
            out.append((12.0, f"mega short interest {si_pct:.1f}% of float"))
        elif si_pct >= SI_PCT_HOT:
            out.append((8.0, f"hot short interest {si_pct:.1f}% of float"))
        elif si_pct >= SI_PCT_NOTABLE:
            out.append((4.0, f"elevated short interest {si_pct:.1f}% of float"))

    # days to cover
    if isinstance(dtc, (int, float)):
        if dtc >= DTC_HOT:
            out.append((6.0, f"{dtc:.1f} days-to-cover (sustained squeeze potential)"))
        elif dtc >= DTC_NOTABLE:
            out.append((3.0, f"{dtc:.1f} days-to-cover"))

    # low float + SI combo
    if (isinstance(flt, (int, float)) and flt > 0 and flt < LOW_FLOAT_THRESHOLD_M
            and isinstance(si_pct, (int, float)) and si_pct >= SI_PCT_NOTABLE):
        out.append((6.0, f"low float {flt:.1f}M with {si_pct:.0f}% SI — squeeze setup"))

    # change vs prior snapshot
    if prior:
        prior_si = prior.get("short_interest_pct_float")
        if (isinstance(si_pct, (int, float)) and isinstance(prior_si, (int, float))
                and prior_si > 1.0):
            delta = (si_pct - prior_si) / prior_si
            if delta >= SI_CHANGE_THRESHOLD:
                out.append((5.0, f"short interest jumped {delta*100:+.0f}% vs prior ({prior_si:.1f}% → {si_pct:.1f}%)"))
            elif delta <= -SI_CHANGE_THRESHOLD:
                out.append((3.0, f"short interest collapsed {delta*100:+.0f}% (covering, {prior_si:.1f}% → {si_pct:.1f}%)"))

    return out


def run() -> dict:
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        with Session(SCAN_NAME, note="Finnhub fundamentals snapshot") as s:
            s.log("FINNHUB_API_KEY not set — skipping", level="info")
            return {"ok": False, "reason": "no_api_key"}

    tickers = _load_watchlist_tickers(SCAN_TOP_N)
    cache = _load_cache()

    with Session(SCAN_NAME, note="Finnhub fundamentals snapshot") as s:
        if not tickers:
            s.log("watchlist empty — nothing to refresh", level="info")
            return {"ok": True, "scanned": 0}

        s.log(f"refreshing fundamentals for {len(tickers)} tickers (~{int(len(tickers)*SLEEP_BETWEEN)}s)", level="info")
        scanned = 0
        awarded = 0
        failed = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for i, tk in enumerate(tickers):
            snap = _fetch_metric(tk, api_key)
            if not snap:
                failed += 1
                if i < len(tickers) - 1:
                    time.sleep(SLEEP_BETWEEN)
                continue
            scanned += 1
            prior = cache.get("tickers", {}).get(tk)
            for pts, reason in _evaluate(tk, snap, prior):
                s.award(tk, pts, reason)
                awarded += 1
            snap["updated_at"] = now_iso
            cache.setdefault("tickers", {})[tk] = snap
            if i < len(tickers) - 1:
                time.sleep(SLEEP_BETWEEN)

        cache["updated_at"] = now_iso
        _save_cache(cache)

        return {
            "ok": True,
            "scanned": scanned,
            "failed": failed,
            "awarded": awarded,
        }


if __name__ == "__main__":
    print(run())
