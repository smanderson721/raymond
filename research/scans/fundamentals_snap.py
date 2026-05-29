"""
Finnhub fundamentals snapshot — broad universe slicing scan.

This scan used to only refresh the top-N current watchlist, which meant
low-float / high-SI candidates could never be *discovered* — they could
only be confirmed once another scan had already promoted them onto the
watchlist. Float and short interest are some of the most actionable
fundamental signals, so we moved this scan upstream: it now rotates
through the entire ~5760-ticker precondition universe in slices of
~SLICE_SIZE per run, fully covering it every ~3.5 hours.

Finnhub free tier is 60 req/min; we cap at REQ_PER_MIN = 55. A 400-
ticker slice takes ~7.3 min wall-clock. The scheduler runs us every
15 min so cycle = (5760 / 400) × 15 min ≈ 3.6 hr.

Rotation cursor is persisted at data/live/_fundamentals_cursor.json,
matching the tech_slice pattern.

Signals (each an independent attribute the engine can tag):
  - low_float        — float in [10M, 50M]                 → upstream-valuable
  - micro_float      — float < 10M                          → extreme float crunch
  - mega_short_interest / hot / elevated                     → squeeze fuel
  - days_to_cover_hot / notable                              → sustained squeeze potential
  - si_jumped / si_covering_wave (vs prior snapshot)        → flow direction
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR, _read, _write
from research.scan_weights import weight

SCAN_NAME = "fundamentals_snap"
FINNHUB_BASE = "https://finnhub.io/api/v1"
CACHE_FILE = Path(LIVE_DIR) / "_fundamentals.json"
CURSOR_FILE = os.path.join(LIVE_DIR, "_fundamentals_cursor.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN_FILE = os.path.join(REPO_ROOT, "research_output", "scan_results.json")

# ── tunables ─────────────────────────────────────────────────────────
SLICE_SIZE = 400                 # tickers processed per scheduled run
REQ_PER_MIN = 55                 # below 60/min Finnhub free-tier cap
SLEEP_BETWEEN = 60.0 / REQ_PER_MIN

# Float bands (in millions of shares)
MICRO_FLOAT_MAX_M = 10.0
LOW_FLOAT_MIN_M = 10.0
LOW_FLOAT_MAX_M = 50.0

# Short-interest tiers (% of float)
SI_PCT_NOTABLE = 20.0
SI_PCT_HOT = 30.0
SI_PCT_MEGA = 40.0

# Days-to-cover tiers
DTC_NOTABLE = 7.0
DTC_HOT = 12.0

# Snapshot-to-snapshot change thresholds (decimal, e.g. 0.30 = ±30%)
SI_CHANGE_THRESHOLD = 0.30

# ── default award points (weight() will override from scan_weights.json) ──
DEFAULT_LOW_FLOAT = 10.0
DEFAULT_MICRO_FLOAT = 12.0
DEFAULT_MEGA_SI = 12.0
DEFAULT_HOT_SI = 8.0
DEFAULT_ELEVATED_SI = 4.0
DEFAULT_DTC_HOT = 6.0
DEFAULT_DTC_NOTABLE = 3.0
DEFAULT_SI_JUMPED = 5.0
DEFAULT_SI_COVERING = 3.0


# ── universe + cursor (same pattern as tech_slice) ──────────────────

def _load_universe() -> list[str]:
    if not os.path.exists(SCAN_FILE):
        return []
    try:
        with open(SCAN_FILE) as f:
            data = json.load(f)
    except Exception:
        return []
    scans = data.get("scans") or []
    if not scans:
        return []
    latest = scans[-1].get("stocks") or []
    out = []
    for s in latest:
        tk = s.get("ticker") or s.get("symbol")
        if not tk:
            continue
        tk = tk.upper()
        # skip non-equity symbols Finnhub doesn't index well
        if any(c in tk for c in (".", "/")):
            continue
        if len(tk) > 5:
            continue
        out.append(tk)
    return out


def _next_slice(universe: list[str]) -> tuple[list[str], int, int]:
    """Return (slice_tickers, start_index, end_index) and advance cursor."""
    cur = _read(CURSOR_FILE, {"cursor": 0, "universe_size": 0})
    n = len(universe)
    cursor = cur.get("cursor", 0)
    if cursor >= n:
        cursor = 0
    end = cursor + SLICE_SIZE
    if end <= n:
        sl = universe[cursor:end]
    else:
        sl = universe[cursor:] + universe[: end - n]
    next_cursor = end % n if n else 0
    _write(CURSOR_FILE, {"cursor": next_cursor, "universe_size": n,
                          "last_slice": [cursor, end]})
    return sl, cursor, end


# ── cache (for snapshot-over-snapshot SI deltas) ─────────────────────

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


# ── Finnhub fetch ────────────────────────────────────────────────────

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
    return {
        "shares_outstanding_m": m.get("shareOutstanding"),  # millions
        "float_m": m.get("freeFloat") or m.get("shareFloat"),  # millions
        "short_interest_pct_float": m.get("shortInterestSharePercent"),  # %
        "short_interest_shares": m.get("shortInterest"),
        "days_to_cover": m.get("shortRatio"),
        "ttm_eps": m.get("epsTTM"),
        "beta": m.get("beta"),
    }


# ── evaluation ───────────────────────────────────────────────────────

def _evaluate(ticker: str, snap: dict, prior: dict | None) -> list[tuple[float, str, str]]:
    """Return list of (points, reason, attr_key) tuples for each detected
    fundamental attribute. Each tuple is awarded as an independent event
    so the engine tags the ticker with one chip per attribute."""
    out: list[tuple[float, str, str]] = []
    si_pct = snap.get("short_interest_pct_float")
    dtc = snap.get("days_to_cover")
    flt = snap.get("float_m")

    # ── float (always an independent attribute, no SI dependency) ──
    if isinstance(flt, (int, float)) and flt > 0:
        if flt < MICRO_FLOAT_MAX_M:
            pts = weight(SCAN_NAME, "micro_float", DEFAULT_MICRO_FLOAT)
            out.append((pts, f"micro float {flt:.1f}M shares", "micro_float"))
        elif LOW_FLOAT_MIN_M <= flt <= LOW_FLOAT_MAX_M:
            pts = weight(SCAN_NAME, "low_float", DEFAULT_LOW_FLOAT)
            out.append((pts, f"low float {flt:.1f}M shares (10M–50M band)", "low_float"))

    # ── short interest % float (independent of float) ──
    if isinstance(si_pct, (int, float)):
        if si_pct >= SI_PCT_MEGA:
            pts = weight(SCAN_NAME, "mega_short_interest", DEFAULT_MEGA_SI)
            out.append((pts, f"mega short interest {si_pct:.1f}% of float", "mega_short_interest"))
        elif si_pct >= SI_PCT_HOT:
            pts = weight(SCAN_NAME, "hot_short_interest", DEFAULT_HOT_SI)
            out.append((pts, f"hot short interest {si_pct:.1f}% of float", "hot_short_interest"))
        elif si_pct >= SI_PCT_NOTABLE:
            pts = weight(SCAN_NAME, "elevated_short_interest", DEFAULT_ELEVATED_SI)
            out.append((pts, f"elevated short interest {si_pct:.1f}% of float", "elevated_short_interest"))

    # ── days to cover ──
    if isinstance(dtc, (int, float)):
        if dtc >= DTC_HOT:
            pts = weight(SCAN_NAME, "days_to_cover_hot", DEFAULT_DTC_HOT)
            out.append((pts, f"{dtc:.1f} days-to-cover (sustained squeeze potential)", "days_to_cover_hot"))
        elif dtc >= DTC_NOTABLE:
            pts = weight(SCAN_NAME, "days_to_cover_notable", DEFAULT_DTC_NOTABLE)
            out.append((pts, f"{dtc:.1f} days-to-cover", "days_to_cover_notable"))

    # ── change vs prior snapshot ──
    if prior:
        prior_si = prior.get("short_interest_pct_float")
        if (isinstance(si_pct, (int, float)) and isinstance(prior_si, (int, float))
                and prior_si > 1.0):
            delta = (si_pct - prior_si) / prior_si
            if delta >= SI_CHANGE_THRESHOLD:
                pts = weight(SCAN_NAME, "si_jumped", DEFAULT_SI_JUMPED)
                out.append((pts, f"short interest jumped {delta*100:+.0f}% vs prior ({prior_si:.1f}% → {si_pct:.1f}%)", "si_jumped"))
            elif delta <= -SI_CHANGE_THRESHOLD:
                pts = weight(SCAN_NAME, "si_covering_wave", DEFAULT_SI_COVERING)
                out.append((pts, f"short interest collapsed {delta*100:+.0f}% (covering, {prior_si:.1f}% → {si_pct:.1f}%)", "si_covering_wave"))

    return out


# ── runner ───────────────────────────────────────────────────────────

def run() -> dict:
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        with Session(SCAN_NAME, note="Finnhub fundamentals snapshot") as s:
            s.log("FINNHUB_API_KEY not set — skipping", level="info")
            return {"ok": False, "reason": "no_api_key"}

    universe = _load_universe()
    if not universe:
        with Session(SCAN_NAME, note="Finnhub fundamentals snapshot") as s:
            s.log("no precondition universe found at research_output/scan_results.json — "
                  "run --stocks-scan first", level="notable")
            return {"ok": False, "reason": "no_universe"}

    slice_, start, end = _next_slice(universe)
    cache = _load_cache()

    with Session(SCAN_NAME,
                 note=f"slice {start}..{end} of {len(universe)}") as s:
        s.log(f"refreshing fundamentals for slice {start}→{end} of {len(universe)} "
              f"(~{int(len(slice_)*SLEEP_BETWEEN)}s)", level="info")

        scanned = 0
        awarded = 0
        failed = 0
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for i, tk in enumerate(slice_):
            snap = _fetch_metric(tk, api_key)
            if not snap:
                failed += 1
                if i < len(slice_) - 1:
                    time.sleep(SLEEP_BETWEEN)
                continue
            scanned += 1
            prior = cache.get("tickers", {}).get(tk)
            for pts, reason, attr_key in _evaluate(tk, snap, prior):
                s.award(tk, pts, reason, attr_key=attr_key)
                awarded += 1
            snap["updated_at"] = now_iso
            cache.setdefault("tickers", {})[tk] = snap
            if i < len(slice_) - 1:
                time.sleep(SLEEP_BETWEEN)

        cache["updated_at"] = now_iso
        _save_cache(cache)

        s.log(f"slice done — scanned={scanned} failed={failed} awarded={awarded}")

        return {
            "ok": True,
            "scanned": scanned,
            "failed": failed,
            "awarded": awarded,
            "slice_start": start,
            "slice_end": end,
        }


if __name__ == "__main__":
    print(run())
