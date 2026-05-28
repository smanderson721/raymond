"""
FINRA Reg SHO daily short-volume scanner.

Pulls the daily FINRA Consolidated NMS short-sale volume file and awards
points to tickers showing extreme short-volume ratios or large dark-pool /
exchange short prints. Runs after the close (T+1 data — FINRA publishes
the file the next morning around 06:00 ET).

File: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

Each scan walks back up to MAX_LOOKBACK_DAYS until it finds a published
file (handles weekends / holidays), then classifies every ticker in our
universe that appears in the file.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR

SCAN_NAME = "reg_sho"
FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
MAX_LOOKBACK_DAYS = 6
USER_AGENT = "raymond-live-scan (smanderson721)"

# award thresholds
EXTREME_RATIO = 0.65       # ≥65% short volume — likely squeeze setup / heavy bearish flow
ELEVATED_RATIO = 0.55      # ≥55% short volume — bearish lean
LOW_RATIO = 0.20           # ≤20% short volume — heavy buying pressure
MIN_VOLUME_EXTREME = 2_000_000
MIN_VOLUME_ELEVATED = 1_000_000
MIN_VOLUME_LOW = 1_000_000

UNIVERSE_FILE = Path("research_output/scan_results.json")
HISTORY_FILE = Path(LIVE_DIR) / "_reg_sho_history.json"   # rolling per-ticker ratio cache
HISTORY_DAYS = 20
MAX_HISTORY_TICKERS = 10_000


def _load_universe() -> set[str]:
    if not UNIVERSE_FILE.exists():
        return set()
    try:
        data = json.loads(UNIVERSE_FILE.read_text())
        scans = data.get("scans") or []
        if not scans:
            return set()
        return {s["ticker"].upper() for s in scans[-1].get("stocks", []) if s.get("ticker")}
    except Exception:
        return set()


def _fetch_latest_file() -> tuple[str | None, str | None]:
    """Walk back from today until we find a published file. Returns (date, text)."""
    today = datetime.now(timezone.utc).date()
    for offset in range(MAX_LOOKBACK_DAYS):
        d = today - timedelta(days=offset)
        # skip Sat/Sun
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        url = FINRA_URL.format(date=date_str)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        except Exception:
            continue
        if r.status_code == 200 and len(r.text) > 1000 and "Symbol" in r.text[:200]:
            return date_str, r.text
    return None, None


def _parse_file(text: str) -> dict[str, dict]:
    """Parse FINRA pipe-delimited file → {ticker: {short, exempt, total, ratio}}."""
    rows: dict[str, dict] = {}
    lines = text.splitlines()
    if not lines:
        return rows
    # header: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue
        try:
            sym = parts[1].strip().upper()
            # FINRA publishes decimal values now (e.g. "601216.066172")
            short = int(float(parts[2] or 0))
            exempt = int(float(parts[3] or 0))
            total = int(float(parts[4] or 0))
        except (ValueError, IndexError):
            continue
        if not sym or total <= 0:
            continue
        # FINRA file has one row per (symbol, market) pair — aggregate
        cur = rows.setdefault(sym, {"short": 0, "exempt": 0, "total": 0})
        cur["short"] += short
        cur["exempt"] += exempt
        cur["total"] += total
    # finalize ratio
    for sym, r in rows.items():
        r["ratio"] = (r["short"] / r["total"]) if r["total"] else 0.0
    return rows


def _load_history() -> dict:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return {"days": [], "tickers": {}}


def _save_history(hist: dict, date_str: str, parsed: dict[str, dict]) -> None:
    # update day list
    days = hist.get("days", [])
    if date_str not in days:
        days = (days + [date_str])[-HISTORY_DAYS:]
    # update per-ticker ratio series (only tickers in the new file or already tracked)
    tickers = hist.get("tickers", {})
    for sym, r in parsed.items():
        series = tickers.setdefault(sym, {"days": [], "ratios": [], "volumes": []})
        series["days"].append(date_str)
        series["ratios"].append(round(r["ratio"], 4))
        series["volumes"].append(int(r["total"]))
        # trim
        if len(series["days"]) > HISTORY_DAYS:
            series["days"] = series["days"][-HISTORY_DAYS:]
            series["ratios"] = series["ratios"][-HISTORY_DAYS:]
            series["volumes"] = series["volumes"][-HISTORY_DAYS:]
    # bound total tickers tracked (drop oldest-touched if huge)
    if len(tickers) > MAX_HISTORY_TICKERS:
        # keep tickers whose most-recent day is the freshest
        ranked = sorted(tickers.items(), key=lambda kv: kv[1]["days"][-1] if kv[1]["days"] else "", reverse=True)
        tickers = dict(ranked[:MAX_HISTORY_TICKERS])
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"days": days, "tickers": tickers, "updated_at": date_str}))
    tmp.replace(HISTORY_FILE)


def _classify(sym: str, r: dict, hist_series: dict | None) -> list[tuple[float, str]]:
    """Return list of (points, reason) for this ticker."""
    out: list[tuple[float, str]] = []
    ratio = r["ratio"]
    total = r["total"]
    short = r["short"]

    # baseline (median of stored ratios — history is updated *after* classify
    # so this naturally excludes today)
    baseline = None
    if hist_series and len(hist_series.get("ratios", [])) >= 5:
        prev = sorted(hist_series["ratios"])
        baseline = prev[len(prev) // 2]

    pct = ratio * 100

    if ratio >= EXTREME_RATIO and total >= MIN_VOLUME_EXTREME:
        out.append((10.0, f"extreme short ratio {pct:.0f}% on {total/1e6:.1f}M vol"))
    elif ratio >= ELEVATED_RATIO and total >= MIN_VOLUME_ELEVATED:
        out.append((5.0, f"elevated short ratio {pct:.0f}% on {total/1e6:.1f}M vol"))
    elif ratio <= LOW_RATIO and total >= MIN_VOLUME_LOW:
        out.append((4.0, f"heavy buying — only {pct:.0f}% short on {total/1e6:.1f}M vol"))

    # ratio spike vs baseline (regardless of absolute level)
    if baseline is not None and total >= 500_000:
        if ratio - baseline >= 0.18:
            out.append((6.0, f"short ratio jump +{(ratio-baseline)*100:.0f}pp vs {HISTORY_DAYS}d baseline ({baseline*100:.0f}%)"))
        elif baseline - ratio >= 0.18 and ratio <= 0.35:
            out.append((3.0, f"short ratio collapse −{(baseline-ratio)*100:.0f}pp (buyers stepped in)"))

    # short volume notional milestone (huge short prints)
    if short >= 20_000_000 and ratio >= 0.40:
        out.append((4.0, f"{short/1e6:.0f}M short shares printed"))

    return out


def run() -> dict:
    universe = _load_universe()
    with Session(SCAN_NAME, note="FINRA Reg SHO daily short-volume scan") as s:
        date_str, text = _fetch_latest_file()
        if not text:
            s.log("no FINRA Reg SHO file found in last 6 days", level="info")
            return {"ok": False, "reason": "no_file"}

        s.log(f"loaded Reg SHO file for {date_str}", level="info")
        parsed = _parse_file(text)
        s.log(f"parsed {len(parsed)} symbols", level="info")

        hist = _load_history()
        # only score tickers that are in our universe
        scored = 0
        for sym in universe:
            r = parsed.get(sym)
            if not r:
                continue
            series = hist.get("tickers", {}).get(sym)
            # update history *before* classify so baseline excludes today via slicing
            for pts, reason in _classify(sym, r, series):
                if pts >= 0:
                    s.award(sym, pts, reason)
                else:
                    s.award(sym, pts, reason)
            scored += 1

        # persist history (use parsed, not just universe, so future baselines are accurate)
        _save_history(hist, date_str, parsed)

        return {
            "ok": True,
            "file_date": date_str,
            "rows_in_file": len(parsed),
            "universe_scored": scored,
        }


if __name__ == "__main__":
    print(run())
