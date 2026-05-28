"""
SEC Form 4 insider-cluster scanner.

Pulls the EDGAR daily form-index for the last few business days, filters
to Form 4 filings (insider transactions), groups by issuer CIK, and
awards points when multiple insiders file at the same issuer within a
short window.

Daily index format:
  https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{N}/form.{YYYYMMDD}.idx
Each line:
  Form Type | Company Name | CIK | Date Filed | Filename

For each issuer that clears the cluster gate (peak single-day filings
>= PEAK_NOTABLE), we now fetch every filing's full-submission .txt and
parse its embedded <ownershipDocument> XML via insider_xml.classify_form4.
Filings classify as buy / sell / noise. We then bucket the issuer:
  - BUY CLUSTER  (n_buy >= 3 and n_buy > n_sell)  → boosted award
  - SELL CLUSTER (n_sell >= 3 and n_sell > n_buy) → reduced award
  - NOISE        (mostly comp / RSU / option vest) → skipped
This collapses the false-positive rate from routine vesting events that
was the main weakness of the v1 “any clustering” signal.

We map CIK → ticker via the SEC's public ticker→CIK file
(https://www.sec.gov/files/company_tickers.json), cached in
data/live/_sec_tickers.json.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR
from research.scans.insider_xml import classify_form4

SCAN_NAME = "insider_cluster"
# SEC fair-access requires a contact email in the User-Agent + proper Accept headers
SEC_HEADERS = {
    "User-Agent": "Raymond Research smanderson721@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/plain, */*",
    "Host": "www.sec.gov",
}

DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/form.{date}.idx"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

LOOKBACK_DAYS = 5
TICKERS_CACHE = Path(LIVE_DIR) / "_sec_tickers.json"
TICKERS_CACHE_DAYS = 7

# award thresholds — weighted toward peak single-day filings rather than
# 5-day sums (the latter over-fires for any large issuer with routine RSU
# vesting events). XML classification (below) further filters comp noise.
PEAK_NOTABLE = 6        # ≥6 Form 4s on a single day = unusual
PEAK_HIT = 10           # ≥10 single-day = strong signal
PEAK_MEGA = 15          # ≥15 single-day = mega cluster

# Buy/sell classification thresholds (after fetching each filing's XML).
# A cluster needs at least this many *real* buys (or sells) to count.
BUY_MIN = 3
SELL_MIN = 3
# polite gap between SEC submission-file fetches (10 req/s ceiling).
FETCH_GAP_SEC = 0.12
# safety cap on how many filings we'll classify per scan run
MAX_CLASSIFY_TOTAL = 600


def _http_get(url: str, timeout: int = 30) -> requests.Response | None:
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


def _load_cik_to_ticker() -> dict[int, str]:
    """Build CIK -> ticker map, using a 7-day disk cache."""
    if TICKERS_CACHE.exists():
        age_days = (datetime.now(timezone.utc).timestamp() - TICKERS_CACHE.stat().st_mtime) / 86400
        if age_days < TICKERS_CACHE_DAYS:
            try:
                cached = json.loads(TICKERS_CACHE.read_text())
                return {int(k): v for k, v in cached.items()}
            except Exception:
                pass
    r = _http_get(TICKERS_URL)
    if not r:
        return {}
    try:
        data = r.json()
    except Exception:
        return {}
    # format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
    mapping: dict[int, str] = {}
    for v in data.values():
        cik = int(v.get("cik_str", 0))
        tk = v.get("ticker", "").upper()
        if cik and tk:
            mapping[cik] = tk
    TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TICKERS_CACHE.write_text(json.dumps({str(k): v for k, v in mapping.items()}))
    return mapping


def _business_days(end_date, n: int) -> list:
    """Return list of last n business days ending at end_date (inclusive)."""
    out = []
    d = end_date
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def _fetch_form_index(date) -> str | None:
    qtr = (date.month - 1) // 3 + 1
    url = DAILY_INDEX_URL.format(year=date.year, qtr=qtr, date=date.strftime("%Y%m%d"))
    r = _http_get(url)
    return r.text if r else None


def _parse_form4_lines(text: str) -> list[dict]:
    """Extract Form 4 entries from a daily index file.

    Returns dicts with form, cik, and filename (relative to
    https://www.sec.gov/Archives/). The filename is the last
    whitespace-separated token on each line.
    """
    out = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("-----"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if not parts:
            continue
        form = parts[0]
        if form != "4":
            continue
        cik = None
        for tok in parts[1:]:
            if tok.isdigit():
                cik = int(tok)
                break
        if not cik:
            continue
        # filename is the trailing token (e.g. edgar/data/.../accno.txt)
        filename = parts[-1]
        if not filename.startswith("edgar/"):
            filename = None
        out.append({"form": form, "cik": cik, "filename": filename})
    return out


def _fetch_submission_text(filename: str) -> str | None:
    """Fetch the full-submission .txt for a Form 4 filing."""
    if not filename:
        return None
    url = "https://www.sec.gov/Archives/" + filename
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return r.text


def _classify_issuer_filings(filings: list[str], budget_remaining: int) -> dict:
    """Fetch + classify every filing for one issuer. Returns aggregate buckets."""
    agg = {
        "buy": 0, "sell": 0, "noise": 0,
        "buy_dollars": 0.0, "sell_dollars": 0.0,
        "fetched": 0,
    }
    for fn in filings:
        if agg["fetched"] >= budget_remaining:
            break
        txt = _fetch_submission_text(fn)
        time.sleep(FETCH_GAP_SEC)
        if not txt:
            continue
        agg["fetched"] += 1
        c = classify_form4(txt)
        agg[c["class"]] += 1
        agg["buy_dollars"]  += c["buy_dollars"]
        agg["sell_dollars"] += c["sell_dollars"]
    return agg


def run() -> dict:
    cik_map = _load_cik_to_ticker()
    if not cik_map:
        with Session(SCAN_NAME, note="Form 4 insider-cluster scan") as s:
            s.log("failed to load SEC ticker→CIK map", level="info")
            return {"ok": False, "reason": "no_ticker_map"}

    with Session(SCAN_NAME, note="Form 4 insider-cluster scan") as s:
        today = datetime.now(timezone.utc).date()
        days = _business_days(today - timedelta(days=1), LOOKBACK_DAYS)
        # build counts per CIK across all days, plus per-day for spike detection
        cik_total: dict[int, int] = defaultdict(int)
        cik_per_day: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        cik_filings: dict[int, list[str]] = defaultdict(list)
        days_with_data = 0
        for d in days:
            txt = _fetch_form_index(d)
            if not txt:
                continue
            days_with_data += 1
            entries = _parse_form4_lines(txt)
            for e in entries:
                cik_total[e["cik"]] += 1
                cik_per_day[e["cik"]][d.isoformat()] += 1
                if e["filename"]:
                    cik_filings[e["cik"]].append(e["filename"])

        s.log(f"loaded {days_with_data}/{LOOKBACK_DAYS} daily form indexes", level="info")
        s.log(f"saw {sum(cik_total.values())} Form 4 filings across {len(cik_total)} issuers", level="info")

        awarded = 0
        buy_clusters = 0
        sell_clusters = 0
        noise_filtered = 0
        classify_budget = MAX_CLASSIFY_TOTAL

        # iterate issuers in peak-first order so the budget is spent on the
        # most signal-rich clusters
        ranked = sorted(
            cik_total.items(),
            key=lambda kv: max(cik_per_day[kv[0]].values()) if cik_per_day[kv[0]] else 0,
            reverse=True,
        )

        for cik, n in ranked:
            ticker = cik_map.get(cik)
            if not ticker:
                continue  # CIK not in our ticker map (private filer or non-equity)

            per_day = cik_per_day[cik]
            peak = max(per_day.values()) if per_day else 0
            active_days = len(per_day)

            # gate on peak single-day count, not 5d sum
            if peak < PEAK_NOTABLE:
                continue

            # ── XML classification: fetch each filing, count buys vs sells ──
            filings = cik_filings.get(cik, [])
            agg = _classify_issuer_filings(filings, classify_budget)
            classify_budget -= agg["fetched"]
            n_buy, n_sell, n_noise = agg["buy"], agg["sell"], agg["noise"]

            is_buy_cluster  = (n_buy  >= BUY_MIN  and n_buy  > n_sell)
            is_sell_cluster = (n_sell >= SELL_MIN and n_sell > n_buy)

            if not (is_buy_cluster or is_sell_cluster):
                noise_filtered += 1
                continue

            # Buy clusters get the original (boosted) scale; sell clusters
            # are reported but at lower magnitude so they don't crowd out
            # bullish setups on the watchlist.
            if is_buy_cluster:
                buy_clusters += 1
                if peak >= PEAK_MEGA:
                    pts, label = 16.0, f"mega BUY cluster: {n_buy} buys / {n_sell} sells, peak {peak}/day"
                elif peak >= PEAK_HIT:
                    pts, label = 12.0, f"hot BUY cluster: {n_buy} buys / {n_sell} sells, peak {peak}/day"
                else:
                    pts, label = 7.0, f"BUY cluster: {n_buy} buys / {n_sell} sells, peak {peak}/day"
                if active_days >= 3 and peak >= PEAK_HIT:
                    pts += 2.0
                    label += f", {active_days} active days"
                if agg["buy_dollars"] >= 1_000_000:
                    label += f", ${agg['buy_dollars']/1e6:.1f}M bought"
            else:
                sell_clusters += 1
                if peak >= PEAK_MEGA:
                    pts, label = 6.0, f"mega SELL cluster: {n_sell} sells / {n_buy} buys, peak {peak}/day"
                elif peak >= PEAK_HIT:
                    pts, label = 4.0, f"hot SELL cluster: {n_sell} sells / {n_buy} buys, peak {peak}/day"
                else:
                    pts, label = 2.0, f"SELL cluster: {n_sell} sells / {n_buy} buys, peak {peak}/day"
                if agg["sell_dollars"] >= 1_000_000:
                    label += f", ${agg['sell_dollars']/1e6:.1f}M sold"

            s.award(ticker, pts, label)
            awarded += 1

            if classify_budget <= 0:
                s.log(f"XML classify budget exhausted after {awarded} awards", level="info")
                break

        return {
            "ok": True,
            "days_with_data": days_with_data,
            "issuers_seen": len(cik_total),
            "filings_seen": sum(cik_total.values()),
            "awarded": awarded,
            "buy_clusters": buy_clusters,
            "sell_clusters": sell_clusters,
            "noise_filtered": noise_filtered,
        }


if __name__ == "__main__":
    print(run())
