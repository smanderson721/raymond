"""
SEC Form 4 insider-cluster scanner.

Pulls the EDGAR daily form-index for the last few business days, filters
to Form 4 filings (insider transactions), groups by issuer CIK, and
awards points when multiple insiders file at the same issuer within a
short window — that pattern ("cluster buy") historically precedes
outperformance.

Daily index format:
  https://www.sec.gov/Archives/edgar/daily-index/{YYYY}/QTR{N}/form.{YYYYMMDD}.idx
Each line:
  Form Type | Company Name | CIK | Date Filed | Filename

This scan does NOT determine buy vs sell — that requires parsing each
filing's XML. For minimal sieving v1 we just flag *any* clustering and
let downstream scans (e.g. analyst PT, news) confirm direction. A future
revision will fetch each filing's XML and split into buy/sell clusters.

We map CIK → ticker via the SEC's public ticker→CIK file
(https://www.sec.gov/files/company_tickers.json), cached in
data/live/_sec_tickers.json.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR

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
# vesting events). A true buy/sell breakdown requires parsing each
# filing's XML — left as a future enhancement.
PEAK_NOTABLE = 6        # ≥6 Form 4s on a single day = unusual
PEAK_HIT = 10           # ≥10 single-day = strong signal
PEAK_MEGA = 15          # ≥15 single-day = mega cluster


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
    """Extract Form 4 entries from a daily index file."""
    out = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("-----"):
            in_table = True
            continue
        if not in_table:
            continue
        # space-padded fixed-width-ish; columns separated by 2+ spaces is unreliable
        # because company names contain spaces. The first column is form type
        # padded to a fixed width; safest is to split on whitespace runs but
        # rejoin company name.
        # Heuristic: line starts with form type (≤12 chars), then company name
        # then CIK (digits), then date (yyyy-mm-dd), then filename.
        # We grab CIK + form via regex-style scan.
        parts = line.split()
        if not parts:
            continue
        form = parts[0]
        if form != "4":
            continue
        # find CIK token (last all-digit token before a date or filename)
        cik = None
        for tok in parts[1:]:
            if tok.isdigit():
                cik = int(tok)
                break
        if not cik:
            continue
        out.append({"form": form, "cik": cik})
    return out


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

        s.log(f"loaded {days_with_data}/{LOOKBACK_DAYS} daily form indexes", level="info")
        s.log(f"saw {sum(cik_total.values())} Form 4 filings across {len(cik_total)} issuers", level="info")

        awarded = 0
        for cik, n in cik_total.items():
            ticker = cik_map.get(cik)
            if not ticker:
                continue  # CIK not in our ticker map (private filer or non-equity)

            per_day = cik_per_day[cik]
            peak = max(per_day.values()) if per_day else 0
            active_days = len(per_day)

            # gate on peak single-day count, not 5d sum
            if peak < PEAK_NOTABLE:
                continue

            if peak >= PEAK_MEGA:
                pts, reason = 14.0, f"mega insider cluster: {peak} Form 4s in one day ({n} over {LOOKBACK_DAYS}d)"
            elif peak >= PEAK_HIT:
                pts, reason = 10.0, f"hot insider cluster: {peak} Form 4s in one day ({n} over {LOOKBACK_DAYS}d)"
            else:
                pts, reason = 5.0, f"insider cluster: {peak} Form 4s in one day ({n} over {LOOKBACK_DAYS}d)"

            # sustained-activity bonus
            if active_days >= 3 and peak >= PEAK_HIT:
                pts += 2.0
                reason += f", {active_days} active days"

            s.award(ticker, pts, reason)
            awarded += 1

        return {
            "ok": True,
            "days_with_data": days_with_data,
            "issuers_seen": len(cik_total),
            "filings_seen": sum(cik_total.values()),
            "awarded": awarded,
        }


if __name__ == "__main__":
    print(run())
