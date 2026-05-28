"""
EDGAR 8-K material-event scanner.

8-Ks are filed only on material corporate events (M&A, leadership change,
guidance revision, earnings preannouncement, bankruptcy, etc.). Any
unscheduled 8-K is signal-worthy. This scan pulls the EDGAR daily form
index and awards points to issuers with one or more 8-K filings in the
lookback window.

We don't parse the 8-K body in v1 — that requires fetching the index
URL for each filing and parsing the Item code from the document text.
Future revision will tag items 1.01 (material agreement), 2.01
(completion of M&A), 5.02 (officer change), 8.01 (other material).
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Reuse the helpers built for insider_cluster
from research.scans.insider_cluster import (
    _http_get, _load_cik_to_ticker, _business_days, _fetch_form_index,
)
from research.live_score_engine import Session
from research.scan_weights import weight

SCAN_NAME = "material_8k"
LOOKBACK_DAYS = 3   # short window — 8-Ks are rare-ish per issuer

# Default reward thresholds (overridable via data/live/scan_weights.json,
# editable from the dashboard Attributes editor).
DEFAULT_SINGLE_8K = 5.0           # one 8-K in window — material event
DEFAULT_MULTIPLE_8K = 10.0        # 2+ 8-Ks in window — sustained material flow
DEFAULT_TRIPLE_8K = 14.0          # 3+ 8-Ks — very active issuer (likely crisis or deal)


def _parse_form_index_for_8k(text: str) -> list[int]:
    """Return CIKs that filed a Form 8-K (or 8-K/A) in this daily index."""
    out: list[int] = []
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
        # Match 8-K, 8-K/A, 8-K12B, 8-K12G3 etc. — anything starting with "8-K"
        if not form.startswith("8-K"):
            continue
        cik = None
        for tok in parts[1:]:
            if tok.isdigit():
                cik = int(tok)
                break
        if cik:
            out.append(cik)
    return out


def run() -> dict:
    cik_map = _load_cik_to_ticker()
    if not cik_map:
        with Session(SCAN_NAME, note="EDGAR 8-K material event scan") as s:
            s.log("failed to load SEC ticker→CIK map", level="info")
            return {"ok": False, "reason": "no_ticker_map"}

    with Session(SCAN_NAME, note="EDGAR 8-K material event scan") as s:
        today = datetime.now(timezone.utc).date()
        days = _business_days(today - timedelta(days=1), LOOKBACK_DAYS)
        cik_total: dict[int, int] = defaultdict(int)
        cik_per_day: dict[int, list[str]] = defaultdict(list)
        days_with_data = 0
        for d in days:
            txt = _fetch_form_index(d)
            if not txt:
                continue
            days_with_data += 1
            ciks = _parse_form_index_for_8k(txt)
            for cik in ciks:
                cik_total[cik] += 1
                cik_per_day[cik].append(d.isoformat())

        s.log(f"loaded {days_with_data}/{LOOKBACK_DAYS} daily form indexes", level="info")
        s.log(f"saw {sum(cik_total.values())} 8-K filings across {len(cik_total)} issuers", level="info")

        awarded = 0
        for cik, n in cik_total.items():
            ticker = cik_map.get(cik)
            if not ticker:
                continue
            days_active = len(set(cik_per_day[cik]))
            if n >= 3:
                key = "triple_8k"
                pts, reason = weight(SCAN_NAME, key, DEFAULT_TRIPLE_8K), f"{n} 8-K filings in {LOOKBACK_DAYS}d ({days_active} days)"
            elif n >= 2:
                key = "multiple_8k"
                pts, reason = weight(SCAN_NAME, key, DEFAULT_MULTIPLE_8K), f"{n} 8-K filings in {LOOKBACK_DAYS}d"
            else:
                key = "single_8k"
                pts, reason = weight(SCAN_NAME, key, DEFAULT_SINGLE_8K), f"8-K filed (material event)"
            s.award(ticker, pts, reason, attr_key=key)
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
