"""
EDGAR 10-Q / 10-K XBRL refresh.

When a company files a fresh 10-Q or 10-K, we pull the issuer's XBRL
companyfacts JSON (https://data.sec.gov/api/xbrl/companyfacts/CIK########.json)
and compute the year-over-year revenue + net-income deltas for the most
recent reported period. That gives us a clean, deterministic earnings-
surprise signal anchored to the SEC's structured filings rather than
yfinance's scraped estimates.

Scoring (revenue YoY for the most recent reported period):
  ≥ +30 %  →  +10 pts  ("revenue acceleration")
  ≥ +15 %  →   +6 pts  ("revenue growth")
  ≤ -20 %  →   +5 pts  ("revenue contraction — review")
  fresh filing alone (no comparable prior year) → +2 pts ("XBRL refresh")

Net income YoY adds +3 pts when it crosses from loss to profit (turn-
around) and +2 pts when it grows ≥ +50 %.

Revenue tag fallback chain: us-gaap:Revenues →
RevenueFromContractWithCustomerExcludingAssessedTax →
SalesRevenueNet → Revenues. Different filers prefer different tags;
the registrant picks whichever fits its disclosure regime.

Scheduling: daily at 14:00 UTC (one hour after fundamentals_snap), with
a 2-business-day lookback window.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

from research.live_score_engine import Session, LIVE_DIR
from research.scan_weights import weight
from research.scans.insider_cluster import (
    _load_cik_to_ticker, _business_days, _fetch_form_index,
)

SCAN_NAME = "xbrl_facts"
LOOKBACK_DAYS = 2

SEC_HEADERS = {
    "User-Agent": "Raymond Research smanderson721@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/plain, */*",
    "Host": "data.sec.gov",
}
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# polite gap between SEC fetches (10 req/s SEC fair-access ceiling)
FETCH_GAP_SEC = 0.12
# safety cap on companyfacts pulls per run
MAX_COMPANYFACTS_PER_RUN = 200

REV_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
NI_TAGS = (
    "NetIncomeLoss",
    "ProfitLoss",
)


def _parse_form_index_for_10qk(text: str) -> list[tuple[str, int]]:
    """Extract (form, cik) tuples for 10-Q / 10-K filings."""
    out: list[tuple[str, int]] = []
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
        if form not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
            continue
        cik = None
        for tok in parts[1:]:
            if tok.isdigit():
                cik = int(tok)
                break
        if cik:
            out.append((form, cik))
    return out


def _fetch_companyfacts(cik: int) -> dict | None:
    url = COMPANYFACTS_URL.format(cik=cik)
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _pick_series(facts: dict, tag_choices: tuple[str, ...]) -> list[dict] | None:
    """Pick the freshest USD series across a list of candidate tags.

    Different filers (and even the same filer across years) use different
    GAAP tags for revenue / net income. After ASC 606 (2018) most issuers
    moved from us-gaap:Revenues to RevenueFromContractWithCustomerExcluding
    AssessedTax, but the legacy tag is often retained for historical
    periods, so a naive "first match wins" would lock onto a stale series.
    Returns the candidate whose most recent record has the highest filed
    date.
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    best = None
    best_latest = ""
    for tag in tag_choices:
        series = gaap.get(tag, {}).get("units", {}).get("USD")
        if not series:
            continue
        latest = max((r.get("filed", "") for r in series), default="")
        if latest > best_latest:
            best = series
            best_latest = latest
    return best


def _period_days(rec: dict) -> int:
    """Return the (end - start) duration in days, or 0 if missing."""
    s, e = rec.get("start"), rec.get("end")
    if not s or not e:
        return 0
    try:
        ds = datetime.strptime(s, "%Y-%m-%d").date()
        de = datetime.strptime(e, "%Y-%m-%d").date()
        return (de - ds).days
    except Exception:
        return 0


def _latest_period_yoy(series: list[dict], form_filter: tuple[str, ...]) -> tuple[dict | None, dict | None]:
    """Return (latest, prior_year_same_period) records.

    Identifies periods by the XBRL `end` date — NOT by the `fy` field.
    A 10-K typically includes multiple years of comparative data in the
    same filing, and the `fy` field on every record reports the filing's
    fiscal year, not the period's. Using `end` makes the match robust to
    those restatements.

    For 10-Q filings, each period often has two records (standalone
    quarter + fiscal YTD). We pick the longest-duration record per
    `end` date — that's the YTD cumulative, well-defined and apples-
    to-apples across years.
    """
    qualified = [r for r in series if r.get("form") in form_filter and r.get("end")]
    if not qualified:
        return None, None

    # bucket by end-date, pick the longest-duration record per bucket
    by_end: dict[str, dict] = {}
    for r in qualified:
        end = r["end"]
        cur = by_end.get(end)
        if cur is None or _period_days(r) > _period_days(cur):
            by_end[end] = r

    latest_end = max(by_end.keys())
    latest = by_end[latest_end]
    latest_dur = _period_days(latest)
    if latest_dur == 0:
        return latest, None

    # prior-year same period: end date ~365 days earlier, similar duration
    try:
        target_end = datetime.strptime(latest_end, "%Y-%m-%d").date()
    except Exception:
        return latest, None

    best_prior = None
    best_delta_days = 999
    for end_str, rec in by_end.items():
        try:
            e = datetime.strptime(end_str, "%Y-%m-%d").date()
        except Exception:
            continue
        gap = (target_end - e).days
        if not (350 <= gap <= 385):
            continue
        # duration must match within 14 days (handles fiscal calendar drift)
        if abs(_period_days(rec) - latest_dur) > 14:
            continue
        delta = abs(gap - 365)
        if delta < best_delta_days:
            best_delta_days = delta
            best_prior = rec

    return latest, best_prior


def run() -> dict:
    cik_map = _load_cik_to_ticker()
    if not cik_map:
        with Session(SCAN_NAME, note="EDGAR 10-Q/10-K XBRL refresh") as s:
            s.log("failed to load SEC ticker→CIK map", level="info")
            return {"ok": False, "reason": "no_ticker_map"}

    with Session(SCAN_NAME, note="EDGAR 10-Q/10-K XBRL refresh") as s:
        today = datetime.now(timezone.utc).date()
        days = _business_days(today - timedelta(days=1), LOOKBACK_DAYS)
        # dedup: one entry per CIK with strongest form seen
        cik_form: dict[int, str] = {}
        days_with_data = 0
        for d in days:
            txt = _fetch_form_index(d)
            if not txt:
                continue
            days_with_data += 1
            for form, cik in _parse_form_index_for_10qk(txt):
                # 10-K dominates 10-Q; prefer non-amendments
                cur = cik_form.get(cik)
                if cur is None:
                    cik_form[cik] = form
                elif "10-K" in form and "10-K" not in cur:
                    cik_form[cik] = form

        s.log(f"loaded {days_with_data}/{LOOKBACK_DAYS} daily form indexes", level="info")
        s.log(f"saw 10-Q/10-K filings from {len(cik_form)} issuers", level="info")

        awarded = 0
        no_facts = 0
        budget = MAX_COMPANYFACTS_PER_RUN
        import time

        # iterate issuers — restrict to ones with a ticker in our map
        for cik, form in cik_form.items():
            if budget <= 0:
                break
            ticker = cik_map.get(cik)
            if not ticker:
                continue

            facts = _fetch_companyfacts(cik)
            time.sleep(FETCH_GAP_SEC)
            budget -= 1
            if not facts:
                no_facts += 1
                continue

            rev_series = _pick_series(facts, REV_TAGS)
            ni_series  = _pick_series(facts, NI_TAGS)
            if rev_series is None:
                continue

            form_filter = ("10-K", "10-K/A") if "10-K" in form else ("10-Q", "10-Q/A")
            rev_latest, rev_prior = _latest_period_yoy(rev_series, form_filter)
            if rev_latest is None:
                continue

            pts = 0.0
            bits: list[str] = []
            primary_attr: str | None = None
            fp_label = rev_latest.get("fp", "?")
            fy_label = rev_latest.get("fy", "?")

            # revenue YoY
            if rev_prior and rev_prior.get("val") not in (None, 0):
                yoy = (rev_latest["val"] - rev_prior["val"]) / abs(rev_prior["val"])
                pct = yoy * 100
                if yoy >= 0.30:
                    pts += weight(SCAN_NAME, "revenue_acceleration", 10.0)
                    primary_attr = "revenue_acceleration"
                    bits.append(f"revenue +{pct:.0f}% YoY")
                elif yoy >= 0.15:
                    pts += weight(SCAN_NAME, "revenue_growth", 6.0)
                    primary_attr = "revenue_growth"
                    bits.append(f"revenue +{pct:.0f}% YoY")
                elif yoy <= -0.20:
                    pts += weight(SCAN_NAME, "revenue_contraction", 5.0)
                    primary_attr = "revenue_contraction"
                    bits.append(f"revenue {pct:.0f}% YoY")
                # else: modest growth/contraction — informational only
                else:
                    bits.append(f"revenue {pct:+.0f}% YoY")
            else:
                pts += weight(SCAN_NAME, "xbrl_refresh_first_time", 2.0)
                primary_attr = "xbrl_refresh_first_time"
                bits.append("fresh XBRL refresh")

            # net income YoY
            if ni_series:
                ni_latest, ni_prior = _latest_period_yoy(ni_series, form_filter)
                if ni_latest and ni_prior and ni_prior.get("val") is not None:
                    pv = ni_prior["val"]
                    lv = ni_latest["val"]
                    if pv < 0 <= lv:
                        pts += weight(SCAN_NAME, "turned_profitable", 3.0)
                        bits.append("turned profitable")
                    elif pv > 0 and lv > 0 and pv != 0:
                        nyoy = (lv - pv) / abs(pv)
                        if nyoy >= 0.50:
                            pts += weight(SCAN_NAME, "ni_strong_growth", 2.0)
                            bits.append(f"NI +{nyoy*100:.0f}% YoY")

            if pts <= 0:
                continue

            reason = f"{form} {fp_label} FY{fy_label}: " + ", ".join(bits)
            s.award(ticker, pts, reason, attr_key=primary_attr)
            awarded += 1

        return {
            "ok": True,
            "days_with_data": days_with_data,
            "issuers_seen": len(cik_form),
            "companyfacts_fetched": MAX_COMPANYFACTS_PER_RUN - budget,
            "no_facts": no_facts,
            "awarded": awarded,
        }


if __name__ == "__main__":
    print(run())
