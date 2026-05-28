"""
NASDAQ trading-halts RSS poll.

NASDAQ publishes every halt and resumption across all US listed equities
(not just NASDAQ-listed — they also carry NYSE/Arca/BATS halts) as a
public RSS-style XML feed at:

  https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts

The XML wraps a stream of <ndaq:NASDAQHaltsReportRecord> rows. Each row
has Issue Symbol, ReasonCode, HaltDate/Time, ResumeDate/Time, plus the
listing market. Reason codes worth knowing:

  T1     news pending                      — pre-news halt; very strong
  T2     news released                     — confirming release
  T5     single-stock circuit breaker      — 10% move trigger; volatile
  T6     extraordinary market activity     — order imbalance / glitch
  T8     ETF component issue               — ETF only
  T12    additional info requested         — usually pre-rerating
  H10    SEC suspension                    — regulatory; very negative
  LUDP   limit up-limit down paused        — Reg NMS volatility halt
  MWC1/2/3  market-wide circuit breaker    — index-level
  M     volatility trading pause           — generic

We score:
  - any halt with no resume time         → currently halted (live signal)
  - any halt where resume was within 30m → fresh-resume (often a fade trade)
  - T1 / H10 / T12 are flagged at higher points than routine LUDP / T5.

We poll every 2 min during US market hours (14:00-20:30 UTC, Mon-Fri).
Outside market hours the feed is empty / stale.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import requests

from research.live_score_engine import Session
from research.scan_weights import weight

SCAN_NAME = "halt_tape"
FEED_URL = "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
HEADERS = {
    "User-Agent": "Raymond Research smanderson721@gmail.com",
    "Accept": "application/xml, text/xml, */*",
}
TIMEOUT = 20

# Per-halt scoring. Higher = stronger signal.
# Defaults shown here; live values come from data/live/scan_weights.json.
# Tuple: (default_points, attr_key, label)
REASON_PTS: dict[str, tuple[float, str, str]] = {
    "T1":   (10.0, "halt_t1_news_pending",          "news-pending halt (T1)"),
    "T12":  (8.0,  "halt_t12_info_requested",       "additional-info halt (T12)"),
    "H10":  (12.0, "halt_h10_sec_suspension",       "SEC trading suspension (H10)"),
    "T6":   (6.0,  "halt_t6_extraordinary_activity", "extraordinary activity (T6)"),
    "T2":   (5.0,  "halt_t2_news_released",         "news-released halt (T2)"),
    "T5":   (4.0,  "halt_t5_circuit_breaker",       "single-stock circuit breaker (T5)"),
    "LUDP": (3.0,  "halt_ludp_limit_up_down",       "limit up-down pause"),
    "M":    (3.0,  "halt_m_volatility_pause",       "volatility trading pause (M)"),
    "T8":   (2.0,  "halt_t8_etf_component",         "ETF component halt (T8)"),
}
DEFAULT_PTS = (3.0, "halt_other", "trading halt")

# Bonus default values (resolved live via weight())
DEFAULT_LIVE_HALT_BONUS = 4.0       # no resume time yet — still halted
FRESH_RESUME_WINDOW_MIN = 30        # resumed in the last N min
DEFAULT_FRESH_RESUME_BONUS = 2.0

# Cap how many filings we score per run (rare to see this many but defends
# against a bad-XML day producing thousands of duplicate rows).
MAX_AWARDS_PER_RUN = 200


def _parse_dt(date_str: str, time_str: str) -> datetime | None:
    """Combine "MM/DD/YYYY" + "HH:MM:SS" (assumed Eastern) → UTC datetime.

    NASDAQ publishes halt times in US/Eastern. We translate via the
    fixed UTC-4 / UTC-5 offset by checking US DST rules approximately:
    DST runs second Sun of March through first Sun of November.
    """
    if not date_str or not time_str:
        return None
    try:
        m, d, y = (int(x) for x in date_str.split("/"))
        hh, mm, ss = (int(x) for x in time_str.split(":"))
    except Exception:
        return None
    # Approximate US DST boundaries (good enough for halt scoring)
    naive = datetime(y, m, d, hh, mm, ss)
    # 2nd Sunday of March
    march_first = datetime(y, 3, 1)
    dst_start = march_first + timedelta(days=(6 - march_first.weekday()) % 7 + 7)
    # 1st Sunday of November
    nov_first = datetime(y, 11, 1)
    dst_end = nov_first + timedelta(days=(6 - nov_first.weekday()) % 7)
    in_dst = dst_start <= naive < dst_end
    offset = timedelta(hours=4 if in_dst else 5)
    return (naive + offset).replace(tzinfo=timezone.utc)


def _parse_feed(xml_text: str) -> list[dict]:
    """Extract halt records from the feed. Each record dict has:
      symbol, reason_code, halt_dt, resume_dt (or None), market

    The feed is a plain RSS 2.0 document — each halt is one ``<item>``
    whose data fields live in the ``ndaq:`` namespace. We avoid the
    XML parser entirely because each item also embeds a CDATA HTML
    table that has historically tripped up strict parsers when NASDAQ
    issues issuer names containing ampersands or quotes.
    """
    out: list[dict] = []
    item_re = re.compile(r"<item>(.*?)</item>", re.DOTALL)
    for blob in item_re.findall(xml_text):
        rec: dict[str, str] = {}
        for tag_re, key in [
            (r"<ndaq:IssueSymbol>([^<]*)</ndaq:IssueSymbol>",   "symbol"),
            (r"<ndaq:ReasonCode>([^<]*)</ndaq:ReasonCode>",     "reason_code"),
            (r"<ndaq:HaltDate>([^<]*)</ndaq:HaltDate>",         "halt_date"),
            (r"<ndaq:HaltTime>([^<]*)</ndaq:HaltTime>",         "halt_time"),
            (r"<ndaq:ResumptionDate>([^<]*)</ndaq:ResumptionDate>",         "resume_date"),
            (r"<ndaq:ResumptionTradeTime>([^<]*)</ndaq:ResumptionTradeTime>", "resume_time"),
            (r"<ndaq:Market>([^<]*)</ndaq:Market>",             "market"),
        ]:
            m = re.search(tag_re, blob)
            if m:
                rec[key] = m.group(1).strip()
        symbol = (rec.get("symbol") or "").upper()
        if not symbol:
            continue
        # halt_time may carry sub-second precision like "15:42:09.063" —
        # _parse_dt only handles HH:MM:SS, so trim the fraction first.
        halt_time = (rec.get("halt_time") or "").split(".")[0]
        resume_time = (rec.get("resume_time") or "").split(".")[0]
        out.append({
            "symbol": symbol,
            "reason_code": rec.get("reason_code") or "",
            "halt_dt": _parse_dt(rec.get("halt_date", ""), halt_time),
            "resume_dt": _parse_dt(rec.get("resume_date", ""), resume_time),
            "market": rec.get("market") or "",
        })
    return out


def run() -> dict:
    with Session(SCAN_NAME, note="NASDAQ trading halts feed") as s:
        try:
            r = requests.get(FEED_URL, headers=HEADERS, timeout=TIMEOUT)
        except Exception as e:
            s.log(f"feed fetch failed: {e}", level="info")
            return {"ok": False, "reason": "fetch_failed"}
        if r.status_code != 200:
            s.log(f"feed HTTP {r.status_code}", level="info")
            return {"ok": False, "reason": f"http_{r.status_code}"}

        records = _parse_feed(r.text)
        s.log(f"parsed {len(records)} halt records from feed", level="info")

        # only score halts whose halt_dt is within the last 6 hours — older
        # rows are noise (the feed keeps a long tail for the day)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=6)
        fresh = [rec for rec in records if rec["halt_dt"] and rec["halt_dt"] >= cutoff]

        # Deduplicate: a ticker can have multiple halts in a day; we keep
        # the most-recent halt per (symbol, reason_code) so each distinct
        # event scores once per run.
        latest: dict[tuple[str, str], dict] = {}
        for rec in fresh:
            key = (rec["symbol"], rec["reason_code"])
            prev = latest.get(key)
            if not prev or rec["halt_dt"] > prev["halt_dt"]:
                latest[key] = rec

        awarded = 0
        for rec in list(latest.values())[:MAX_AWARDS_PER_RUN]:
            default_pts, attr_key, label = REASON_PTS.get(rec["reason_code"], DEFAULT_PTS)
            pts = weight(SCAN_NAME, attr_key, default_pts)
            reason_bits = [label]
            if rec["resume_dt"] is None:
                pts += weight(SCAN_NAME, "halt_still_live_bonus", DEFAULT_LIVE_HALT_BONUS)
                reason_bits.append("still halted")
            elif (now - rec["resume_dt"]).total_seconds() <= FRESH_RESUME_WINDOW_MIN * 60:
                pts += weight(SCAN_NAME, "halt_fresh_resume_bonus", DEFAULT_FRESH_RESUME_BONUS)
                resume_age_min = int((now - rec["resume_dt"]).total_seconds() // 60)
                reason_bits.append(f"resumed {resume_age_min}m ago")
            else:
                reason_bits.append("resumed")
            if rec["market"]:
                reason_bits.append(rec["market"])
            s.award(rec["symbol"], pts, " · ".join(reason_bits), attr_key=attr_key)
            awarded += 1

        return {
            "ok": True,
            "records_in_feed": len(records),
            "fresh_records": len(fresh),
            "distinct_halt_events": len(latest),
            "awarded": awarded,
        }


if __name__ == "__main__":
    print(run())
