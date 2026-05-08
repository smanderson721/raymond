#!/usr/bin/env python3
"""
Digest Builder — Formats Finnhub + EDGAR data into readable batches
for LLM-in-chat scoring.

Reads the latest news scan data and scan_results.json, then outputs
numbered batch files to research_output/digest/ that the LLM can read
one at a time without compression.

Usage:
    python pipeline.py --build-digest
    python pipeline.py --build-digest --batch-size 25
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output",
)
DIGEST_DIR = os.path.join(OUTPUT_DIR, "digest")
SCORES_FILE = os.path.join(OUTPUT_DIR, "catalyst_scores.json")
SCAN_FILE = os.path.join(OUTPUT_DIR, "scan_results.json")
NEWS_CACHE_FILE = os.path.join(OUTPUT_DIR, "news_cache.json")


def load_news_cache() -> dict:
    """Load cached Finnhub + EDGAR raw data for resume support."""
    if os.path.exists(NEWS_CACHE_FILE):
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_news_cache(
    finnhub_data: dict[str, list[dict]],
    edgar_data: dict[str, list[dict]],
):
    """Cache raw Finnhub + EDGAR data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "finnhub": finnhub_data,
        "edgar": edgar_data,
    }
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    fh_count = sum(len(v) for v in finnhub_data.values())
    ed_count = sum(len(v) for v in edgar_data.values())
    print(f"  Cached {len(finnhub_data)} Finnhub tickers ({fh_count} articles) + {len(edgar_data)} EDGAR tickers ({ed_count} filings)")


def load_scan_stocks() -> dict[str, dict]:
    """Load the latest scan_results.json and return {ticker: stock_data}."""
    if not os.path.exists(SCAN_FILE):
        return {}
    with open(SCAN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    scans = data.get("scans", [])
    if not scans:
        return {}
    stocks = scans[0].get("stocks", [])
    return {s["ticker"]: s for s in stocks}


def load_scores() -> dict:
    """Load existing catalyst scores."""
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sessions": []}


def save_scores(data: dict):
    """Save catalyst scores."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_digest(
    finnhub_data: dict[str, list[dict]],
    edgar_data: dict[str, list[dict]],
    scan_stocks: dict[str, dict],
    batch_size: int = 25,
    max_headlines: int = 5,
    max_cap: float | None = 10e9,
    full_tickers: list[str] | None = None,
) -> list[str]:
    """Build digest batch files from raw Finnhub + EDGAR data.

    Tiers stocks by signal strength:
      Tier 1 — EDGAR 8-K/6-K filings (material events, highest signal)
      Tier 2 — Finnhub articles only (news coverage, moderate signal)
    Stocks with no data get auto-scored 0 (not included in batches).

    If full_tickers is provided, any ticker with no data is auto-scored 0
    in catalyst_scores.json so they don't need manual review.

    Returns list of batch file paths created.
    """
    import re

    # Generic headline filter
    generic_re = re.compile(
        r"stocks? (?:are )?moving|top gainers|top losers|after-market session|"
        r"pre-market session|intraday session|after-hours session|"
        r"notable movement|unusual volume|notable gaps|"
        r"what'?s going on in today|let'?s take a look|let'?s uncover|"
        r"wondering what|which stocks are experiencing|"
        r"top movers|top stock movements|curious about|these stocks are the most active|"
        r"these stocks (?:are|that are) (?:show|movi)|gapping stocks|"
        r"stay updated|here are the top movers|stock movements in today|"
        r"^\d+ \w+ stocks (?:are )?moving|opening bell|"
        r"top gainers and losers|latest market trends|latest stock movements|"
        r"trading volume of these stocks|deviating from the norm|"
        r"dow (?:dips|jumps|falls|rises|surges)|s&p 500 (?:dips|jumps|falls|rises)|"
        r"market (?:wrap|roundup|recap)|wall street (?:opens|closes)",
        re.IGNORECASE,
    )

    # Collect all tickers that have any data
    all_tickers = set(finnhub_data.keys()) | set(edgar_data.keys())

    # Build per-ticker entries, assign tier
    tier1 = []  # EDGAR 8-K/6-K
    tier2 = []  # Finnhub-only (no material EDGAR)
    skipped_cap = 0
    no_data = 0

    for ticker in sorted(all_tickers):
        scan_info = scan_stocks.get(ticker, {})
        mcap = scan_info.get("market_cap")
        if max_cap and mcap and mcap > max_cap:
            skipped_cap += 1
            continue

        fh_articles = finnhub_data.get(ticker, [])
        ed_filings = edgar_data.get(ticker, [])

        real_articles = [
            a for a in fh_articles
            if a.get("headline") and not generic_re.search(a["headline"])
        ]

        # Article-count cap: if a stock has more than this many articles,
        # it's almost certainly a high-volume mid/large cap getting
        # generic coverage (analyst notes, ETF mentions, sector roundups).
        # Skip it unless EDGAR has a material filing.
        ARTICLE_COUNT_CAP = 15
        has_material = any(
            f.get("form") in ("8-K", "6-K", "8-K/A", "6-K/A") for f in ed_filings
        )
        if len(real_articles) > ARTICLE_COUNT_CAP and not has_material:
            no_data += 1  # treated as noise; auto-zeroed below
            continue

        if not real_articles and not ed_filings:
            no_data += 1
            continue

        entry = {
            "ticker": ticker,
            "scan_info": scan_info,
            "articles": real_articles,
            "filings": ed_filings,
        }

        if has_material:
            tier1.append(entry)
        else:
            tier2.append(entry)

    # Sort within each tier by article count (more articles = more to analyze)
    tier1.sort(key=lambda e: -len(e["articles"]))
    tier2.sort(key=lambda e: -len(e["articles"]))

    total_with_data = len(tier1) + len(tier2)
    print(f"  Digest: {total_with_data} tickers with data (from {len(all_tickers)} total)")
    print(f"    Tier 1 (EDGAR 8-K/6-K): {len(tier1)} stocks")
    print(f"    Tier 2 (Finnhub only):   {len(tier2)} stocks")
    if skipped_cap:
        print(f"    Skipped (market cap):    {skipped_cap}")
    print(f"    No useful data:          {no_data}")

    # Auto-score zero-signal tickers at catalyst_score = 0
    if full_tickers:
        tickers_with_data = {e["ticker"] for e in tier1} | {e["ticker"] for e in tier2}
        zero_tickers = [t for t in full_tickers if t not in tickers_with_data]
        if zero_tickers:
            zero_scores = {t: {"score": 0, "blurb": "No news or EDGAR filings in scan window."} for t in zero_tickers}
            save_batch_scores(zero_scores, session_note="auto-zero (no signal)")
            print(f"    Auto-scored at 0:        {len(zero_tickers)} stocks")

    # Combine tiers (tier 1 first, then tier 2)
    all_entries = tier1 + tier2

    # Split into batches
    batches = []
    for i in range(0, len(all_entries), batch_size):
        batches.append(all_entries[i : i + batch_size])

    # Write batch files
    os.makedirs(DIGEST_DIR, exist_ok=True)

    # Clear old batches
    for f in os.listdir(DIGEST_DIR):
        if f.startswith("batch_") and f.endswith(".txt"):
            os.remove(os.path.join(DIGEST_DIR, f))

    tier1_batches = (len(tier1) + batch_size - 1) // batch_size if tier1 else 0

    batch_files = []
    for batch_idx, batch in enumerate(batches, 1):
        tier_label = "TIER 1 (EDGAR)" if batch_idx <= tier1_batches else "TIER 2 (Finnhub)"
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"  BATCH {batch_idx} of {len(batches)} — {len(batch)} stocks — {tier_label}")
        lines.append(f"  Score each stock's catalyst potential 0-10.")
        lines.append(f"  Write a 1-sentence blurb summarizing your analysis.")
        lines.append(f"{'=' * 70}")
        lines.append("")

        for entry in batch:
            _write_stock_entry(lines, entry, max_headlines)

        filename = f"batch_{batch_idx:03d}.txt"
        filepath = os.path.join(DIGEST_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        batch_files.append(filepath)

    print(f"  Written {len(batch_files)} batch files to {DIGEST_DIR}/")
    print(f"    Batches 1-{tier1_batches}: Tier 1 (EDGAR material events)")
    if len(batch_files) > tier1_batches:
        print(f"    Batches {tier1_batches+1}-{len(batch_files)}: Tier 2 (Finnhub articles only)")
    return batch_files


def _write_stock_entry(lines: list[str], entry: dict, max_headlines: int = 5):
    """Write a single stock entry to the batch lines."""
    ticker = entry["ticker"]
    info = entry["scan_info"]
    articles = entry["articles"]
    filings = entry["filings"]

    lines.append(f"--- {ticker} ---")

    # Company info from scan
    if info:
        company = info.get("company", "")
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        mcap = info.get("market_cap")
        mcap_str = f"${mcap / 1e6:.0f}M" if mcap and mcap < 1e9 else f"${mcap / 1e9:.1f}B" if mcap else "?"
        pscore = info.get("precondition_score", 0)
        price = info.get("price")
        price_str = f"${price:.2f}" if price else "?"
        lines.append(f"  {company} | {sector}/{industry} | MCap: {mcap_str} | Price: {price_str} | P-score: {pscore}")

        preconds = info.get("preconditions", [])
        active = [p for p in preconds if p.get("score", 0) > 0]
        if active:
            plist = ", ".join(f"{p['name']}={p['score']}" for p in active[:5])
            lines.append(f"  Preconditions: {plist}")
    else:
        lines.append(f"  (no precondition data)")

    # EDGAR filings
    if filings:
        lines.append(f"  EDGAR ({len(filings)} filings):")
        for f in filings:
            form = f.get("form", "")
            date = f.get("date", "")
            filer = f.get("filer", "")
            items = f.get("items", [])
            cats = f.get("catalysts", [])
            desc = f.get("description", "")
            parts = [f"    {date} {form}"]
            if items:
                parts.append(f"[{','.join(items)}]")
            if cats:
                parts.append(f"→ {', '.join(cats)}")
            if desc:
                parts.append(f"— {desc}")
            if filer:
                parts.append(f"({filer})")
            lines.append(" ".join(parts))

    # Finnhub articles
    if articles:
        sorted_arts = sorted(articles, key=lambda a: a.get("datetime", 0))
        shown = sorted_arts[:max_headlines]
        lines.append(f"  Finnhub ({len(articles)} real articles, showing {len(shown)}):")
        for a in shown:
            dt = a.get("datetime", 0)
            date_str = datetime.fromtimestamp(dt).strftime("%m/%d") if dt else "?"
            headline = a.get("headline", "")[:120]
            summary = a.get("summary", "")[:250]
            src = a.get("source", "")
            lines.append(f"    [{date_str}] ({src}) {headline}")
            if summary.strip():
                lines.append(f"      {summary}")
        if len(articles) > max_headlines:
            lines.append(f"    ... +{len(articles) - max_headlines} more")
    elif not filings:
        lines.append(f"  (no articles, no filings)")

    lines.append("")


def save_batch_scores(scores: dict[str, dict], session_note: str = ""):
    """Save a batch of LLM-scored catalyst scores.

    Args:
        scores: {ticker: {"score": int, "blurb": str}}
        session_note: optional note about this scoring session
    """
    db = load_scores()

    # Get or create the current session
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    current_session = None
    for session in db["sessions"]:
        if session.get("date") == today:
            current_session = session
            break

    if current_session is None:
        current_session = {
            "date": today,
            "note": session_note,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "scores": {},
        }
        db["sessions"].insert(0, current_session)

    # Merge new scores into session
    current_session["scores"].update(scores)
    current_session["scored_at"] = datetime.now(timezone.utc).isoformat()
    if session_note:
        current_session["note"] = session_note

    total = len(current_session["scores"])
    save_scores(db)
    print(f"  Saved {len(scores)} scores (session total: {total}) to {SCORES_FILE}")


def get_latest_scores() -> dict[str, dict]:
    """Get merged scores across all sessions as {ticker: {score, blurb}}.
    
    Later sessions override earlier ones for the same ticker.
    """
    db = load_scores()
    merged = {}
    # Sessions are ordered newest-first, so iterate oldest-first
    for session in reversed(db["sessions"]):
        merged.update(session.get("scores", {}))
    return merged
