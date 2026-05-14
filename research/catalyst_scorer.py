#!/usr/bin/env python3
"""
Catalyst Scorer — Automated Gemini scoring of stocks with news/EDGAR data.

Feeds each stock's Finnhub articles + EDGAR filings to Gemini one at a time,
scores catalyst potential 0-10 with a blurb, and saves to catalyst_scores.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ATTRIBUTES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output", "causal_attributes.json",
)

# Generic headline filter (same as digest_builder)
_GENERIC_RE = re.compile(
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


def _load_catalyst_attrs() -> list[dict]:
    with open(ATTRIBUTES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("catalysts", [])


def _format_stock_context(ticker: str, scan_info: dict,
                          articles: list[dict], filings: list[dict],
                          max_headlines: int = 8) -> str:
    """Format a single stock's news data for the Gemini prompt."""
    lines = [f"TICKER: {ticker}"]

    if scan_info:
        company = scan_info.get("company", "")
        sector = scan_info.get("sector", "")
        industry = scan_info.get("industry", "")
        mcap = scan_info.get("market_cap")
        price = scan_info.get("price")
        mcap_str = (f"${mcap / 1e6:.0f}M" if mcap and mcap < 1e9
                    else f"${mcap / 1e9:.1f}B" if mcap else "?")
        price_str = f"${price:.2f}" if price else "?"
        lines.append(f"Company: {company} | {sector}/{industry} | MCap: {mcap_str} | Price: {price_str}")

        preconds = scan_info.get("preconditions", [])
        active = [p for p in preconds if p.get("score", 0) > 0]
        if active:
            plist = ", ".join(f"{p['name']}={p['score']}" for p in active[:5])
            lines.append(f"Preconditions: {plist}")

    if filings:
        lines.append(f"EDGAR filings ({len(filings)}):")
        for f in filings:
            form = f.get("form", "")
            date = f.get("date", "")
            items = f.get("items", [])
            cats = f.get("catalysts", [])
            desc = f.get("description", "")
            filer = f.get("filer", "")
            parts = [f"  {date} {form}"]
            if items:
                parts.append(f"[{','.join(items)}]")
            if cats:
                parts.append(f"→ {', '.join(cats)}")
            if desc:
                parts.append(f"— {desc}")
            if filer:
                parts.append(f"({filer})")
            lines.append(" ".join(parts))

    if articles:
        sorted_arts = sorted(articles, key=lambda a: a.get("datetime", 0))
        shown = sorted_arts[:max_headlines]
        lines.append(f"News articles ({len(articles)} total, showing {len(shown)}):")
        for a in shown:
            dt = a.get("datetime", 0)
            date_str = datetime.fromtimestamp(dt).strftime("%Y-%m-%d %H:%M") if dt else "?"
            headline = a.get("headline", "")[:120]
            summary = a.get("summary", "")[:200]
            src = a.get("source", "")
            lines.append(f"  [{date_str}] ({src}) {headline}")
            if summary.strip():
                lines.append(f"    {summary}")

    return "\n".join(lines)


def _score_one_stock(client, ticker: str, context: str,
                     catalyst_attrs: list[dict]) -> dict:
    """Score a single stock via Gemini. Returns {"score": int, "blurb": str}."""
    cat_list = "\n".join(
        f"- {a['id']}: {a['name']} — {a['description']}"
        for a in catalyst_attrs
    )

    prompt = f"""You are a stock catalyst analyst. Based on the NEWS ARTICLES and EDGAR FILINGS below, score this stock's catalyst potential.

IMPORTANT: A "catalyst" is a specific recent NEWS EVENT, filing, or announcement — NOT the stock's technical setup, valuation, or precondition scores. Do NOT restate precondition data (short interest, 52-week position, float size, insider ownership, growth metrics) as catalysts. If the only information is precondition/technical data with no real news events, return total_score: 0.

DATES MATTER. Every event has a date in the context below (articles are tagged [YYYY-MM-DD HH:MM]; filings start with their filing date). In the "blurb" and in every catalyst "detail", you MUST cite the relevant date(s) in YYYY-MM-DD form so the reader knows when each event happened. If multiple events are summarized, cite the date of each.

{context}

Score against these catalyst types (0-10 each, 0 = not present, only score based on actual news/filings):
{cat_list}

Return a JSON object:
{{
  "total_score": <sum of all catalyst scores, 0-100>,
  "blurb": "<1-2 sentence summary of the catalysts found, WITH the date(s) the events occurred (e.g. 'On 2026-05-12, ...'). Or 'No significant catalysts.'>",
  "catalysts": [
    {{"id": "catalyst_id", "score": N, "date": "YYYY-MM-DD", "detail": "brief why, including the date of the event"}}
  ]
}}

Only include catalysts with score > 0. Return ONLY valid JSON, no markdown."""

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=config.MODEL_RESEARCH,
                contents=prompt,
            )
            text_parts = [
                p.text for p in resp.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            ]
            text = "\n".join(text_parts).strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                total = result.get("total_score", 0)
                blurb = result.get("blurb", "")
                catalysts = result.get("catalysts", [])
                return {"score": total, "blurb": blurb, "catalysts": catalysts}
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    return {"score": 0, "blurb": "Scoring failed."}


def score_all_catalysts(
    finnhub_data: dict[str, list[dict]],
    edgar_data: dict[str, list[dict]],
    scan_stocks: dict[str, dict],
    max_cap: float | None = 10e9,
):
    """Score all stocks with news/EDGAR data via Gemini, one at a time."""
    from google import genai
    from research.digest_builder import save_batch_scores

    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    catalyst_attrs = _load_catalyst_attrs()

    # Collect stocks that have any data
    all_tickers = set(finnhub_data.keys()) | set(edgar_data.keys())
    to_score = []

    for ticker in sorted(all_tickers):
        scan_info = scan_stocks.get(ticker, {})
        mcap = scan_info.get("market_cap")
        if max_cap and mcap and mcap > max_cap:
            continue

        fh_articles = finnhub_data.get(ticker, [])
        ed_filings = edgar_data.get(ticker, [])

        real_articles = [
            a for a in fh_articles
            if a.get("headline") and not _GENERIC_RE.search(a["headline"])
        ]

        if not real_articles and not ed_filings:
            continue

        to_score.append({
            "ticker": ticker,
            "scan_info": scan_info,
            "articles": real_articles,
            "filings": ed_filings,
        })

    print(f"\n── Catalyst scoring: {len(to_score)} stocks with news data ──\n")

    scored = 0
    batch_scores = {}
    batch_lock = None

    # Parallelism: latency-bound (~5-10s per Gemini call), so concurrent
    # workers give a near-linear speedup until we hit the model's RPM quota.
    # Configurable via env var; default 6 keeps us safely under typical limits.
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = int(os.environ.get("CATALYST_SCORE_WORKERS", "6"))
    batch_lock = threading.Lock()

    def _do_one(i_entry):
        i, entry = i_entry
        ticker = entry["ticker"]
        context = _format_stock_context(
            ticker, entry["scan_info"], entry["articles"], entry["filings"]
        )
        result = _score_one_stock(client, ticker, context, catalyst_attrs)
        return i, ticker, result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, (i, e)) for i, e in enumerate(to_score, 1)]
        for fut in as_completed(futures):
            i, ticker, result = fut.result()
            with batch_lock:
                batch_scores[ticker] = result
                scored += 1
                local_scored = scored
                checkpoint = (local_scored % 50 == 0)
                if checkpoint:
                    to_save = batch_scores
                    batch_scores = {}

            score = result["score"]
            label = f"C={score}" if score > 0 else "C=0"
            print(f"  {i}/{len(to_score)} {ticker:6s} {label}", flush=True)

            if checkpoint:
                save_batch_scores(to_save, session_note="auto-gemini")

    # Save remaining
    if batch_scores:
        save_batch_scores(batch_scores, session_note="auto-gemini")

    print(f"\n  Done. Scored {scored} stocks via Gemini.")
