#!/usr/bin/env python3
"""
Market-Wide Stock Scanner — yfinance precondition scoring + Gemini catalyst search.

1. Downloads all NASDAQ/NYSE tickers
2. Fetches yfinance .info for each (threaded) to score preconditions
3. Stocks above precondition threshold get Gemini search grounding for catalysts
4. Final score = precondition + catalyst scores
5. Saves results to research_output/scan_results.json

Usage:
    python pipeline.py --stocks-scan
    python pipeline.py --stocks-scan --exchange nasdaq --min-precondition 15
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

SCAN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output", "scan_results.json",
)

ATTRIBUTES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output", "causal_attributes.json",
)

# ── Hot sectors/industries that attract outsized capital flows ─────────
HOT_SECTORS = {"Technology", "Healthcare", "Energy", "Industrials"}
HOT_INDUSTRIES = {
    "artificial intelligence", "machine learning", "ai", "semiconductor",
    "biotechnology", "genomics", "gene therapy", "drug discovery",
    "oncology", "rare disease", "psychedelic", "psilocybin",
    "space", "aerospace", "defense", "military",
    "solar", "renewable", "clean energy", "ev", "electric vehicle",
    "oil", "gas", "refining", "energy",
    "quantum", "cybersecurity", "cloud", "saas", "robotics",
    "blockchain", "crypto", "digital asset",
}


def load_scan_db() -> dict:
    if os.path.exists(SCAN_FILE):
        with open(SCAN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"scans": []}


def save_scan_db(data: dict):
    with open(SCAN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_attributes() -> dict:
    if os.path.exists(ATTRIBUTES_FILE):
        with open(ATTRIBUTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"preconditions": [], "catalysts": []}


# ── Ticker fetching ───────────────────────────────────────────────────

def get_tickers(exchange: str = "all") -> list[str]:
    """Get ticker list from public GitHub repo of US stock symbols."""
    import urllib.request

    print("  Fetching ticker lists...", flush=True)
    tickers = set()

    sources = []
    if exchange in ("all", "nasdaq"):
        sources.append(("NASDAQ", "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt"))
    if exchange in ("all", "nyse"):
        sources.append(("NYSE", "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_tickers.txt"))

    for name, url in sources:
        try:
            print(f"    Fetching {name}...", end="", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
                batch = [line.strip() for line in text.strip().split("\n") if line.strip()]
                tickers.update(batch)
                print(f" {len(batch)}", flush=True)
        except Exception as e:
            print(f" failed: {e}", flush=True)

    # Filter out warrants, units, preferred, etc.
    clean = set()
    for t in tickers:
        t = t.strip().upper()
        if not t or len(t) > 5:
            continue
        if any(c in t for c in [".", "-", "/", "+"]):
            continue
        clean.add(t)

    result = sorted(clean)
    print(f"  Total clean tickers: {len(result)}", flush=True)
    return result


# ── Precondition scoring from yfinance .info ──────────────────────────

def score_preconditions(info: dict) -> list[dict]:
    """Score a stock's precondition attributes from yfinance .info data.
    Returns list of {id, name, score, detail}."""
    scores = []
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    # 1. Low float
    float_shares = info.get("floatShares")
    if float_shares and float_shares > 0:
        if float_shares < 5_000_000:
            scores.append({"id": "low_float_volatility", "name": "Low Float", "score": 10,
                           "detail": f"Float: {float_shares/1e6:.1f}M shares"})
        elif float_shares < 10_000_000:
            scores.append({"id": "low_float_volatility", "name": "Low Float", "score": 8,
                           "detail": f"Float: {float_shares/1e6:.1f}M shares"})
        elif float_shares < 20_000_000:
            scores.append({"id": "low_float_volatility", "name": "Low Float", "score": 6,
                           "detail": f"Float: {float_shares/1e6:.1f}M shares"})
        elif float_shares < 50_000_000:
            scores.append({"id": "low_float_volatility", "name": "Low Float", "score": 3,
                           "detail": f"Float: {float_shares/1e6:.1f}M shares"})

    # 2. Short squeeze setup
    si_pct = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")
    if si_pct and si_pct > 0:
        if si_pct > 0.30:
            scores.append({"id": "short_squeeze_setup", "name": "Short Squeeze Setup", "score": 10,
                           "detail": f"SI: {si_pct*100:.1f}% of float"})
        elif si_pct > 0.20:
            scores.append({"id": "short_squeeze_setup", "name": "Short Squeeze Setup", "score": 8,
                           "detail": f"SI: {si_pct*100:.1f}% of float"})
        elif si_pct > 0.15:
            scores.append({"id": "short_squeeze_setup", "name": "Short Squeeze Setup", "score": 6,
                           "detail": f"SI: {si_pct*100:.1f}% of float"})
        elif si_pct > 0.10:
            scores.append({"id": "short_squeeze_setup", "name": "Short Squeeze Setup", "score": 4,
                           "detail": f"SI: {si_pct*100:.1f}% of float"})
    elif short_ratio and short_ratio > 5:
        scores.append({"id": "short_squeeze_setup", "name": "Short Squeeze Setup", "score": 5,
                       "detail": f"Short ratio: {short_ratio:.1f} days"})

    # 3. Nasdaq compliance risk (sub-$1 stocks attract compliance rallies)
    if price and price < 1.00:
        scores.append({"id": "compliance_risk", "name": "Compliance Risk",
                       "score": 8, "detail": f"Price: ${price:.2f} (below $1 bid)"})
    elif price and price < 2.00:
        scores.append({"id": "compliance_risk", "name": "Compliance Risk",
                       "score": 5, "detail": f"Price: ${price:.2f} (near $1 bid)"})
    elif price and price < 3.00:
        scores.append({"id": "compliance_risk", "name": "Compliance Risk",
                       "score": 3, "detail": f"Price: ${price:.2f}"})

    # 4. Insider accumulation
    insider_pct = info.get("heldPercentInsiders")
    if insider_pct and insider_pct > 0:
        if insider_pct > 0.40:
            scores.append({"id": "insider_accumulation", "name": "High Insider Ownership", "score": 9,
                           "detail": f"Insiders: {insider_pct*100:.1f}%"})
        elif insider_pct > 0.25:
            scores.append({"id": "insider_accumulation", "name": "High Insider Ownership", "score": 7,
                           "detail": f"Insiders: {insider_pct*100:.1f}%"})
        elif insider_pct > 0.15:
            scores.append({"id": "insider_accumulation", "name": "High Insider Ownership", "score": 4,
                           "detail": f"Insiders: {insider_pct*100:.1f}%"})

    # 5. Beaten down / near 52-week low
    low52 = info.get("fiftyTwoWeekLow")
    high52 = info.get("fiftyTwoWeekHigh")
    if price and low52 and high52 and high52 > low52:
        range_pct = (price - low52) / (high52 - low52)
        if range_pct < 0.10:
            scores.append({"id": "beaten_down", "name": "Near 52W Low", "score": 10,
                           "detail": f"At {range_pct*100:.0f}% of 52W range"})
        elif range_pct < 0.20:
            scores.append({"id": "beaten_down", "name": "Near 52W Low", "score": 7,
                           "detail": f"At {range_pct*100:.0f}% of 52W range"})
        elif range_pct < 0.30:
            scores.append({"id": "beaten_down", "name": "Near 52W Low", "score": 4,
                           "detail": f"At {range_pct*100:.0f}% of 52W range"})

    # 6. Hot sector (low weight — nice to have, not a strong predictor)
    sector = info.get("sector", "")
    industry = (info.get("industry") or "").lower()
    if sector in HOT_SECTORS or any(kw in industry for kw in HOT_INDUSTRIES):
        scores.append({"id": "hot_sector", "name": "Hot Sector", "score": 2,
                       "detail": f"{sector} / {info.get('industry', '?')}"})

    # 7. Revenue/earnings growth
    rev_growth = info.get("revenueGrowth")
    earn_growth = info.get("earningsGrowth")
    growth_score = 0
    growth_details = []
    if rev_growth and rev_growth > 0.50:
        growth_score += 5
        growth_details.append(f"Rev: +{rev_growth*100:.0f}%")
    elif rev_growth and rev_growth > 0.25:
        growth_score += 3
        growth_details.append(f"Rev: +{rev_growth*100:.0f}%")
    if earn_growth and earn_growth > 0.50:
        growth_score += 5
        growth_details.append(f"Earn: +{earn_growth*100:.0f}%")
    elif earn_growth and earn_growth > 0.25:
        growth_score += 3
        growth_details.append(f"Earn: +{earn_growth*100:.0f}%")
    if growth_score >= 3:
        scores.append({"id": "revenue_growth_signal", "name": "Growth Signal",
                       "score": min(growth_score, 10), "detail": ", ".join(growth_details)})

    # 8. Deep value (trading below book value)
    ptb = info.get("priceToBook")
    if ptb is not None and ptb > 0:
        if ptb < 0.25:
            scores.append({"id": "deep_value", "name": "Deep Value", "score": 10,
                           "detail": f"P/B: {ptb:.2f}"})
        elif ptb < 0.50:
            scores.append({"id": "deep_value", "name": "Deep Value", "score": 7,
                           "detail": f"P/B: {ptb:.2f}"})
        elif ptb < 0.75:
            scores.append({"id": "deep_value", "name": "Deep Value", "score": 4,
                           "detail": f"P/B: {ptb:.2f}"})

    return scores


def fetch_info_safe(ticker: str) -> tuple[str, dict | None]:
    """Fetch yfinance .info for a single ticker (no price history — that
    lives in price_refresher.py and runs daily on a smaller subset)."""
    import yfinance as yf
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info and info.get("symbol"):
            return ticker, info
    except Exception:
        pass
    return ticker, None


# ── Catalyst scoring via Gemini ───────────────────────────────────────

def score_catalysts(client, search_tool, ticker: str, company: str,
                    catalyst_attrs: list[dict]) -> dict | None:
    """Search for recent catalysts for a stock via Gemini search grounding."""
    cat_list = "\n".join(
        f"- {a['id']}: {a['name']} — {a['description']}"
        for a in catalyst_attrs
    )

    prompt = f"""What recent news or events in the past 30 days could be catalysts for {ticker} ({company})?

Score this stock against EACH of these catalyst attributes (0-10, where 0 = not present, 10 = strongly present):

{cat_list}

Return a JSON object with:
- "analysis": brief summary of recent catalysts found (2-3 sentences)
- "catalysts": array of {{"id": "catalyst_id", "name": "catalyst_name", "score": N, "detail": "brief explanation"}} for each catalyst with score > 0

Return ONLY valid JSON, no markdown."""

    from google.genai import types as gtypes

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=config.MODEL_RESEARCH,
                contents=prompt,
                config=gtypes.GenerateContentConfig(tools=[search_tool]),
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
                return json.loads(match.group())
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return None


# ── Main scan pipeline ────────────────────────────────────────────────

def run_scan(exchange: str = "all") -> dict:
    """Run the full market scan pipeline."""

    print(f"\n{'=' * 60}")
    print(f"  MARKET SCAN — {exchange.upper()}")
    print(f"{'=' * 60}\n", flush=True)

    # ── Step 1: Get tickers ───────────────────────────────────────────
    tickers = get_tickers(exchange)
    if not tickers:
        return {"error": "Failed to fetch ticker list"}

    total_tickers = len(tickers)

    # ── Step 2: Fetch .info & score preconditions (sequential) ────────
    print(f"\n  Fetching .info for {len(tickers)} tickers (one at a time)...", flush=True)

    precondition_results = []
    failed = 0

    # Per-call delay for shared/CI IPs (yfinance rate limits more aggressively
    # against datacenter IPs like GitHub Actions runners). Default 0 locally.
    rate_delay = float(os.environ.get("YFINANCE_RATE_DELAY", "0") or 0)

    for i, ticker in enumerate(tickers, 1):
        _ticker, info = fetch_info_safe(ticker)
        if info is None:
            failed += 1
            print(f"    {i}/{len(tickers)} {ticker} — FAILED", flush=True)
        else:
            scores = score_preconditions(info)
            total = sum(s["score"] for s in scores)
            if total > 0:
                precondition_results.append({
                    "ticker": ticker,
                    "company": info.get("shortName") or info.get("longName") or "",
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "market_cap": info.get("marketCap"),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "float_shares": info.get("floatShares"),
                    "short_interest": info.get("shortPercentOfFloat"),
                    "precondition_score": total,
                    "preconditions": scores,
                })
                print(f"    {i}/{len(tickers)} {ticker} — P={total:.0f}", flush=True)
            else:
                print(f"    {i}/{len(tickers)} {ticker} — 0", flush=True)
        if rate_delay > 0:
            time.sleep(rate_delay)

    # Sort by precondition score
    precondition_results.sort(key=lambda x: x["precondition_score"], reverse=True)

    print(f"\n  Precondition scoring complete:")
    print(f"    {len(tickers)} tickers checked, {failed} failed to fetch")
    print(f"    {len(precondition_results)} had any precondition score")

    # ── Step 3: Save ──────────────────────────────────────────────────
    scan = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange,
        "total_tickers": total_tickers,
        "info_fetched": total_tickers - failed,
        "precondition_scored": len(precondition_results),
        "stocks": precondition_results,
    }

    db = load_scan_db()
    db["scans"].insert(0, scan)
    save_scan_db(db)

    print(f"\n  Done. {len(precondition_results)} stocks saved.")
    if precondition_results:
        print(f"\n  Top 10 by precondition score:")
        for i, s in enumerate(precondition_results[:10], 1):
            attrs = ", ".join(f"{a['name']}={a['score']}" for a in s["preconditions"][:4])
            print(f"    {i:2d}. {s['ticker']:6s}  P={s['precondition_score']:.0f}  [{attrs}]")

    return scan
