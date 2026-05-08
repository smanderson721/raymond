#!/usr/bin/env python3
"""
Daily News Monitor — Finnhub + EDGAR filings.

Fetches news from Finnhub and SEC filings (Form 3/4/5 insider + 8-K/6-K
material events) from EDGAR for all US-listed tickers. Classifies articles
by catalyst type using keyword matching. Outputs a per-ticker news score.

Usage:
    python pipeline.py --news-scan
    python pipeline.py --news-scan --exchange nasdaq --news-days 30
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research_output",
)
NEWS_FILE = os.path.join(OUTPUT_DIR, "news_data.json")

# ── SEC EDGAR config ──────────────────────────────────────────────────
EDGAR_USER_AGENT = "StockMonitor/1.0 contact@example.com"
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# ── Finnhub config ────────────────────────────────────────────────────
FINNHUB_BASE = "https://finnhub.io/api/v1"

# ── Generic/filler headline patterns to skip ──────────────────────────
GENERIC_PATTERNS = [
    r"stocks moving",
    r"top gainers",
    r"top losers",
    r"after-market session",
    r"pre-market session",
    r"intraday session",
    r"after-hours session",
    r"notable movement",
    r"unusual volume",
    r"notable gaps",
    r"what'?s going on in today",
    r"keep an eye on",
    r"let'?s uncover",
    r"let'?s take a look",
    r"wondering what",
    r"which stocks are experiencing",
    r"get insights into the top",
    r"these stocks are experiencing",
    r"trading volume of these",
    r"in today'?s session",
    r"^\d+ \w+ stocks moving",
    r"here are the top movers",
    r"top movers in \w+'?s session",
    r"curious about the stocks",
    r"these stocks that are showing",
]
_GENERIC_RE = re.compile("|".join(GENERIC_PATTERNS), re.IGNORECASE)

# ── Catalyst keyword classification ───────────────────────────────────
# Each catalyst type has a list of keyword patterns. A headline/summary
# matching any pattern is tagged with that catalyst.
CATALYST_KEYWORDS = {
    "earnings_beat": [
        r"q[1-4]\s+eps", r"beats?\s+\$?\d", r"earnings", r"revenue\s+(growth|beat|surge)",
        r"sales\s+\$[\d.]+[mb]\s+beat", r"profit", r"ebitda", r"financial results",
        r"quarterly results", r"annual results",
    ],
    "merger_acquisition": [
        r"merg(er|ing|es)", r"acqui(re|sition|ring)", r"take-?over", r"buyout",
        r"buy(s|ing) .{0,30}for \$", r"definitive agreement", r"combination",
        r"house of doge",  # TBH-specific but illustrative
    ],
    "partnership_contract": [
        r"partner(ship|s|ing)", r"\bmou\b", r"memorandum of understanding",
        r"contract\b", r"agreement\b.{0,20}(with|between)", r"collaboration",
        r"joint (venture|development)", r"strategic (alliance|deal)",
        r"awarded\s+\$", r"wins?\s+(contract|\$)",
    ],
    "regulatory_fda": [
        r"\bfda\b", r"approval", r"clearance", r"510\(?k\)?", r"phase\s+[i123]",
        r"clinical trial", r"ce mark", r"gmp.compliant", r"executive order",
        r"regulatory milestone", r"pdufa", r"nda\b", r"eua\b",
    ],
    "debt_restructuring": [
        r"debt (restructur|reduc|repay|clear)", r"convertible (note|debt).{0,30}(repay|clear|settl)",
        r"principal haircut", r"exchange offer", r"tender offer",
        r"refinanc", r"credit facility",
    ],
    "compliance_listing": [
        r"nasdaq (compliance|deficiency|listing|non-?compliance)",
        r"regain(s|ed)?\s+compliance", r"listing (standard|rule|requirement)",
        r"bid price (require|compliance)", r"stockholders.{0,20}equity",
        r"trading (resump|halt|suspend)",
    ],
    "insider_filing": [
        r"form [345]", r"insider (buy|purchase|acqui|filing|disclosure)",
        r"director.{0,30}(shares|stake|holding)", r"schedule 13d",
        r"beneficial ownership", r"open market purchase",
    ],
    "strategic_pivot": [
        r"pivot", r"rebrand", r"transformation", r"new (strategy|direction|segment)",
        r"enters?\s+(new|into)", r"expan(d|sion)", r"(exit|divest)\w*\s+.{0,30}(business|segment)",
        r"asset.light", r"restructur",
    ],
    "leadership_change": [
        r"(appoint|name[sd]?|hire[sd]?)\s+.{0,20}(ceo|cfo|coo|cto|director|officer|board)",
        r"board (reshuffle|change|overhaul|shakeup)", r"activist",
        r"(resign|depart)\w*\s+.{0,20}(ceo|cfo|director|officer)",
        r"cooperation agreement",
    ],
    "legal_litigation": [
        r"(lawsuit|litigation|ruling|verdict|settlement|patent).{0,30}(win|victory|favor|upheld|award)",
        r"supreme court", r"court\s+rul(e|ing)", r"patent\s+(valid|infring)",
        r"damages\s+\$",
    ],
    "offering_capital": [
        r"equity (facility|offering|raise)", r"capital (raise|infusion|injection)",
        r"\$\d+\s*m(illion)?\s+(equity|convertible|note|facility)",
        r"public offering", r"private placement", r"shelf registration",
    ],
    "sector_macro": [
        r"oil (price|surge|spike|rally)", r"ceasefire", r"geopoliti",
        r"strait of hormuz", r"iran", r"tariff", r"trade (war|deal)",
        r"interest rate", r"fed (cut|hike|pause)", r"sector\s+rall",
        r"defense (contract|spend)", r"pentagon",
    ],
    "short_squeeze": [
        r"short (squeeze|interest|cover)", r"si\s+(drop|plung|surg)",
        r"days to cover", r"borrow (fee|rate|cost)",
    ],
    "analyst_coverage": [
        r"(initiat|upgrad|downgrad|reiter)\w*\s+.{0,20}(coverage|rating|buy|sell|overweight)",
        r"price target\s+\$", r"analyst",
    ],
    "social_momentum": [
        r"meme stock", r"\breddit\b", r"wallstreetbets", r"\bwsb\b",
        r"discord", r"retail (trad|buy|interest|hype)", r"fomo",
        r"stocktwits", r"social media",
    ],
}
_CATALYST_RES = {
    cat: re.compile("|".join(patterns), re.IGNORECASE)
    for cat, patterns in CATALYST_KEYWORDS.items()
}

# ── Catalyst score weights ────────────────────────────────────────────
# Some catalyst types are stronger signals than others.
CATALYST_WEIGHTS = {
    "earnings_beat": 3,
    "merger_acquisition": 5,
    "partnership_contract": 3,
    "regulatory_fda": 4,
    "debt_restructuring": 3,
    "compliance_listing": 2,
    "insider_filing": 3,
    "strategic_pivot": 3,
    "leadership_change": 2,
    "legal_litigation": 4,
    "offering_capital": 2,
    "sector_macro": 1,
    "short_squeeze": 2,
    "analyst_coverage": 2,
    "social_momentum": 2,
}

# Points for article volume (quantity signal like CAR)
VOLUME_THRESHOLDS = [
    (50, 5),   # 50+ articles = 5 bonus points
    (20, 3),   # 20+ = 3
    (10, 2),   # 10+ = 2
    (5, 1),    # 5+  = 1
]


# ── Load / save ───────────────────────────────────────────────────────

def load_news_db() -> dict:
    if os.path.exists(NEWS_FILE):
        with open(NEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"scans": []}


def save_news_db(data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(NEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── 8-K item code → catalyst type mapping ────────────────────────────
# See https://www.sec.gov/files/form8-k.pdf for item descriptions
_8K_ITEM_CATALYST = {
    "1.01": "partnership_contract",   # Entry into Material Definitive Agreement
    "1.02": "debt_restructuring",     # Termination of Material Definitive Agreement
    "1.03": "compliance_listing",     # Bankruptcy or Receivership
    "2.01": "merger_acquisition",     # Completion of Acquisition or Disposition
    "2.02": "earnings_beat",          # Results of Operations and Financial Condition
    "2.03": "debt_restructuring",     # Creation of Direct Financial Obligation
    "2.04": "debt_restructuring",     # Triggering Events / Default
    "2.05": "debt_restructuring",     # Costs for Exit or Disposal Activities
    "2.06": "debt_restructuring",     # Material Impairments
    "3.01": "compliance_listing",     # Notice of Delisting / Noncompliance
    "3.02": "offering_capital",       # Unregistered Sales of Equity Securities
    "3.03": "offering_capital",       # Material Modification to Rights of Holders
    "4.01": "earnings_beat",          # Changes in Registrant's Certifying Accountant
    "4.02": "earnings_beat",          # Non-Reliance on Previously Issued Financials
    "5.01": "strategic_pivot",        # Changes in Control of Registrant
    "5.02": "leadership_change",      # Departure/Election of Directors/Officers
    "5.03": "strategic_pivot",        # Amendments to Articles of Incorporation
    "5.07": "leadership_change",      # Submission of Matters to Vote of Holders
    "7.01": "strategic_pivot",        # Regulation FD Disclosure (often press releases)
    "8.01": "strategic_pivot",        # Other Events
}


# ── EDGAR: Fetch all SEC filings (3/4/5 + 8-K + 6-K) ────────────────

_TICKER_FROM_NAME_RE = re.compile(r"\(([A-Z][A-Z0-9]{0,5}(?:,\s*[A-Z][A-Z0-9-]{0,5})*)\)")


def _extract_tickers_from_display_name(name: str) -> list[str]:
    """Extract ticker(s) from EDGAR display_name.
    Format: 'Company Name  (TICK)  (CIK 0001234567)'
    or:     'Company Name  (TICK, TICK-A)  (CIK 0001234567)'"""
    matches = _TICKER_FROM_NAME_RE.findall(name)
    tickers = []
    for m in matches:
        if m.startswith("CIK"):
            continue
        for t in m.split(","):
            t = t.strip()
            if t and not t.startswith("CIK"):
                tickers.append(t.upper())
    return tickers


def fetch_edgar_filings(
    tickers: list[str],
    days: int = 30,
    forms: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Fetch recent SEC filings from EDGAR for given tickers.
    Fetches Form 3/4/5 (insider), 8-K (material events), 6-K (foreign issuer).
    Pass forms=["8-K","6-K"] to skip slow Form 3/4/5 pagination.
    Returns {ticker: [filing_dict, ...]}."""

    results: dict[str, list[dict]] = {}
    ticker_set = set(t.upper() for t in tickers)

    # Default: all forms. Caller can pass ["8-K","6-K"] to skip slow 3/4/5.
    if forms is None:
        form_groups = ["3,4,5", "8-K,6-K"]
        label = "3/4/5 + 8-K + 6-K"
    else:
        form_groups = [",".join(forms)]
        label = ",".join(forms)

    print(f"  Fetching EDGAR filings ({label}, last {days} days)...", flush=True)

    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    total_filings = 0
    counts_by_form = {}

    for form_query in form_groups:
        page_from = 0
        page_size = 100
        retries = 0

        while True:
            params = (
                f"?forms={form_query}"
                f"&dateRange=custom&startdt={start_date}&enddt={end_date}"
                f"&from={page_from}&size={page_size}"
            )
            url = EDGAR_EFTS_URL + params
            req = urllib.request.Request(url, headers={"User-Agent": EDGAR_USER_AGENT})

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                retries = 0  # Reset on success
            except Exception as e:
                retries += 1
                if retries <= 2:
                    time.sleep(2)
                    continue
                print(f"    EDGAR EFTS error on {form_query} page {page_from}: {e}", flush=True)
                break

            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {}).get("value", 0)

            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                names = src.get("display_names", [])
                file_date = src.get("file_date", "")
                form_type = src.get("form", "") or src.get("form_type", "")
                items = src.get("items", [])
                file_desc = src.get("file_description", "")

                # Extract ticker(s) from display_names
                for name in names:
                    matched_tickers = _extract_tickers_from_display_name(name)
                    for t in matched_tickers:
                        if t not in ticker_set:
                            continue
                        if t not in results:
                            results[t] = []

                        filing = {
                            "form": form_type,
                            "date": file_date,
                            "filer": name.split("(")[0].strip().rstrip(","),
                            "source": "EDGAR",
                        }

                        # Add 8-K item codes and their catalyst mappings
                        if items:
                            filing["items"] = items
                            catalysts = []
                            for item in items:
                                cat = _8K_ITEM_CATALYST.get(item)
                                if cat and cat not in catalysts:
                                    catalysts.append(cat)
                            if catalysts:
                                filing["catalysts"] = catalysts

                        # Add file description for 6-K (often "PRESS RELEASE" etc.)
                        if file_desc and file_desc not in ("FORM 8-K", "8-K", "6-K", "FORM 6-K"):
                            filing["description"] = file_desc

                        results[t].append(filing)
                        total_filings += 1
                        counts_by_form[form_type] = counts_by_form.get(form_type, 0) + 1

            page_from += page_size
            if page_from >= total or page_from >= 10000:  # SEC caps at 10k
                break

            time.sleep(0.2)  # Be polite to EDGAR

    form_summary = ", ".join(f"{f}:{c}" for f, c in sorted(counts_by_form.items()))
    print(f"    Found {total_filings} filings across {len(results)} tickers ({form_summary})", flush=True)
    return results


# ── Finnhub: Fetch company news ──────────────────────────────────────
def prefilter_finnhub_tickers(
    tickers: list[str],
    scan_stocks: dict[str, dict],
    edgar_data: dict[str, list[dict]] | None = None,
    max_cap: float | None = 10e9,
    min_precondition_total: float = 1.0,
) -> list[str]:
    """Reduce the Finnhub fetch list to tickers worth the 1.05s/req cost.

    Rules:
      • Drop tickers whose market_cap exceeds max_cap (mega/large caps
        produce mostly generic noise that gets filtered later anyway).
      • Drop tickers with precondition_total below the threshold (no
        technical/fundamental setup → catalyst is unlikely to matter).
      • ALWAYS include tickers that already have EDGAR 8-K/6-K data
        (an SEC material event → we want the Finnhub color regardless
        of cap/precondition score).
      • Tickers absent from scan_stocks (no precondition data) are
        dropped UNLESS they're in the EDGAR keep set.

    Returns the filtered ticker list, preserving original order.
    """
    edgar_keep: set[str] = set()
    if edgar_data:
        for t, filings in edgar_data.items():
            if any(
                f.get("form") in ("8-K", "6-K", "8-K/A", "6-K/A")
                for f in filings
            ):
                edgar_keep.add(t)

    kept: list[str] = []
    skipped_cap = 0
    skipped_pre = 0
    skipped_no_data = 0
    for t in tickers:
        if t in edgar_keep:
            kept.append(t)
            continue
        info = scan_stocks.get(t)
        if not info:
            skipped_no_data += 1
            continue
        mcap = info.get("market_cap")
        if max_cap and mcap and mcap > max_cap:
            skipped_cap += 1
            continue
        # precondition_total is the sum of attribute scores (yfinance scan)
        ptot = info.get("precondition_total")
        if ptot is None:
            ptot = info.get("total_score") or info.get("precondition_score") or 0
        if (ptot or 0) < min_precondition_total:
            skipped_pre += 1
            continue
        kept.append(t)

    print(
        f"  Finnhub prefilter: {len(tickers)} → {len(kept)} tickers "
        f"(cap-skip {skipped_cap}, pre-skip {skipped_pre}, "
        f"no-scan {skipped_no_data}, edgar-keep {len(edgar_keep)})",
        flush=True,
    )
    return kept

def fetch_finnhub_news(tickers: list[str], api_key: str, days: int = 1) -> dict[str, list[dict]]:
    """Fetch Finnhub company news for all tickers.
    Returns {ticker: [article_dict, ...]}."""

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results: dict[str, list[dict]] = {}
    total_articles = 0
    request_count = 0

    print(f"  Fetching Finnhub news for {len(tickers)} tickers ({days}d window)...", flush=True)

    for i, ticker in enumerate(tickers, 1):
        url = (
            f"{FINNHUB_BASE}/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                articles = json.loads(resp.read())

            if articles:
                results[ticker] = articles
                total_articles += len(articles)

            request_count += 1

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — wait and retry
                print(f"\n    Rate limited at {ticker}, waiting 60s...", flush=True)
                time.sleep(60)
                try:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        articles = json.loads(resp.read())
                    if articles:
                        results[ticker] = articles
                        total_articles += len(articles)
                except Exception:
                    pass
            else:
                pass  # Skip failed tickers silently
        except Exception:
            pass

        # Rate limit: Finnhub free tier = 60 req/min
        # Use 1.05s delay to stay safely under
        if request_count % 60 == 0:
            time.sleep(1.0)
        else:
            time.sleep(1.05)

        # Progress every 100 tickers
        if i % 100 == 0:
            elapsed_min = (request_count * 1.05) / 60
            eta_min = ((len(tickers) - i) * 1.05) / 60
            print(f"    {i}/{len(tickers)} ({total_articles} articles, ~{eta_min:.0f}min left)", flush=True)

    print(f"    Done: {total_articles} articles across {len(results)} tickers", flush=True)
    return results


# ── Classify articles ─────────────────────────────────────────────────

def classify_article(headline: str, summary: str = "") -> list[str]:
    """Classify a single article into catalyst categories via keyword matching.
    Returns list of matching catalyst IDs."""
    text = f"{headline} {summary}".strip()
    if not text:
        return []

    matched = []
    for cat_id, pattern in _CATALYST_RES.items():
        if pattern.search(text):
            matched.append(cat_id)
    return matched


def is_generic(headline: str) -> bool:
    """Check if a headline is a generic 'stocks moving' filler."""
    return bool(_GENERIC_RE.search(headline))


# ── Score a single ticker ─────────────────────────────────────────────

def score_ticker_news(
    ticker: str,
    finnhub_articles: list[dict],
    edgar_filings: list[dict],
) -> dict:
    """Score a ticker based on its Finnhub articles + EDGAR filings.

    Returns:
        {
            "news_score": int,
            "article_count": int,
            "real_article_count": int,
            "catalyst_tags": [{"id": str, "name": str, "score": int, "count": int, "detail": str}],
            "top_headlines": [str],
            "edgar_filings": [{"form": str, "date": str, "filer": str}],
        }
    """
    # Separate real vs generic articles
    real_articles = []
    for a in finnhub_articles:
        h = a.get("headline", "")
        if h and not is_generic(h):
            real_articles.append(a)

    # Classify all real articles
    catalyst_counts: dict[str, int] = {}
    catalyst_details: dict[str, str] = {}  # first matching headline per catalyst
    for a in real_articles:
        h = a.get("headline", "")
        s = a.get("summary", "")
        cats = classify_article(h, s)
        for c in cats:
            catalyst_counts[c] = catalyst_counts.get(c, 0) + 1
            if c not in catalyst_details:
                catalyst_details[c] = h[:100]

    # Add EDGAR filings as catalyst signals
    for f in edgar_filings:
        form = f.get("form", "")
        detail_str = f"{form} by {f.get('filer','?')} ({f.get('date','')})"
        desc = f.get("description", "")
        if desc:
            detail_str += f" — {desc}"

        if form in ("3", "4", "5"):
            # Insider filings
            catalyst_counts["insider_filing"] = catalyst_counts.get("insider_filing", 0) + 1
            if "insider_filing" not in catalyst_details:
                catalyst_details["insider_filing"] = detail_str
        elif form in ("8-K", "6-K"):
            # Material event filings — use item-derived catalysts if available
            catalysts = f.get("catalysts", [])
            if catalysts:
                for cat in catalysts:
                    catalyst_counts[cat] = catalyst_counts.get(cat, 0) + 1
                    if cat not in catalyst_details:
                        items_str = ",".join(f.get("items", []))
                        catalyst_details[cat] = f"{form} [{items_str}] {detail_str}"
            else:
                # 6-K or 8-K without recognized items — tag as strategic_pivot
                cat = "strategic_pivot"
                catalyst_counts[cat] = catalyst_counts.get(cat, 0) + 1
                if cat not in catalyst_details:
                    catalyst_details[cat] = detail_str

    # Calculate score
    # 1. Catalyst type scores (weighted by importance × count, capped per type)
    catalyst_score = 0
    catalyst_tags = []
    for cat_id, count in catalyst_counts.items():
        weight = CATALYST_WEIGHTS.get(cat_id, 1)
        # Diminishing returns: first article = full weight, subsequent = +1 each, cap at weight*2
        type_score = min(weight + (count - 1), weight * 2)
        catalyst_score += type_score
        catalyst_tags.append({
            "id": cat_id,
            "name": cat_id.replace("_", " ").title(),
            "score": type_score,
            "count": count,
            "detail": catalyst_details.get(cat_id, ""),
        })

    # 2. Volume bonus (high article count = more attention)
    volume_bonus = 0
    for threshold, bonus in VOLUME_THRESHOLDS:
        if len(real_articles) >= threshold:
            volume_bonus = bonus
            break

    total_score = catalyst_score + volume_bonus

    # Sort catalyst tags by score descending
    catalyst_tags.sort(key=lambda x: x["score"], reverse=True)

    # Top headlines (first 5 real articles, chronological)
    sorted_articles = sorted(real_articles, key=lambda a: a.get("datetime", 0))
    top_headlines = []
    for a in sorted_articles[:5]:
        dt = datetime.fromtimestamp(a.get("datetime", 0)).strftime("%Y-%m-%d")
        top_headlines.append(f"[{dt}] {a.get('headline', '')[:100]}")

    return {
        "news_score": total_score,
        "article_count": len(finnhub_articles),
        "real_article_count": len(real_articles),
        "volume_bonus": volume_bonus,
        "catalyst_tags": catalyst_tags,
        "top_headlines": top_headlines,
        "edgar_filings": [
            {"form": f["form"], "date": f["date"], "filer": f.get("filer", "")}
            for f in edgar_filings[:5]
        ],
    }


# ── Main scan ─────────────────────────────────────────────────────────

def run_news_scan(
    tickers: list[str],
    finnhub_key: str,
    days: int = 30,
) -> dict:
    """Run a full news scan for all tickers.

    Args:
        tickers: list of ticker symbols
        finnhub_key: Finnhub API key
        days: lookback window in days

    Returns:
        scan result dict with per-ticker news scores
    """
    print(f"\n{'=' * 60}")
    print(f"  NEWS SCAN — {len(tickers)} tickers, {days}-day window")
    print(f"{'=' * 60}\n", flush=True)

    start_time = time.time()

    # 1. Fetch Finnhub news
    finnhub_data = fetch_finnhub_news(tickers, finnhub_key, days=days)

    # 2. Fetch EDGAR filings (insider + 8-K + 6-K)
    edgar_data = fetch_edgar_filings(tickers, days=days)

    # 3. Score each ticker
    print(f"\n  Scoring {len(tickers)} tickers...", flush=True)
    scores = {}
    for ticker in tickers:
        fh_articles = finnhub_data.get(ticker, [])
        ed_filings = edgar_data.get(ticker, [])

        if not fh_articles and not ed_filings:
            continue  # No data — skip entirely

        result = score_ticker_news(ticker, fh_articles, ed_filings)
        if result["news_score"] > 0:
            scores[ticker] = result

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1]["news_score"], reverse=True)

    elapsed = time.time() - start_time

    print(f"\n  News scan complete in {elapsed/60:.1f} min")
    print(f"    {len(finnhub_data)} tickers had Finnhub articles")
    print(f"    {len(edgar_data)} tickers had EDGAR filings")
    print(f"    {len(scores)} tickers scored > 0\n")

    if ranked:
        print(f"  Top 20 by news score:")
        for i, (ticker, data) in enumerate(ranked[:20], 1):
            cats = ", ".join(t["name"] for t in data["catalyst_tags"][:3])
            print(f"    {i:3d}. {ticker:6s}  score={data['news_score']:3d}  "
                  f"articles={data['real_article_count']:3d}  [{cats}]")

    # Build scan record
    scan = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "total_tickers": len(tickers),
        "tickers_with_news": len(scores),
        "elapsed_minutes": round(elapsed / 60, 1),
        "scores": {ticker: data for ticker, data in ranked},
    }

    # Save
    db = load_news_db()
    db["scans"].insert(0, scan)
    # Keep only last 10 scans
    db["scans"] = db["scans"][:10]
    save_news_db(db)

    print(f"  Saved to {NEWS_FILE}")
    return scan
