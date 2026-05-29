"""
News Catalyst scan — Alpaca News REST poll + Gemini classification.

Pulls every article published in a short lookback window from the Alpaca
News API (Benzinga-powered), deduplicates against a rolling seen-id
cache, then classifies each article with Gemini Flash Lite. The model
returns:

  - ``known_catalysts``  — entries whose ``id`` matches the predefined
    catalog of catalyst attributes (e.g. ``earnings_beat``,
    ``merger_acquisition``). These award points using the live-editable
    ``scan_weights.json`` rewards.

  - ``proposed_catalysts`` — Gemini-invented catalysts that don't fit any
    known id. These are auto-registered in ``scan_weights.json`` with
    ``status: "proposed"`` and a default reward (so the signal isn't
    lost while the operator triages); they show up in the Attributes
    subtab with Accept / Rename / Reject buttons.

Every catalyst returned by Gemini must include an ``evidence_quote`` that
is a literal substring of the article body. Quotes that don't match are
discarded before any points are awarded — this kills the bulk of LLM
hallucinations.

Cadence: every 10 minutes via ``live_daemon.py``. At ~500 small/mid-cap
articles/day and ~50 input tokens of prompt scaffolding + ~800 tokens of
article body, the Gemini cost runs about $3-5/month.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from research.live_score_engine import Session
from research.scan_weights import all_weights, write_weights, weight

# ── paths ─────────────────────────────────────────────────────────────
SCAN_NAME = "news_catalyst"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE_DIR = os.path.join(REPO_ROOT, "data", "live")
SEEN_FILE = os.path.join(LIVE_DIR, "_news_catalyst_seen.json")
POOL_FILE = os.path.join(LIVE_DIR, "_news_catalyst_pool.json")
SCAN_RESULTS_FILE = os.path.join(REPO_ROOT, "research_output", "scan_results.json")

# ── tuning ────────────────────────────────────────────────────────────
LOOKBACK_MIN = 15                  # window slightly larger than scan cadence
MAX_CAP = 10e9                     # filter out mega-caps (configurable later)
MAX_ARTICLES_PER_RUN = 80          # safety cap on Gemini calls per cycle
ARTICLE_BODY_MAX_CHARS = 4000      # truncate very long bodies before prompting
SEEN_TTL_SEC = 48 * 3600           # dedup memory horizon
DEFAULT_PROPOSED_POINTS = 2.0      # starting reward for an unrecognized catalyst
SIMILARITY_MERGE = 0.65            # if proposed key is this similar to an existing id, merge

# Alpaca News
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

# ── default catalyst catalog ──────────────────────────────────────────
# Each entry: (id, default_points, description). The scan auto-seeds these
# into ``scan_weights.json`` on first run if missing, then reads the live
# (operator-editable) values on every subsequent run.
DEFAULT_CATALYSTS: list[tuple[str, float, str]] = [
    ("news_earnings_beat",        3.0, "Quarterly results, EPS or revenue beat, guidance raise"),
    ("news_merger_acquisition",   5.0, "M&A, takeover, buyout, definitive agreement to combine"),
    ("news_partnership_contract", 3.0, "New customer contract, partnership, MOU, JV, distribution deal"),
    ("news_regulatory_fda",       4.0, "FDA approval/clearance, CE mark, clinical-trial milestone, PDUFA, regulatory decision"),
    ("news_debt_restructuring",   3.0, "Debt restructure, refinancing, tender offer, convertible cleanup, covenant waiver"),
    ("news_compliance_listing",   2.0, "Regained Nasdaq/NYSE listing compliance, bid-price recovery, reverse split"),
    ("news_insider_filing",       3.0, "Insider open-market purchase, 13D/13G filing, beneficial-ownership disclosure"),
    ("news_strategic_pivot",      3.0, "Major strategy shift, new business segment, transformation, restructuring"),
    ("news_leadership_change",    2.0, "CEO/CFO/board appointment or departure, activist involvement, cooperation agreement"),
    ("news_legal_litigation",     4.0, "Lawsuit win, favorable ruling, settlement, patent verdict, court decision"),
    ("news_offering_capital",     2.0, "Equity offering, capital raise, public offering, ATM, shelf registration, PIPE"),
    ("news_sector_macro",         1.0, "Sector-wide event: commodity move, geopolitics, tariff, rate decision affecting the issuer's sector"),
    ("news_short_squeeze",        2.0, "Short-interest spike/collapse, days-to-cover surge, borrow-fee move, squeeze chatter"),
    ("news_analyst_coverage",     2.0, "Analyst initiation, upgrade, downgrade, price-target revision"),
    ("news_social_momentum",      2.0, "Reddit/WSB/Stocktwits mention, viral retail interest, FOMO chatter"),
]
DEFAULT_CATALYST_IDS = {c[0] for c in DEFAULT_CATALYSTS}


# ──────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────
def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def _load_seen() -> dict:
    """Return {article_id: ts}. Prune entries older than SEEN_TTL_SEC."""
    blob = _read_json(SEEN_FILE, {"ids": {}})
    now = time.time()
    ids = {k: v for k, v in blob.get("ids", {}).items() if (now - float(v or 0)) < SEEN_TTL_SEC}
    return ids


def _save_seen(ids: dict) -> None:
    _write_json(SEEN_FILE, {"ids": ids, "updated_at": _now_iso()})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_mcap_map() -> dict[str, float]:
    """Pull the most recent ``scan_results.json`` and return ticker→market_cap.
    Missing file or empty scan returns ``{}`` — the scan then processes
    every article (no cap filter), which is safe but pricier."""
    d = _read_json(SCAN_RESULTS_FILE, {})
    scans = d.get("scans", []) if isinstance(d, dict) else []
    if not scans:
        return {}
    out: dict[str, float] = {}
    for s in scans[0].get("stocks", []):
        tk = (s.get("ticker") or "").upper()
        mc = s.get("market_cap")
        if tk and isinstance(mc, (int, float)) and mc > 0:
            out[tk] = float(mc)
    return out


# ──────────────────────────────────────────────────────────────────────
# Scan-weights catalyst catalog management
# ──────────────────────────────────────────────────────────────────────
def _ensure_default_catalysts() -> None:
    """Make sure every entry in DEFAULT_CATALYSTS exists in
    ``scan_weights.json`` under ``scans.news_catalyst.attributes``. Existing
    keys (including operator-tuned points) are preserved."""
    data = all_weights() or {"scans": {}}
    data.setdefault("scans", {})
    scan_block = data["scans"].setdefault(SCAN_NAME, {
        "label": "News Catalyst",
        "live_editable": True,
        "attributes": {},
    })
    scan_block.setdefault("attributes", {})
    attrs = scan_block["attributes"]
    changed = False
    for key, pts, desc in DEFAULT_CATALYSTS:
        if key not in attrs:
            attrs[key] = {
                "points": float(pts),
                "description": desc,
                "status": "active",
            }
            changed = True
        else:
            # ensure status field exists for legacy entries
            if "status" not in attrs[key]:
                attrs[key]["status"] = "active"
                changed = True
    if changed:
        write_weights(data)


def _active_known_ids() -> list[str]:
    """All catalyst ids the LLM may legally return in ``known_catalysts``.
    Active = default + any operator-accepted proposals; excludes rejected."""
    data = all_weights() or {}
    attrs = (((data.get("scans") or {}).get(SCAN_NAME) or {}).get("attributes") or {})
    out = []
    for k, v in attrs.items():
        status = (v or {}).get("status", "active")
        if status == "active":
            out.append(k)
    return sorted(out)


def _rejected_ids() -> list[str]:
    data = all_weights() or {}
    attrs = (((data.get("scans") or {}).get(SCAN_NAME) or {}).get("attributes") or {})
    return sorted([k for k, v in attrs.items() if (v or {}).get("status") == "rejected"])


def _trigram_set(s: str) -> set[str]:
    s = re.sub(r"[^a-z0-9]", " ", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    pad = f"  {s}  "
    return {pad[i:i + 3] for i in range(len(pad) - 2)}


def _similarity(a: str, b: str) -> float:
    """Trigram Jaccard, 0-1."""
    A, B = _trigram_set(a), _trigram_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _register_proposed(prop_id: str, prop_name: str, prop_desc: str,
                       evidence_quote: str, article_id: str,
                       article_headline: str) -> tuple[str, bool]:
    """Add ``prop_id`` to ``scan_weights.json`` as a proposed catalyst.

    If a similar id already exists (proposed or rejected) merge into it;
    rejected ids never get re-promoted automatically.

    Returns ``(final_attr_key, was_new)``.
    """
    if not prop_id.startswith("news_"):
        prop_id = f"news_{prop_id}"
    prop_id = re.sub(r"[^a-z0-9_]", "_", prop_id.lower())[:60]

    data = all_weights() or {"scans": {}}
    scan_block = data.setdefault("scans", {}).setdefault(SCAN_NAME, {
        "label": "News Catalyst",
        "live_editable": True,
        "attributes": {},
    })
    attrs = scan_block.setdefault("attributes", {})

    # similarity merge against any existing key (active, proposed, rejected)
    best_key, best_sim = None, 0.0
    for k in attrs.keys():
        sim = _similarity(prop_id, k)
        if sim > best_sim:
            best_sim, best_key = sim, k
    if best_key and best_sim >= SIMILARITY_MERGE:
        attr = attrs[best_key]
        if attr.get("status") == "rejected":
            return best_key, False     # don't re-promote rejected ids
        attr["seen_count"] = int(attr.get("seen_count", 0)) + 1
        examples = attr.get("example_articles", [])
        if article_id and article_id not in [e.get("id") for e in examples]:
            examples.append({"id": article_id, "headline": article_headline[:140]})
            attr["example_articles"] = examples[-5:]
        attr["last_seen"] = _now_iso()
        write_weights(data)
        return best_key, False

    # genuinely new proposal
    attrs[prop_id] = {
        "points": DEFAULT_PROPOSED_POINTS,
        "description": prop_desc[:240] or prop_name[:240] or "Gemini-proposed catalyst",
        "status": "proposed",
        "proposed_name": prop_name[:80] or prop_id.replace("_", " ").title(),
        "first_seen": _now_iso(),
        "last_seen": _now_iso(),
        "seen_count": 1,
        "example_articles": [{"id": article_id, "headline": article_headline[:140]}] if article_id else [],
        "evidence_quote": evidence_quote[:240],
    }
    write_weights(data)
    return prop_id, True


# ──────────────────────────────────────────────────────────────────────
# Alpaca News fetch
# ──────────────────────────────────────────────────────────────────────
def _alpaca_headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY_ID", "")
    sec = os.environ.get("ALPACA_API_SECRET_KEY", "")
    if not key or not sec:
        return {}
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def _fetch_alpaca_news(start: datetime, end: datetime,
                       page_limit: int = 5) -> list[dict]:
    """REST-page through Alpaca News for the window. Returns raw article dicts.

    Each article looks like::

        {
          "id": 12345,
          "headline": "...",
          "summary": "...",
          "content": "<p>...</p>",     # full body when include_content=true
          "url": "...",
          "symbols": ["AAPL", "TSLA"],
          "created_at": "2026-05-29T18:00:00Z",
          "updated_at": "...",
          "source": "benzinga"
        }
    """
    headers = _alpaca_headers()
    if not headers:
        return []
    # Alpaca wants strict RFC3339 with Z suffix. Force UTC + Z so callers
    # passing naive datetimes (e.g. ad-hoc test scripts) don't 400.
    def _fmt(d):
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "start": _fmt(start),
        "end":   _fmt(end),
        "limit": "50",
        "sort":  "desc",
        "include_content":      "true",
        "exclude_contentless":  "false",   # keep headline-only so we still classify
    }
    out: list[dict] = []
    page_token: Optional[str] = None
    for _ in range(page_limit):
        if page_token:
            params["page_token"] = page_token
        elif "page_token" in params:
            del params["page_token"]
        qs = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{ALPACA_NEWS_URL}?{qs}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode()[:200]
            except Exception:
                err_body = ""
            raise RuntimeError(f"alpaca news HTTP {e.code}: {err_body}") from None
        except Exception as e:
            raise RuntimeError(f"alpaca news fetch failed: {e}") from None
        batch = payload.get("news", []) or []
        out.extend(batch)
        page_token = payload.get("next_page_token")
        if not page_token or not batch:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
# Article body sanitation
# ──────────────────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def _article_text(article: dict) -> str:
    """Concatenate headline + summary + stripped HTML body. Truncate."""
    parts = []
    if article.get("headline"):
        parts.append(article["headline"].strip())
    if article.get("summary"):
        parts.append(article["summary"].strip())
    body = article.get("content") or ""
    if body:
        body = _HTML_TAG_RE.sub(" ", body)
        body = _WS_RE.sub(" ", body)
        body = _NL_RE.sub("\n\n", body).strip()
        parts.append(body)
    text = "\n\n".join(parts).strip()
    if len(text) > ARTICLE_BODY_MAX_CHARS:
        text = text[:ARTICLE_BODY_MAX_CHARS] + "…[truncated]"
    return text


# ──────────────────────────────────────────────────────────────────────
# Gemini classifier
# ──────────────────────────────────────────────────────────────────────
def _build_prompt(article_text: str, symbols: list[str],
                  known_ids: list[str], rejected_ids: list[str]) -> str:
    known_block = "\n".join(f"  - {k}" for k in known_ids)
    rejected_block = (
        "\n".join(f"  - {k}" for k in rejected_ids)
        if rejected_ids else "  (none yet)"
    )
    sym_str = ", ".join(symbols) if symbols else "(none tagged)"
    return f"""You are classifying a single financial news article for trading-catalyst attributes.

ARTICLE SYMBOLS: {sym_str}

ARTICLE TEXT:
\"\"\"
{article_text}
\"\"\"

KNOWN CATALYST IDS (use these whenever a category from this list matches the article):
{known_block}

REJECTED IDS (do NOT propose these; they have been previously rejected by the operator):
{rejected_block}

YOUR TASK
1. Read the article carefully.
2. Identify every distinct catalyst the article describes for the listed symbols.
3. For each catalyst:
   - If it clearly matches one of the KNOWN CATALYST IDS, include it under "known_catalysts" using that exact id.
   - If the article describes a genuinely novel and meaningful catalyst that does NOT fit any known id, invent a short snake_case id (start it with "news_") and include it under "proposed_catalysts" with a human-readable name and one-line description. Use proposed_catalysts SPARINGLY — only for genuinely novel categories, not minor variants of existing ones.
   - In both cases, include an "evidence_quote" — a SHORT LITERAL SUBSTRING (8-180 chars) copied verbatim from the article text above that justifies the catalyst. The quote must be present in the article exactly as you write it (preserve original capitalization and punctuation).
   - Include a "confidence" of "high", "medium", or "low".
4. If the article has no meaningful catalyst (market wrap, price recap, "stocks moving" filler, etc.), return both arrays empty.

OUTPUT FORMAT — return ONLY this JSON object, no markdown, no commentary:
{{
  "known_catalysts": [
    {{ "id": "news_xxx", "evidence_quote": "...", "confidence": "high" }}
  ],
  "proposed_catalysts": [
    {{ "id": "news_xxx",
       "name": "Human Readable Name",
       "description": "One-line definition of the category",
       "evidence_quote": "...",
       "confidence": "high" }}
  ],
  "summary": "≤25 words summarizing the article event(s)."
}}"""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _call_gemini(client, article_text: str, symbols: list[str],
                 known_ids: list[str], rejected_ids: list[str]) -> Optional[dict]:
    prompt = _build_prompt(article_text, symbols, known_ids, rejected_ids)
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=config.MODEL_RESEARCH,
                contents=prompt,
            )
            txt = "".join(
                p.text for p in resp.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            ).strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
            m = _JSON_BLOCK_RE.search(txt)
            if not m:
                continue
            return json.loads(m.group())
        except Exception:
            if attempt == 0:
                time.sleep(1.5)
                continue
            return None
    return None


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _evidence_valid(quote: str, body_norm: str) -> bool:
    q = _norm(quote)
    if len(q) < 8 or len(q) > 240:
        return False
    return q in body_norm


# ──────────────────────────────────────────────────────────────────────
# Main scan
# ──────────────────────────────────────────────────────────────────────
def run() -> dict:
    _ensure_default_catalysts()
    known_ids = _active_known_ids()
    rejected_ids = _rejected_ids()
    seen = _load_seen()
    mcap_map = _load_mcap_map()
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=LOOKBACK_MIN)

    with Session(SCAN_NAME, note="Alpaca News + Gemini classification") as s:
        if not config.GEMINI_API_KEY:
            s.log("GEMINI_API_KEY missing — skipping", level="info")
            return {"ok": False, "reason": "no_gemini_key"}
        if not os.environ.get("ALPACA_API_KEY_ID") or not os.environ.get("ALPACA_API_SECRET_KEY"):
            s.log("ALPACA credentials missing — skipping", level="info")
            return {"ok": False, "reason": "no_alpaca_creds"}

        try:
            articles = _fetch_alpaca_news(start, now)
        except Exception as e:
            s.log(f"alpaca fetch failed: {e}", level="info")
            return {"ok": False, "reason": "fetch_failed", "error": str(e)}

        # Dedup
        fresh = [a for a in articles if str(a.get("id")) not in seen]
        # Limit per-run Gemini spend
        fresh = fresh[:MAX_ARTICLES_PER_RUN]

        if not fresh:
            s.log(f"{len(articles)} fetched · 0 new", level="info")
            return {"ok": True, "fetched": len(articles), "new": 0, "awarded": 0}

        # Lazy import so daemon stays cheap when scan never runs
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)

        classified = 0
        awarded_count = 0
        proposed_new = 0
        article_pool_entries: list[dict] = []
        top_awards: list[tuple[str, str, float]] = []

        for article in fresh:
            article_id = str(article.get("id") or "")
            seen[article_id] = time.time()

            raw_symbols = article.get("symbols") or []
            symbols = sorted({(t or "").upper() for t in raw_symbols if t})

            # filter to non-mega-cap when we have mcap data; tickers with
            # unknown mcap pass through (better to process than miss)
            keep_symbols = []
            for tk in symbols:
                mc = mcap_map.get(tk)
                if mc is None or mc < MAX_CAP:
                    keep_symbols.append(tk)
            if not keep_symbols:
                # nothing actionable on this article — skip the LLM call
                continue

            text = _article_text(article)
            if len(text) < 40:        # nothing for Gemini to read
                continue

            result = _call_gemini(client, text, keep_symbols, known_ids, rejected_ids)
            if not result:
                continue
            classified += 1

            body_norm = _norm(text)
            known_list = result.get("known_catalysts") or []
            proposed_list = result.get("proposed_catalysts") or []
            summary = (result.get("summary") or "").strip()[:200]

            this_article_awards: list[dict] = []

            # ── known catalysts ──
            for c in known_list:
                if not isinstance(c, dict):
                    continue
                cid = (c.get("id") or "").strip()
                if not cid or cid not in known_ids:
                    continue
                if not _evidence_valid(c.get("evidence_quote", ""), body_norm):
                    continue
                pts = weight(SCAN_NAME, cid, 1.0)
                if pts <= 0:
                    continue
                friendly = cid.replace("news_", "").replace("_", " ")
                reason = f"{friendly} · {(c.get('evidence_quote') or '')[:120]}"
                for tk in keep_symbols:
                    s.award(tk, pts, reason, attr_key=cid)
                    awarded_count += 1
                    top_awards.append((tk, friendly, pts))
                this_article_awards.append({
                    "kind": "known", "id": cid, "points": pts,
                    "symbols": keep_symbols, "quote": c.get("evidence_quote", ""),
                })

            # ── proposed catalysts ──
            for c in proposed_list:
                if not isinstance(c, dict):
                    continue
                raw_id = (c.get("id") or "").strip()
                if not raw_id:
                    continue
                if not _evidence_valid(c.get("evidence_quote", ""), body_norm):
                    continue
                final_key, was_new = _register_proposed(
                    raw_id,
                    c.get("name", "") or raw_id,
                    c.get("description", "") or "",
                    c.get("evidence_quote", ""),
                    article_id,
                    article.get("headline", ""),
                )
                if was_new:
                    proposed_new += 1
                # award even when merged; rewards live in scan_weights.json
                pts = weight(SCAN_NAME, final_key, DEFAULT_PROPOSED_POINTS)
                if pts <= 0:
                    continue
                friendly = "[proposed] " + final_key.replace("news_", "").replace("_", " ")
                reason = f"{friendly} · {(c.get('evidence_quote') or '')[:120]}"
                for tk in keep_symbols:
                    s.award(tk, pts, reason, attr_key=final_key)
                    awarded_count += 1
                    top_awards.append((tk, friendly, pts))
                this_article_awards.append({
                    "kind": "proposed", "id": final_key, "points": pts,
                    "symbols": keep_symbols, "quote": c.get("evidence_quote", ""),
                })

            # pool entry for the dashboard
            if this_article_awards:
                article_pool_entries.append({
                    "id": article_id,
                    "headline": article.get("headline", "")[:200],
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "created_at": article.get("created_at", ""),
                    "symbols": keep_symbols,
                    "summary": summary,
                    "awards": this_article_awards,
                })

        # persist seen-cache + rolling pool
        _save_seen(seen)
        if article_pool_entries:
            pool = _read_json(POOL_FILE, {"articles": []})
            combined = (pool.get("articles", []) + article_pool_entries)[-200:]
            _write_json(POOL_FILE, {"updated_at": _now_iso(), "articles": combined})

        # single concise summary log line
        top_awards.sort(key=lambda x: -x[2])
        top_str = ", ".join(f"{tk} {f} +{p:.1f}" for tk, f, p in top_awards[:5])
        s.log(
            f"{len(articles)} fetched · {len(fresh)} new · {classified} classified · "
            f"{awarded_count} awarded · {proposed_new} new proposals"
            + (f" · top: {top_str}" if top_str else ""),
            level="notable" if awarded_count else "info",
        )

        return {
            "ok": True,
            "fetched": len(articles),
            "new": len(fresh),
            "classified": classified,
            "awarded": awarded_count,
            "new_proposals": proposed_new,
        }


if __name__ == "__main__":
    print(run())
