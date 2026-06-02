"""
Episodic memory for the trading agent.

Wraps the FTS5 index in journal.py to surface the most relevant past
decisions (with their outcomes, if any) for a given ticker + context.
Used by decide.py to enrich the Gemini prompt.

Keep this dependency-free: just sqlite + journal. No embeddings, no
external services. FTS5 is fast enough for tens of thousands of rows.
"""

from __future__ import annotations

import json
from typing import Any

from agent import journal


def _context_to_query(ctx: dict) -> str:
    """Turn a context dict into FTS query text (catalyst types + sectors)."""
    bits: list[str] = []
    pre = ctx.get("preconditions") or {}
    if pre.get("sector"):   bits.append(str(pre["sector"]))
    if pre.get("industry"): bits.append(str(pre["industry"]))
    cat = ctx.get("catalyst") or {}
    if cat.get("blurb"):    bits.append(str(cat["blurb"]))
    for c in (cat.get("catalysts") or []):
        if isinstance(c, dict):
            if c.get("name"):   bits.append(str(c["name"]))
            if c.get("detail"): bits.append(str(c["detail"]))
    for h in (ctx.get("recent_hits") or []):
        if isinstance(h, dict) and h.get("reason"):
            bits.append(str(h["reason"]))
    return " ".join(bits)


def find_similar(ticker: str, ctx: dict, limit: int = 4) -> list[dict]:
    """Return up to `limit` past decisions whose rationale matches the
    current context. Excludes prior decisions for the same ticker (those
    are surfaced separately via own_history)."""
    q = _context_to_query(ctx)
    rows = journal.find_similar_decisions(q, limit=limit,
                                          exclude_ticker=ticker)
    return [_decorate(r) for r in rows]


def own_history(ticker: str, limit: int = 3) -> list[dict]:
    """All prior decisions for the same ticker (most recent first)."""
    with journal.connect() as c:
        rows = c.execute(
            "SELECT * FROM decisions WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (ticker.upper(), limit),
        ).fetchall()
    return [_decorate(dict(r)) for r in rows]


def _decorate(row: dict) -> dict:
    out = journal.get_outcome_for(row["id"])
    return {
        "ticker":    row.get("ticker"),
        "action":    row.get("action"),
        "size_usd":  row.get("size_usd"),
        "bup":       row.get("bup_score"),
        "pct_30d":   row.get("pct_30d"),
        "rationale": (row.get("rationale") or "")[:280],
        "ts":        row.get("ts"),
        "outcome": (
            {
                "pnl_pct":     out.get("pnl_pct"),
                "exit_reason": out.get("exit_reason"),
            } if out else None
        ),
    }


def episode_block(ticker: str, ctx: dict) -> dict:
    """Compact block ready to inline into the Gemini prompt."""
    return {
        "own_history":     own_history(ticker, 3),
        "similar_setups":  find_similar(ticker, ctx, 4),
    }
