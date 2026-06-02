"""
Trading-agent decision pipeline (v1).

Cron entry point — called every ~5 min during market hours by the live_daemon
scheduler via research/scans/agent_decide.py.

Flow per call:
  1. Read research_output/watchlist_combined.json (BUP top-N list).
  2. Pick tickers we haven't decided on in the last DECISION_COOLDOWN_SEC.
  3. For each, assemble context (scan info, recent news, current position).
  4. Ask Gemini for a JSON action.
  5. Record decision in journal.
  6. If action ∈ {SMALL, NORMAL, CONVICTION} and not dry-run, fire a paper
     market order sized at action.size_pct of buying power, capped at
     MAX_USD_PER_TRADE.

Safety:
  - Live trading requires alpaca_client.ALLOW_LIVE_OVERRIDE = True (a code
    change, not just env) AND env var ALPACA_ALLOW_LIVE=1. Both off ⇒ paper.
  - Per-call limit MAX_DECISIONS_PER_RUN keeps Gemini cost predictable.
  - Per-ticker cooldown DECISION_COOLDOWN_SEC prevents thrash.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

from agent import alpaca_client, episodes, gemini_client, journal  # noqa: E402

WATCHLIST_FILE      = REPO_ROOT / "research_output" / "watchlist_combined.json"
SCAN_RESULTS_FILE   = REPO_ROOT / "research_output" / "scan_results.json"
CATALYST_FILE       = REPO_ROOT / "research_output" / "catalyst_scores.json"
HITS_FILE           = REPO_ROOT / "data" / "live" / "hits.json"

# Tunables
MAX_DECISIONS_PER_RUN  = 3
DECISION_COOLDOWN_SEC  = 6 * 60 * 60     # 6h: don't re-decide same ticker
TOP_N_FROM_WATCHLIST   = 10               # only look at top-10 BUP tickers
MAX_USD_PER_TRADE      = 400.0            # hard cap per single decision
MIN_USD_PER_TRADE      = 25.0             # alpaca min notional ~$1, but be polite
MAX_PCT_30D_FOR_BUY    = 60.0             # if pct_30d above this, force SKIP


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _latest_scan_info() -> dict[str, dict]:
    """Map ticker → latest precondition scan row."""
    sr = _read_json(SCAN_RESULTS_FILE, {})
    scans = sr.get("scans") or []
    if not scans:
        return {}
    last = scans[-1]
    out = {}
    for s in last.get("stocks") or []:
        tk = s.get("ticker")
        if tk:
            out[tk.upper()] = s
    return out


def _latest_catalyst_scores() -> dict[str, dict]:
    cs = _read_json(CATALYST_FILE, {})
    sessions = cs.get("sessions") or []
    if not sessions:
        return {}
    last = sessions[-1]
    return {k.upper(): v for k, v in (last.get("scores") or {}).items()}


def _recent_hits(ticker: str, limit: int = 8) -> list[dict]:
    hits = _read_json(HITS_FILE, {}).get("hits") or []
    return [h for h in hits if (h.get("ticker") or "").upper() == ticker.upper()][-limit:]


def _watchlist_top(n: int) -> list[dict]:
    wl = _read_json(WATCHLIST_FILE, {})
    return (wl.get("tickers") or [])[:n]


def _build_context(ticker: str, wl_row: dict,
                   scan_info: dict, catalyst: dict,
                   position: dict | None) -> dict:
    return {
        "ticker":      ticker,
        "watchlist": {
            "best_bup":      wl_row.get("best_bup"),
            "best_bup_date": wl_row.get("best_bup_date"),
            "days_in_list":  wl_row.get("days_in_list"),
            "first_seen":    wl_row.get("first_seen"),
            "last_seen":     wl_row.get("last_seen"),
        },
        "preconditions": {
            "company":        scan_info.get("company"),
            "sector":         scan_info.get("sector"),
            "industry":       scan_info.get("industry"),
            "market_cap":     scan_info.get("market_cap"),
            "pct_30d":        scan_info.get("pct_30d"),
            "total_score":    scan_info.get("total_score"),
            "attributes":     scan_info.get("attributes"),
        },
        "catalyst": {
            "score":     catalyst.get("score"),
            "blurb":     catalyst.get("blurb"),
            "catalysts": catalyst.get("catalysts"),
        },
        "recent_hits": _recent_hits(ticker),
        "position":    position,
    }


def _size_usd(size_pct: float, buying_power_usd: float) -> float:
    raw = (size_pct / 100.0) * buying_power_usd
    return float(max(0.0, min(raw, MAX_USD_PER_TRADE)))


def _decide_one(ticker: str, wl_row: dict, scan_map: dict, cat_map: dict,
                bp: float, dry_run: bool, logger=print) -> dict:
    scan_info = scan_map.get(ticker, {})
    catalyst  = cat_map.get(ticker, {})
    try:
        position = alpaca_client.get_position(ticker)
    except Exception:
        position = None

    pct_30d = scan_info.get("pct_30d")
    try:
        pct_30d_val = float(pct_30d) * 100 if pct_30d is not None else None
    except Exception:
        pct_30d_val = None

    # Hard guard: too-extended ⇒ SKIP without burning a Gemini call
    if pct_30d_val is not None and pct_30d_val > MAX_PCT_30D_FOR_BUY:
        logger(f"  {ticker}: pct_30d={pct_30d_val:.1f}% > {MAX_PCT_30D_FOR_BUY} — auto-SKIP")
        decision_id = journal.record_decision(
            ticker=ticker, action="SKIP", size_usd=0.0,
            rationale=f"Auto-skip: pct_30d {pct_30d_val:.1f}% exceeds chase threshold.",
            bup_score=wl_row.get("best_bup"), pct_30d=pct_30d_val,
            context={"reason": "auto_skip_extended"},
            dry_run=dry_run, status="filled",
        )
        return {"ticker": ticker, "action": "SKIP", "decision_id": decision_id}

    ctx = _build_context(ticker, wl_row, scan_info, catalyst, position)
    ep  = episodes.episode_block(ticker, ctx)

    try:
        proposal = gemini_client.decide(ticker, ctx, episodes=ep)
    except Exception as e:
        logger(f"  {ticker}: gemini error: {e}")
        return {"ticker": ticker, "error": str(e)}

    try:
        verdict = gemini_client.critique(ticker, proposal, ctx, episodes=ep)
    except Exception as e:
        logger(f"  {ticker}: critic error (failing open): {e}")
        verdict = {"verdict": "APPROVE", "concern": str(e), "downgrade_to": ""}

    out = gemini_client.apply_verdict(proposal, verdict)
    if verdict.get("verdict") != "APPROVE":
        logger(f"  {ticker}: critic {verdict.get('verdict')} → {out['action']}")

    action    = out["action"]
    size_pct  = out["size_pct"]
    rationale = out["rationale"]
    size_usd  = _size_usd(size_pct, bp) if action in {"SMALL", "NORMAL", "CONVICTION"} else 0.0

    order_id = None
    status   = "open"

    if action == "EXIT" and position:
        if not dry_run:
            try:
                r = alpaca_client.close_position(ticker)
                order_id = str(r.get("id") or "")
                status = "filled"
            except Exception as e:
                logger(f"  {ticker}: EXIT order failed: {e}")
                status = "rejected"
        else:
            status = "filled"  # dry-run treats as successful
    elif action in {"SMALL", "NORMAL", "CONVICTION"} and size_usd >= MIN_USD_PER_TRADE:
        if not dry_run:
            try:
                r = alpaca_client.submit_notional_order(ticker, "buy", size_usd)
                order_id = str(r.get("id") or "")
                status = "filled"
            except Exception as e:
                logger(f"  {ticker}: BUY order failed: {e}")
                status = "rejected"
        else:
            status = "filled"
    else:
        # SKIP, or buy size below minimum
        status = "filled"

    decision_id = journal.record_decision(
        ticker=ticker, action=action, size_usd=size_usd,
        rationale=rationale, bup_score=wl_row.get("best_bup"),
        pct_30d=pct_30d_val, context=ctx, order_id=order_id,
        dry_run=dry_run, status=status,
    )
    logger(f"  {ticker}: {action} size_pct={size_pct:.1f} size_usd=${size_usd:.0f} status={status}")
    return {"ticker": ticker, "action": action, "size_usd": size_usd,
            "decision_id": decision_id, "status": status, "order_id": order_id}


def run_batch(dry_run: bool | None = None, logger=print) -> dict:
    """One pass of the decision pipeline. Returns a summary dict."""
    if dry_run is None:
        dry_run = os.environ.get("AGENT_DRY_RUN", "0") == "1"

    if not config.GEMINI_API_KEY:
        return {"ok": False, "reason": "no_gemini_key"}

    wl_top = _watchlist_top(TOP_N_FROM_WATCHLIST)
    if not wl_top:
        return {"ok": False, "reason": "no_watchlist"}

    scan_map = _latest_scan_info()
    cat_map  = _latest_catalyst_scores()

    # Eligible: in top-N AND not decided in cooldown window
    eligible = []
    for row in wl_top:
        tk = (row.get("ticker") or "").upper()
        if not tk:
            continue
        if journal.already_decided_recently(tk, DECISION_COOLDOWN_SEC):
            continue
        eligible.append(row)

    eligible = eligible[:MAX_DECISIONS_PER_RUN]
    if not eligible:
        return {"ok": True, "decided": 0, "reason": "all_in_cooldown"}

    # Read account state once
    try:
        bp = alpaca_client.buying_power() if not dry_run else 10_000.0
    except Exception as e:
        if not dry_run:
            return {"ok": False, "reason": f"alpaca_creds_missing: {e}"}
        bp = 10_000.0

    logger(f"agent: {len(eligible)} eligible · BP=${bp:,.0f} · dry_run={dry_run}")

    results = []
    for row in eligible:
        tk = row["ticker"].upper()
        results.append(_decide_one(tk, row, scan_map, cat_map, bp,
                                   dry_run=dry_run, logger=logger))
        time.sleep(1.0)

    summary = {
        "ok":        True,
        "decided":   len(results),
        "buying_power": bp,
        "dry_run":   dry_run,
        "is_paper":  alpaca_client.is_paper() if not dry_run else True,
        "results":   results,
    }
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_batch(dry_run=args.dry_run), indent=2, default=str))
