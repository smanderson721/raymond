"""
Shadow ledger — mechanical baseline that the Gemini agent's PnL gets
compared against.

Rules (no LLM, no context, no judgment):
  - Open a $SHADOW_NOTIONAL_USD trade on every NEW top-N BUP entrant
    that we haven't shadowed in the last SHADOW_COOLDOWN_SEC window.
  - Close every shadow trade after SHADOW_HOLD_DAYS calendar days.
  - Use yfinance live price for entry + exit. No order is placed
    anywhere — shadow PnL is purely accounting in agent.db.

If the Gemini agent's win-rate doesn't beat this, we're paying Gemini
for nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent import journal  # noqa: E402

WATCHLIST_FILE       = REPO_ROOT / "research_output" / "watchlist_combined.json"
SHADOW_TOP_N         = 10
SHADOW_NOTIONAL_USD  = 100.0
SHADOW_HOLD_DAYS     = 5
SHADOW_COOLDOWN_SEC  = 7 * 86_400      # don't re-shadow same ticker for a week
MAX_OPENS_PER_RUN    = 4               # cap yfinance calls per cron tick
ONE_DAY_SEC          = 86_400


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _yf_price(ticker: str) -> float | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "fast_info", None)
        if info:
            for k in ("last_price", "lastPrice", "regular_market_price"):
                v = getattr(info, k, None) if not isinstance(info, dict) else info.get(k)
                if v:
                    return float(v)
        hist = t.history(period="1d", interval="1m")
        if hist is not None and len(hist):
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


def _watchlist_top(n: int) -> list[dict]:
    wl = _read_json(WATCHLIST_FILE, {})
    return (wl.get("tickers") or [])[:n]


def open_new_shadows(logger=print) -> list[dict]:
    top = _watchlist_top(SHADOW_TOP_N)
    if not top:
        return []
    recent = journal.shadow_recent_tickers(SHADOW_COOLDOWN_SEC)
    opens: list[dict] = []
    for row in top:
        tk = (row.get("ticker") or "").upper()
        if not tk or tk in recent:
            continue
        price = _yf_price(tk)
        if not price or price <= 0:
            logger(f"  {tk}: no price, skipping")
            continue
        sid = journal.shadow_open(
            tk, price, SHADOW_NOTIONAL_USD,
            note=f"bup={row.get('best_bup')}",
        )
        opens.append({"id": sid, "ticker": tk, "entry": price})
        logger(f"  shadow open {tk} @ {price:.2f} (#{sid})")
        if len(opens) >= MAX_OPENS_PER_RUN:
            break
    return opens


def close_due_shadows(logger=print) -> list[dict]:
    """Close any open shadow ≥ SHADOW_HOLD_DAYS old."""
    due = journal.shadow_open_trades(older_than_sec=SHADOW_HOLD_DAYS * ONE_DAY_SEC)
    closed: list[dict] = []
    for s in due:
        tk = s["ticker"]
        price = _yf_price(tk)
        if not price or price <= 0:
            logger(f"  {tk}: no exit price, leaving open")
            continue
        journal.shadow_close(int(s["id"]), price, note="time_stop")
        ep = float(s["entry_price"])
        pnl = (price - ep) / ep * 100 if ep else 0.0
        closed.append({"id": s["id"], "ticker": tk,
                       "entry": ep, "exit": price, "pnl_pct": pnl})
        logger(f"  shadow close {tk}: {ep:.2f} → {price:.2f} ({pnl:+.2f}%)")
    return closed


def run_batch(dry_run: bool = False, logger=print) -> dict:
    if dry_run:
        return {"ok": True, "dry_run": True,
                "would_open": len(_watchlist_top(SHADOW_TOP_N)),
                "open_now":   len(journal.shadow_open_trades()),
                "stats":      journal.shadow_stats()}
    opens  = open_new_shadows(logger=logger)
    closes = close_due_shadows(logger=logger)
    return {
        "ok":      True,
        "opened":  len(opens),
        "closed":  len(closes),
        "stats":   journal.shadow_stats(),
        "opens":   opens,
        "closes":  closes,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_batch(dry_run=args.dry_run), indent=2, default=str))
