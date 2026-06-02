"""
Outcomes tracker — walks open BUY decisions, queries Alpaca, records PnL.

Runs every 30 min during market hours. For each filled BUY decision that
doesn't yet have an outcome row:

  1. If Alpaca no longer reports a position for that ticker  → realized
     exit. Pull the most recent matching SELL order to recover the exit
     price + realized PnL, write the outcome.
  2. If the position is still open AND age ≥ MAX_HOLD_DAYS              → force-close
     via alpaca_client.close_position(), record outcome with reason
     "time_stop".
  3. If the position is still open AND unrealized_pl_pc ≤ -STOP_LOSS_PCT → force-close,
     record outcome with reason "stop_loss".
  4. If the position is still open AND unrealized_pl_pc ≥ TAKE_PROFIT_PCT → force-close,
     record outcome with reason "take_profit".
  5. Otherwise: leave it open.

Dry-run decisions are ignored (no real position exists to track).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from agent import alpaca_client, journal

MAX_HOLD_DAYS    = 5
STOP_LOSS_PCT    = 8.0    # close if unrealized loss ≥ 8%
TAKE_PROFIT_PCT  = 15.0   # close if unrealized gain ≥ 15%
ONE_DAY_SEC      = 86_400


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _find_recent_sell(orders: list[dict], symbol: str, after_ts: int) -> dict | None:
    """Find the most recent filled SELL order for `symbol` since after_ts."""
    sym = symbol.upper()
    best = None
    best_t = 0
    for o in orders:
        if (o.get("symbol") or "").upper() != sym:
            continue
        if (o.get("side") or "").lower() != "sell":
            continue
        if (o.get("status") or "").lower() not in ("filled", "partially_filled"):
            continue
        ts_str = o.get("filled_at") or o.get("updated_at") or o.get("submitted_at") or ""
        try:
            t = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp())
        except Exception:
            continue
        if t < after_ts:
            continue
        if t > best_t:
            best = o
            best_t = t
    return best


def _record_exit(decision: dict, exit_price: float, reason: str,
                 logger) -> dict:
    entry_usd = _safe_float(decision.get("size_usd"))
    # We submitted notional orders so qty ≈ entry_usd / entry_price; we don't
    # have entry_price stored — pull from Alpaca order if needed. Use percent.
    # For simpler bookkeeping, compute pnl_pct from Alpaca's position avg_entry
    # when available (passed in via decision["_avg_entry"] hack), else 0.
    avg_entry = _safe_float(decision.get("_avg_entry"))
    if avg_entry > 0 and exit_price > 0:
        pnl_pct = (exit_price - avg_entry) / avg_entry * 100.0
        realized = (pnl_pct / 100.0) * entry_usd
    else:
        pnl_pct = _safe_float(decision.get("_unrealized_plpc"), 0.0) * 100.0
        realized = (pnl_pct / 100.0) * entry_usd
    oid = journal.record_outcome(
        decision_id=int(decision["id"]),
        realized_pnl=realized,
        pnl_pct=pnl_pct,
        exit_reason=reason,
        notes=f"exit_price={exit_price:.4f} entry_usd={entry_usd:.2f}",
    )
    logger(f"  {decision['ticker']}: exit recorded ({reason}) pnl={pnl_pct:+.2f}%")
    return {"decision_id": decision["id"], "ticker": decision["ticker"],
            "pnl_pct": pnl_pct, "reason": reason, "outcome_id": oid}


def run_batch(dry_run: bool = False, logger=print) -> dict:
    open_decisions = journal.open_buy_decisions()
    if not open_decisions:
        return {"ok": True, "checked": 0, "closed": 0, "reason": "no_open_decisions"}

    try:
        positions = alpaca_client.list_positions()
        orders    = alpaca_client.list_orders(status="closed", limit=200)
    except Exception as e:
        return {"ok": False, "error": f"alpaca_unavailable: {e}"}

    pos_by_ticker = {(p.get("symbol") or "").upper(): p for p in positions}
    now = int(time.time())
    closed: list[dict] = []

    for d in open_decisions:
        tk = (d.get("ticker") or "").upper()
        age_sec = now - int(d.get("ts") or now)
        pos = pos_by_ticker.get(tk)

        if pos is None:
            sell = _find_recent_sell(orders, tk, int(d.get("ts") or 0))
            if not sell:
                logger(f"  {tk}: no position, no recent sell — skipping")
                continue
            exit_price = _safe_float(sell.get("filled_avg_price"))
            if exit_price <= 0:
                continue
            d["_avg_entry"] = 0.0
            d["_unrealized_plpc"] = 0.0
            closed.append(_record_exit(d, exit_price, "alpaca_closed", logger))
            continue

        avg_entry  = _safe_float(pos.get("avg_entry_price"))
        cur_price  = _safe_float(pos.get("current_price"))
        plpc       = _safe_float(pos.get("unrealized_plpc"))   # 0.05 = 5%
        d["_avg_entry"]      = avg_entry
        d["_unrealized_plpc"] = plpc
        pct = plpc * 100.0

        # Time stop
        if age_sec >= MAX_HOLD_DAYS * ONE_DAY_SEC:
            if not dry_run:
                try:
                    alpaca_client.close_position(tk)
                except Exception as e:
                    logger(f"  {tk}: close failed: {e}")
                    continue
            closed.append(_record_exit(d, cur_price, "time_stop", logger))
            continue
        if pct <= -STOP_LOSS_PCT:
            if not dry_run:
                try:
                    alpaca_client.close_position(tk)
                except Exception as e:
                    logger(f"  {tk}: stop close failed: {e}")
                    continue
            closed.append(_record_exit(d, cur_price, "stop_loss", logger))
            continue
        if pct >= TAKE_PROFIT_PCT:
            if not dry_run:
                try:
                    alpaca_client.close_position(tk)
                except Exception as e:
                    logger(f"  {tk}: take_profit close failed: {e}")
                    continue
            closed.append(_record_exit(d, cur_price, "take_profit", logger))
            continue
        # otherwise leave it open
        logger(f"  {tk}: open age={age_sec//ONE_DAY_SEC}d plpc={pct:+.2f}%")

    return {"ok": True, "checked": len(open_decisions),
            "closed": len(closed), "results": closed}


if __name__ == "__main__":
    import json as _json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(_json.dumps(run_batch(dry_run=args.dry_run), indent=2, default=str))
