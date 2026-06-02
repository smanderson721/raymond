"""Scheduler shim for agent.shadow_ledger (mechanical baseline)."""

from __future__ import annotations

from research.live_score_engine import Session

SCAN_NAME = "agent_shadow"


def run() -> dict:
    from agent import shadow_ledger
    with Session(SCAN_NAME, note="Trading-agent shadow baseline") as s:
        try:
            summary = shadow_ledger.run_batch(
                logger=lambda m: s.log(m, level="info"))
        except Exception as e:
            s.log(f"shadow failed: {e}", level="warn")
            return {"ok": False, "error": str(e)}
        opened = summary.get("opened", 0)
        closed = summary.get("closed", 0)
        stats  = summary.get("stats") or {}
        s.log(
            f"{opened} opened · {closed} closed · "
            f"book: open={stats.get('open',0)} closed={stats.get('closed',0)} "
            f"avg_pnl={stats.get('avg_pnl_pct')}",
            level="notable" if (opened or closed) else "info",
        )
        return summary


if __name__ == "__main__":
    print(run())
