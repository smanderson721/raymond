"""Scheduler shim for agent.reflect (nightly retrospective)."""

from __future__ import annotations

from research.live_score_engine import Session

SCAN_NAME = "agent_reflect"


def run() -> dict:
    from agent import reflect
    with Session(SCAN_NAME, note="Trading-agent nightly retrospective") as s:
        try:
            summary = reflect.run_batch(
                logger=lambda m: s.log(m, level="info"))
        except Exception as e:
            s.log(f"reflect failed: {e}", level="warn")
            return {"ok": False, "error": str(e)}
        if not summary.get("ok"):
            s.log(f"skipped: {summary.get('reason') or summary.get('error')}",
                  level="info")
            return summary
        stats = summary.get("stats") or {}
        msg = (f"refl #{summary.get('reflection_id')}: "
               f"{stats.get('outcomes',0)} outcomes "
               f"avg_pnl={stats.get('avg_pnl_pct')}")
        s.log(msg, level="notable" if summary.get("reflection_id") else "info")
        return summary


if __name__ == "__main__":
    print(run())
