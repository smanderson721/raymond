"""Scheduler shim for agent.outcomes_tracker."""

from __future__ import annotations

from research.live_score_engine import Session

SCAN_NAME = "agent_outcomes"


def run() -> dict:
    from agent import outcomes_tracker
    with Session(SCAN_NAME, note="Trading-agent outcomes/PnL tracker") as s:
        try:
            summary = outcomes_tracker.run_batch(
                logger=lambda m: s.log(m, level="info"))
        except Exception as e:
            s.log(f"outcomes failed: {e}", level="warn")
            return {"ok": False, "error": str(e)}
        if not summary.get("ok"):
            s.log(f"skipped: {summary.get('error') or summary.get('reason')}",
                  level="info")
            return summary
        checked = summary.get("checked", 0)
        closed  = summary.get("closed", 0)
        s.log(
            f"{checked} open · {closed} closed",
            level="notable" if closed else "info",
        )
        return summary


if __name__ == "__main__":
    print(run())
