"""
Scheduler shim: makes ``agent.decide`` runnable as a Raymond scan so it
shows up in /api/state and on the timeline alongside the other scans.

Registered in live_daemon.py SCHEDULES.
"""

from __future__ import annotations

from research.live_score_engine import Session

SCAN_NAME = "agent_decide"


def run() -> dict:
    from agent import decide
    with Session(SCAN_NAME, note="Trading-agent v1 (paper)") as s:
        try:
            summary = decide.run_batch(logger=lambda m: s.log(m, level="info"))
        except Exception as e:
            s.log(f"agent batch failed: {e}", level="warn")
            return {"ok": False, "error": str(e)}

        if not summary.get("ok"):
            s.log(f"skipped: {summary.get('reason')}", level="info")
            return summary

        decided = summary.get("decided", 0)
        if decided == 0:
            s.log(summary.get("reason") or "no eligible tickers", level="info")
        else:
            actions = [
                f"{r.get('ticker')}:{r.get('action')}"
                for r in summary.get("results", []) if "action" in r
            ]
            s.log(
                f"{decided} decided · BP=${summary.get('buying_power', 0):,.0f}"
                + (f" · {', '.join(actions)}" if actions else ""),
                level="notable",
            )
        return summary


if __name__ == "__main__":
    print(run())
