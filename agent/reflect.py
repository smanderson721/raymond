"""
Nightly retrospective. Reads the last N hours of decisions + outcomes,
asks Gemini for (1) a one-paragraph summary and (2) an updated playbook
(3-7 short rules). Writes both into the `reflections` table.

The playbook is auto-loaded by gemini_client._load_playbook() and
prepended to every future system prompt.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

from agent import journal  # noqa: E402

LOOKBACK_HOURS = 24
MAX_DECISIONS_IN_PROMPT = 60


def _gather(since_ts: int) -> dict:
    decisions, outcomes = journal.decisions_outcomes_since(since_ts)
    by_id = {d["id"]: d for d in decisions}
    enriched_outcomes = []
    for o in outcomes:
        d = by_id.get(o["decision_id"], {})
        enriched_outcomes.append({
            "ticker":      o.get("ticker"),
            "action":      o.get("action"),
            "size_usd":    o.get("size_usd"),
            "pnl_pct":     o.get("pnl_pct"),
            "exit_reason": o.get("exit_reason"),
            "rationale":   (d.get("rationale") or "")[:240],
        })

    by_action: dict[str, int] = {}
    for d in decisions:
        by_action[d["action"]] = by_action.get(d["action"], 0) + 1

    pnls = [float(o.get("pnl_pct") or 0) for o in outcomes]
    realized = [float(o.get("realized_pnl") or 0) for o in outcomes]
    stats = {
        "decisions":      len(decisions),
        "outcomes":       len(outcomes),
        "by_action":      by_action,
        "wins":           sum(1 for p in pnls if p > 0),
        "losses":         sum(1 for p in pnls if p < 0),
        "avg_pnl_pct":    (sum(pnls) / len(pnls)) if pnls else None,
        "total_pnl_usd":  sum(realized) if realized else 0.0,
    }
    return {
        "stats":     stats,
        "decisions": [
            {
                "ticker":    d.get("ticker"),
                "action":    d.get("action"),
                "size_usd":  d.get("size_usd"),
                "bup":       d.get("bup_score"),
                "pct_30d":   d.get("pct_30d"),
                "rationale": (d.get("rationale") or "")[:240],
                "status":    d.get("status"),
            }
            for d in decisions[-MAX_DECISIONS_IN_PROMPT:]
        ],
        "outcomes":  enriched_outcomes,
    }


def _build_prompt(prior_playbook: str, payload: dict) -> str:
    return (
        "You are reviewing the Raymond trading agent's last 24 hours of paper "
        "trading. Produce a brief retrospective and update the playbook.\n\n"
        f"=== prior playbook ===\n{prior_playbook or '(none yet)'}\n\n"
        f"=== last 24h summary ===\n{json.dumps(payload['stats'], indent=2, default=str)}\n\n"
        f"=== decisions ===\n{json.dumps(payload['decisions'], indent=2, default=str)[:6000]}\n\n"
        f"=== outcomes ===\n{json.dumps(payload['outcomes'], indent=2, default=str)[:4000]}\n\n"
        "=== task ===\n"
        "Return STRICT JSON with exactly these keys:\n"
        '  "summary":  one paragraph (~80 words) covering what worked and what didn\'t\n'
        '  "playbook": 3-7 short imperative rules, one per line, prefixed with "- ".\n'
        "             Each rule must be actionable in the decide() prompt — not\n"
        "             generic platitudes. Keep rules that still apply, drop\n"
        "             stale ones, add new ones learned from this window.\n"
        "No markdown fences. JSON only."
    )


def _coerce(raw: str) -> dict:
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in reflection output: {raw[:200]!r}")
    data = json.loads(m.group())
    return {
        "summary":  str(data.get("summary", "")).strip()[:1200],
        "playbook": str(data.get("playbook", "")).strip()[:4000],
    }


def run_batch(dry_run: bool = False, logger=print) -> dict:
    if not config.GEMINI_API_KEY:
        return {"ok": False, "reason": "no_gemini_key"}

    since_ts = int(time.time()) - LOOKBACK_HOURS * 3600
    payload = _gather(since_ts)
    if payload["stats"]["decisions"] == 0 and payload["stats"]["outcomes"] == 0:
        return {"ok": True, "reason": "no_activity", "stats": payload["stats"]}

    prior = journal.latest_reflection() or {}
    prior_pb = prior.get("playbook") or ""

    prompt = _build_prompt(prior_pb, payload)

    if dry_run:
        return {"ok": True, "dry_run": True,
                "stats": payload["stats"],
                "prompt_chars": len(prompt)}

    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=config.MODEL_RESEARCH,
                contents=prompt,
            )
            text = (resp.text or "").strip()
            if not text and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                text = "\n".join(p.text for p in parts
                                 if hasattr(p, "text") and p.text)
            out = _coerce(text)
            rid = journal.record_reflection(out["summary"], out["playbook"])
            logger(f"reflection #{rid}: {payload['stats']['outcomes']} outcomes, "
                   f"avg_pnl={payload['stats']['avg_pnl_pct']}")
            return {"ok": True, "reflection_id": rid,
                    "stats": payload["stats"],
                    "summary": out["summary"],
                    "playbook_chars": len(out["playbook"])}
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    return {"ok": False, "error": f"gemini failed: {last_err}"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_batch(dry_run=args.dry_run), indent=2, default=str))
