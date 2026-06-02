"""
Gemini client for the trading agent. Wraps a single `decide()` call that
asks the model to evaluate one ticker and return a structured JSON action.

Uses MODEL_RESEARCH from config (currently gemini-3.1-flash-lite).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_VALID_ACTIONS = {"SKIP", "SMALL", "NORMAL", "CONVICTION", "EXIT"}


def _load_system_prompt() -> str:
    p = PROMPT_DIR / "system.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _load_playbook() -> str:
    """Optional. Latest reflection's playbook, if one exists."""
    try:
        from agent.journal import latest_reflection
        r = latest_reflection()
        return (r or {}).get("playbook") or ""
    except Exception:
        return ""


def _build_client():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    from google import genai
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _format_context(ctx: dict) -> str:
    """Render the per-ticker context block as compact JSON."""
    return json.dumps(ctx, default=str, indent=2)[:6000]


def _coerce_decision(raw: str) -> dict:
    """Extract {action, size_pct, rationale} from model output."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in model output: {raw[:200]!r}")
    data = json.loads(m.group())
    action = str(data.get("action", "SKIP")).upper()
    if action not in _VALID_ACTIONS:
        action = "SKIP"
    size_pct = float(data.get("size_pct", 0) or 0)
    size_pct = max(0.0, min(size_pct, 100.0))
    rationale = str(data.get("rationale", "")).strip()[:1200]
    return {"action": action, "size_pct": size_pct, "rationale": rationale}


def decide(ticker: str, context: dict, episodes: dict | None = None) -> dict:
    """Ask Gemini for a decision on one ticker. Returns
    {action, size_pct, rationale}. Raises on hard failure.

    If `episodes` is provided (own_history + similar_setups blocks from
    agent.episodes), it's appended to the prompt so the model can draw
    on past decisions and their realized outcomes."""
    client = _build_client()
    system = _load_system_prompt()
    playbook = _load_playbook()
    ctx_block = _format_context(context)
    ep_block = ""
    if episodes:
        ep_block = "\n\n=== past episodes ===\n" + json.dumps(
            episodes, default=str, indent=2)[:3500]
    user = (
        f"TICKER: {ticker.upper()}\n\n"
        f"=== context ===\n{ctx_block}{ep_block}\n\n"
        "=== task ===\n"
        "Return STRICT JSON with these keys exactly:\n"
        '  "action":   one of SKIP, SMALL, NORMAL, CONVICTION, EXIT\n'
        '  "size_pct": 0-100 (percent of available buying power; 0 for SKIP/EXIT)\n'
        '  "rationale": 1-3 short sentences citing the strongest signals\n'
        "If the past-episodes block shows similar setups lost money, prefer a\n"
        "smaller size or SKIP. No markdown. No commentary. Just the JSON object."
    )
    prompt = system + ("\n\n=== playbook ===\n" + playbook if playbook else "") + "\n\n" + user

    last_err: Exception | None = None
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
            return _coerce_decision(text)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini decide failed after 3 attempts: {last_err}")


# ── critic ────────────────────────────────────────────────────────────

_ACTION_TIER = {"SKIP": 0, "SMALL": 1, "NORMAL": 2, "CONVICTION": 3, "EXIT": 0}


def _coerce_verdict(raw: str) -> dict:
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {"verdict": "APPROVE", "concern": "", "downgrade_to": ""}
    try:
        d = json.loads(m.group())
    except Exception:
        return {"verdict": "APPROVE", "concern": "", "downgrade_to": ""}
    verdict = str(d.get("verdict", "APPROVE")).upper()
    if verdict not in {"APPROVE", "DOWNGRADE", "SKIP"}:
        verdict = "APPROVE"
    return {
        "verdict":      verdict,
        "concern":      str(d.get("concern", "")).strip()[:400],
        "downgrade_to": str(d.get("downgrade_to", "")).upper(),
    }


def critique(ticker: str, proposal: dict, context: dict,
             episodes: dict | None = None) -> dict:
    """Second-opinion call. Returns {verdict, concern, downgrade_to}.

    Set env var AGENT_CRITIC=0 to disable (saves one Gemini call per
    decision)."""
    if os.environ.get("AGENT_CRITIC", "1") == "0":
        return {"verdict": "APPROVE", "concern": "", "downgrade_to": ""}
    if proposal.get("action") in {"SKIP", "EXIT"}:
        # Don't bother critiquing inactions
        return {"verdict": "APPROVE", "concern": "", "downgrade_to": ""}

    client = _build_client()
    ctx_block = _format_context(context)
    ep_block = ""
    if episodes:
        ep_block = "\n\n=== past episodes ===\n" + json.dumps(
            episodes, default=str, indent=2)[:2500]
    prompt = (
        "You are the CRITIC. A proposer agent has suggested the following "
        "trade. Your job is to argue against it and either APPROVE, "
        "DOWNGRADE (to a smaller size tier), or SKIP it.\n\n"
        f"TICKER: {ticker.upper()}\n"
        f"=== proposal ===\n{json.dumps(proposal, indent=2)}\n\n"
        f"=== context ===\n{ctx_block}{ep_block}\n\n"
        "Reasons to DOWNGRADE or SKIP include: extended pct_30d, weak/generic "
        "catalyst, similar past setups that lost, missing news despite high BUP, "
        "tiny float pump risk, or rationale that doesn't match the data.\n\n"
        "Return STRICT JSON: "
        '{"verdict": "APPROVE|DOWNGRADE|SKIP", '
        '"downgrade_to": "SKIP|SMALL|NORMAL" (only if DOWNGRADE), '
        '"concern": "one sentence"}. '
        "No markdown."
    )

    last_err = None
    for attempt in range(2):
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
            return _coerce_verdict(text)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    # Fail-open: if critic call fails, fall back to APPROVE
    return {"verdict": "APPROVE",
            "concern": f"critic_unavailable: {last_err}",
            "downgrade_to": ""}


def apply_verdict(proposal: dict, verdict: dict) -> dict:
    """Merge the proposer's proposal with the critic's verdict and
    return the final {action, size_pct, rationale} we'll record."""
    v = verdict.get("verdict", "APPROVE")
    if v == "APPROVE":
        return proposal
    if v == "SKIP":
        return {
            "action":    "SKIP",
            "size_pct":  0.0,
            "rationale": (proposal.get("rationale", "") +
                          f" | CRITIC SKIP: {verdict.get('concern','')}")[:1200],
        }
    if v == "DOWNGRADE":
        target = verdict.get("downgrade_to") or "SMALL"
        if target not in _ACTION_TIER:
            target = "SMALL"
        # Pick a sensible size at the new tier
        size_map = {"SKIP": 0.0, "SMALL": 3.0, "NORMAL": 8.0}
        return {
            "action":    target,
            "size_pct":  size_map.get(target, 0.0),
            "rationale": (proposal.get("rationale", "") +
                          f" | CRITIC DOWNGRADE→{target}: {verdict.get('concern','')}")[:1200],
        }
    return proposal
