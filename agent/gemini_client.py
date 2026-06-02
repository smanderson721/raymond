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


def decide(ticker: str, context: dict) -> dict:
    """Ask Gemini for a decision on one ticker. Returns
    {action, size_pct, rationale}. Raises on hard failure."""
    client = _build_client()
    system = _load_system_prompt()
    playbook = _load_playbook()
    ctx_block = _format_context(context)
    user = (
        f"TICKER: {ticker.upper()}\n\n"
        f"=== context ===\n{ctx_block}\n\n"
        "=== task ===\n"
        "Return STRICT JSON with these keys exactly:\n"
        '  "action":   one of SKIP, SMALL, NORMAL, CONVICTION, EXIT\n'
        '  "size_pct": 0-100 (percent of available buying power; 0 for SKIP/EXIT)\n'
        '  "rationale": 1-3 short sentences citing the strongest signals\n'
        "No markdown. No commentary. Just the JSON object."
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
