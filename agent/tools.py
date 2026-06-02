"""
Function-call schemas for Gemini.

The proposer call asks Gemini to call `propose_action(...)` with structured
arguments rather than emitting raw JSON in free text. Same goes for the
critic call which uses `judge_action(...)`. Using tool schemas lets the
model fill arguments against a type system instead of generating quoted
JSON that has to be re-parsed.

We *also* keep the raw-JSON fallback in gemini_client._coerce_decision so
the agent keeps working if the SDK / model variant ever drops tool
support.
"""

from __future__ import annotations


# google-genai uses google.genai.types.FunctionDeclaration / Tool.
# We build them lazily inside gemini_client to avoid importing the SDK
# at module load time; here we just keep the schema dicts.

PROPOSE_ACTION_SCHEMA = {
    "name": "propose_action",
    "description": (
        "Submit the agent's trading decision for one ticker. "
        "Action must reflect the rubric in the system prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["SKIP", "SMALL", "NORMAL", "CONVICTION", "EXIT"],
                "description": "Decision label per the rubric.",
            },
            "size_pct": {
                "type": "number",
                "description": "Percent of available buying power, 0-100. "
                               "Use 0 for SKIP and EXIT.",
            },
            "rationale": {
                "type": "string",
                "description": "1-3 short sentences citing the strongest "
                               "signals justifying the action.",
            },
        },
        "required": ["action", "size_pct", "rationale"],
    },
}


JUDGE_ACTION_SCHEMA = {
    "name": "judge_action",
    "description": (
        "Critique the proposer's decision. Either approve as-is or "
        "recommend a downgrade with a one-sentence concern."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["APPROVE", "DOWNGRADE", "SKIP"],
                "description": "APPROVE keeps the proposal. DOWNGRADE moves "
                               "to a smaller size tier. SKIP rejects the "
                               "proposal entirely.",
            },
            "downgrade_to": {
                "type": "string",
                "enum": ["SKIP", "SMALL", "NORMAL"],
                "description": "Required when verdict=DOWNGRADE. The new "
                               "action to use.",
            },
            "concern": {
                "type": "string",
                "description": "1 sentence explaining the worry. Empty on "
                               "APPROVE.",
            },
        },
        "required": ["verdict", "concern"],
    },
}
