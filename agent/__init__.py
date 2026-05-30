"""Raymond trading agent.

A small Gemini-driven harness that consumes raymond's live signal feed and
manages a paper trading account on Alpaca. See ``agent/README.md`` for the
overall architecture; the short version is:

    agent/
      alpaca_client.py    paper trading wrapper (positions, orders, account)
      journal.py          sqlite journal of every decision + outcome
      episodes.py         vector-ish memory over past decisions
      gemini_client.py    proposer + critic + reflection calls
      tools.py            Gemini function-call schemas
      shadow_ledger.py    mechanical baseline (no LLM, $100/5-day fixed rule)
      decide.py           main decision pipeline (called per top-10 entrant)
      reflect.py          nightly retrospective → playbook.md
      prompts/            system + playbook (auto-updated)
      __main__.py         CLI dispatch

Stage 1 (this commit): the substrate. CLI-driven, ``--dry-run`` by default,
no live trades fire unless you pass ``--live``. Once you've verified the
pipeline by hand we'll wire it to raymond's SSE for automatic triggering.
"""
