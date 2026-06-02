"""
Raymond Trading Agent — v1 (paper trading).

This is the substrate the user has been asking for. It is:

  - Paper-only by default. Live trading requires BOTH the env var
    ALPACA_ALLOW_LIVE=1 AND the constant ALLOW_LIVE_OVERRIDE in
    alpaca_client.py being flipped to True (a code change). One alone
    is not enough.
  - Cron-driven via research/scans/agent_decide.py — Raymond's scheduler
    fires it every 5 minutes during market hours.
  - Conservative: per-call MAX_DECISIONS_PER_RUN cap, per-ticker
    DECISION_COOLDOWN_SEC cooldown, hard MAX_USD_PER_TRADE cap, and an
    automatic SKIP on tickers that have already run up beyond
    MAX_PCT_30D_FOR_BUY in the last 30 days.

Modules
-------
  journal.py       sqlite-backed decisions / outcomes / reflections log
  alpaca_client.py paper trading client (urllib, no SDK)
  gemini_client.py one-shot Gemini decide() call
  decide.py        per-call pipeline
  prompts/         system prompt (and, later, auto-updated playbook)
  __main__.py      `python -m agent decide|journal|account`

HTTP surface (in live_daemon.py)
--------------------------------
  GET  /api/agent-journal  → entries (decisions, outcomes, reflections)
  GET  /api/agent-account  → Alpaca paper account state + positions
  POST /api/agent-run      → manually trigger a decision batch
                             body: {"dry_run": true|false}; default true

Deferred to v2: episodes.py (FTS5 memory), reflect.py (nightly retrospect),
shadow_ledger.py (mechanical baseline), tools.py (function-call schemas).
"""
