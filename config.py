"""
Minimal config for the Raymond stocks scanner.

API keys must come from the environment (or a local .env file in dev).
.env is gitignored — never commit secrets.
"""

import os
from pathlib import Path

# Auto-load .env file if present (local dev only — CI uses GitHub Secrets)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# Gemini model used for catalyst scoring
MODEL_RESEARCH = "gemini-3-flash-preview"
