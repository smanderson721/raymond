"""
Minimal Alpaca paper-trading client — urllib only, no SDK dependency.

Modeled after research/alpaca_watchlist.py. Paper endpoint is the default
and the only one that fires unless ALPACA_ALLOW_LIVE=1 AND the constant
ALLOW_LIVE_OVERRIDE is flipped in code (intentionally not env-driven).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE  = "https://api.alpaca.markets"
ALLOW_LIVE_OVERRIDE = False  # second gate on top of env var — flip in code only


class AlpacaError(Exception):
    pass


def _creds() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID", "")
    sec = os.environ.get("ALPACA_API_SECRET_KEY", "")
    if not key or not sec:
        raise AlpacaError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing")
    return key, sec


def _base_url() -> str:
    live = os.environ.get("ALPACA_ALLOW_LIVE") == "1" and ALLOW_LIVE_OVERRIDE
    return LIVE_BASE if live else PAPER_BASE


def _request(method: str, path: str, *, body: dict | None = None,
             query: dict | None = None, timeout: float = 15.0) -> Any:
    key, sec = _creds()
    url = _base_url() + path
    if query:
        from urllib.parse import urlencode
        url += "?" + urlencode({k: v for k, v in query.items() if v is not None})
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("APCA-API-KEY-ID", key)
    req.add_header("APCA-API-SECRET-KEY", sec)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        raise AlpacaError(f"{method} {path} → HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise AlpacaError(f"{method} {path} → URL error: {e.reason}") from None


# ── account ───────────────────────────────────────────────────────────

def get_account() -> dict:
    return _request("GET", "/v2/account")


def buying_power() -> float:
    try:
        return float(get_account().get("buying_power", 0) or 0)
    except Exception:
        return 0.0


def cash_available() -> float:
    try:
        return float(get_account().get("cash", 0) or 0)
    except Exception:
        return 0.0


# ── positions ─────────────────────────────────────────────────────────

def list_positions() -> list[dict]:
    res = _request("GET", "/v2/positions")
    return res if isinstance(res, list) else []


def get_position(symbol: str) -> dict | None:
    try:
        return _request("GET", f"/v2/positions/{symbol.upper()}")
    except AlpacaError as e:
        if "404" in str(e):
            return None
        raise


# ── orders ────────────────────────────────────────────────────────────

def submit_notional_order(symbol: str, side: str, notional_usd: float,
                          time_in_force: str = "day") -> dict:
    """Submit a market notional (fractional) order. side ∈ {'buy','sell'}."""
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be buy or sell, got {side}")
    body = {
        "symbol":         symbol.upper(),
        "side":           side,
        "type":           "market",
        "time_in_force":  time_in_force,
        "notional":       f"{float(notional_usd):.2f}",
    }
    return _request("POST", "/v2/orders", body=body)


def close_position(symbol: str) -> dict:
    return _request("DELETE", f"/v2/positions/{symbol.upper()}")


def list_orders(status: str = "all", limit: int = 50) -> list[dict]:
    res = _request("GET", "/v2/orders",
                   query={"status": status, "limit": str(limit),
                          "direction": "desc"})
    return res if isinstance(res, list) else []


# ── helpers ───────────────────────────────────────────────────────────

def is_paper() -> bool:
    return _base_url() == PAPER_BASE
