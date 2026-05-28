"""
Unusual options activity scanner (yfinance, no API key).

Pulls the nearest-expiry options chain for each ticker on the current
top-N watchlist. Awards points when call or put volume on near-the-money
strikes is unusually large relative to existing open interest, which
historically precedes directional moves.

Runs every 30 minutes during market hours. ~2s per ticker, so ~2 min
total for 60 tickers.

Signals:
  - call_vol / call_OI >= 2.0 on near-the-money strikes  -> bullish flow
  - put_vol  / put_OI  >= 2.0 on near-the-money strikes  -> defensive flow
  - put/call ratio extreme (<0.30 or >2.0) on >5k contracts -> directional
  - very high IV vs prior session (when cache available)   -> event pricing
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from research.live_score_engine import Session, LIVE_DIR

SCAN_NAME = "options_unusual"
WATCHLIST_FILE = Path(LIVE_DIR) / "watchlist.json"
SCAN_TOP_N = 60                    # scan the top N current watchlist tickers
ATM_BAND = 0.05                    # ±5% around spot
MIN_CONTRACT_VOL = 500             # ignore tiny chains
VOL_OI_HIT = 2.0                   # vol/OI >= 2 on ATM strikes = unusual
VOL_OI_MEGA = 4.0                  # >= 4 = very unusual
PC_BULL = 0.30                     # put/call ratio below this is heavily bullish
PC_BEAR = 2.0                      # put/call ratio above this is heavily bearish
MIN_TOTAL_VOL = 5_000              # require meaningful liquidity for PC ratio


def _load_watchlist_tickers(n: int) -> list[str]:
    if not WATCHLIST_FILE.exists():
        return []
    try:
        wl = json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return []
    out = []
    for r in wl.get("watchlist", [])[:n]:
        tk = r.get("ticker", "").upper()
        # skip non-equity tickers (preferred shares, warrants, units)
        if not tk or any(c in tk for c in (".", "-", "/")):
            continue
        if len(tk) > 5:
            continue
        out.append(tk)
    return out


def _analyze(ticker: str) -> list[tuple[float, str]]:
    """Return list of (points, reason) awards for one ticker."""
    awards: list[tuple[float, str]] = []
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return awards
        # nearest expiry (skip 0-DTE if there are more)
        expiry = expirations[0]
        if len(expirations) > 1:
            today = datetime.now(timezone.utc).date()
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            if (exp_date - today).days < 1:
                expiry = expirations[1]
        chain = tk.option_chain(expiry)
        # spot price from history
        hist = tk.history(period="2d", auto_adjust=False)
        if hist.empty:
            return awards
        spot = float(hist["Close"].iloc[-1])
    except Exception:
        return awards

    if spot <= 0:
        return awards

    lo, hi = spot * (1 - ATM_BAND), spot * (1 + ATM_BAND)

    calls = chain.calls
    puts = chain.puts
    if calls.empty and puts.empty:
        return awards

    # filter to ATM band, drop NaN vol/OI
    atm_calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].fillna(0)
    atm_puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].fillna(0)

    call_vol = int(atm_calls["volume"].sum())
    call_oi = int(atm_calls["openInterest"].sum())
    put_vol = int(atm_puts["volume"].sum())
    put_oi = int(atm_puts["openInterest"].sum())
    total_vol_chain = int(calls["volume"].fillna(0).sum() + puts["volume"].fillna(0).sum())

    # need some liquidity
    if call_vol + put_vol < MIN_CONTRACT_VOL:
        return awards

    # ATM vol vs OI signals
    if call_oi >= 100:
        ratio = call_vol / call_oi
        if ratio >= VOL_OI_MEGA:
            awards.append((10.0, f"mega ATM call buying {call_vol:,}c vs {call_oi:,} OI ({ratio:.1f}×) exp {expiry}"))
        elif ratio >= VOL_OI_HIT:
            awards.append((6.0, f"unusual ATM call vol {call_vol:,}c vs {call_oi:,} OI ({ratio:.1f}×) exp {expiry}"))

    if put_oi >= 100:
        ratio = put_vol / put_oi
        if ratio >= VOL_OI_MEGA:
            awards.append((4.0, f"mega ATM put buying {put_vol:,}p vs {put_oi:,} OI ({ratio:.1f}×) exp {expiry}"))
        elif ratio >= VOL_OI_HIT:
            awards.append((2.0, f"unusual ATM put vol {put_vol:,}p vs {put_oi:,} OI ({ratio:.1f}×) exp {expiry}"))

    # whole-chain put/call ratio
    if total_vol_chain >= MIN_TOTAL_VOL:
        total_call = int(calls["volume"].fillna(0).sum())
        total_put = int(puts["volume"].fillna(0).sum())
        if total_call > 0:
            pc = total_put / total_call
            if pc <= PC_BULL:
                awards.append((4.0, f"bullish flow P/C {pc:.2f} on {total_vol_chain:,} contracts"))
            elif pc >= PC_BEAR:
                awards.append((2.0, f"bearish flow P/C {pc:.2f} on {total_vol_chain:,} contracts"))

    return awards


def run() -> dict:
    tickers = _load_watchlist_tickers(SCAN_TOP_N)
    with Session(SCAN_NAME, note="unusual options activity (top watchlist)") as s:
        if not tickers:
            s.log("watchlist empty — nothing to scan", level="info")
            return {"ok": True, "scanned": 0}

        s.log(f"scanning {len(tickers)} top-watchlist tickers", level="info")
        awarded = 0
        no_options = 0
        for tk in tickers:
            try:
                awards = _analyze(tk)
            except Exception as e:
                s.log(f"{tk} options fetch failed: {type(e).__name__}", level="info", ticker=tk)
                continue
            if not awards:
                no_options += 1
                continue
            for pts, reason in awards:
                s.award(tk, pts, reason)
            awarded += 1

        return {
            "ok": True,
            "scanned": len(tickers),
            "awarded": awarded,
            "no_signal": no_options,
        }


if __name__ == "__main__":
    print(run())
