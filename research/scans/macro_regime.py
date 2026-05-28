#!/usr/bin/env python3
"""
Macro Regime Scan — environmental scan that runs every 15 minutes.

Pulls a small basket of macro tickers via yfinance and emits log events
+ awards modest points to the 11 SPDR sector ETFs based on relative
strength. The macro state itself (VIX level, 10y yield, 2s10s spread, SPY
trend) drives a multiplier consumed by per-ticker scans.

Outputs:
  - events appended to data/live/events.json
  - sector ETF points awarded to scores.json
  - data/live/macro_state.json (used by per-ticker scans)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import yfinance as yf

from research.live_score_engine import Session, LIVE_DIR, _write

MACRO_FILE = os.path.join(LIVE_DIR, "macro_state.json")

INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10y yield",
    "^IRX": "3m T-bill",
    "DX-Y.NYB": "Dollar Index",
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communications",
}


def _fetch_snapshot(tickers: list[str]) -> dict:
    """Return {ticker: {price, pct_1d, pct_5d, sma50, above_sma50, rsi14}}."""
    out: dict[str, dict] = {}
    df = yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )
    if df is None or len(df) == 0:
        return out
    for tk in tickers:
        try:
            sub = df[tk] if len(tickers) > 1 else df
            closes = sub["Close"].dropna()
            if len(closes) < 5:
                continue
            price = float(closes.iloc[-1])
            pct_1d = float(closes.pct_change().iloc[-1])
            pct_5d = float(closes.iloc[-1] / closes.iloc[-min(6, len(closes))] - 1)
            sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else None
            above_sma50 = (sma50 is not None) and (price > sma50)
            out[tk] = {
                "price": round(price, 4),
                "pct_1d": round(pct_1d, 5),
                "pct_5d": round(pct_5d, 5),
                "sma50": round(sma50, 4) if sma50 else None,
                "above_sma50": above_sma50,
            }
        except Exception:
            continue
    return out


def _classify_regime(snap: dict) -> dict:
    """Derive the macro regime + multiplier from the snapshot."""
    spy = snap.get("SPY", {})
    vix = snap.get("^VIX", {})
    tnx = snap.get("^TNX", {})
    irx = snap.get("^IRX", {})

    spy_above = spy.get("above_sma50", False)
    spy_pct_5d = spy.get("pct_5d") or 0.0
    vix_level = vix.get("price")
    vix_pct_1d = vix.get("pct_1d") or 0.0
    spread_2s10s = None
    if tnx.get("price") is not None and irx.get("price") is not None:
        spread_2s10s = round(tnx["price"] - irx["price"], 3)

    # naive regime label
    if vix_level and vix_level >= 25 and spy_pct_5d < -0.02:
        regime = "risk_off"
        mult = 0.7
    elif vix_level and vix_level <= 14 and spy_above and spy_pct_5d > 0.005:
        regime = "risk_on"
        mult = 1.3
    elif spy_above:
        regime = "constructive"
        mult = 1.1
    elif vix_level and vix_level >= 20:
        regime = "defensive"
        mult = 0.85
    else:
        regime = "neutral"
        mult = 1.0

    return {
        "regime": regime,
        "multiplier": mult,
        "spy_above_sma50": spy_above,
        "spy_pct_5d": round(spy_pct_5d, 5),
        "vix_level": vix_level,
        "vix_pct_1d": round(vix_pct_1d, 5),
        "spread_2s10s": spread_2s10s,
    }


def run() -> dict:
    with Session("macro_regime", note="env scan") as s:
        s.log("pulling macro basket (SPY/QQQ/IWM/VIX/yields/DXY + 11 SPDR sectors)")

        all_tickers = list(INDICES.keys()) + list(SECTOR_ETFS.keys())
        snap = _fetch_snapshot(all_tickers)

        if not snap:
            s.log("macro pull failed — yfinance returned empty", level="notable")
            return {"ok": False}

        # Log each macro index
        for tk, label in INDICES.items():
            d = snap.get(tk)
            if not d:
                continue
            sign = "+" if (d.get("pct_1d") or 0) >= 0 else ""
            s.log(
                f"{label:<14} {d['price']:>10.2f}  "
                f"{sign}{(d.get('pct_1d') or 0) * 100:+.2f}% 1d  "
                f"{(d.get('pct_5d') or 0) * 100:+.2f}% 5d"
            )

        # Sector RS — rank by 5d return, award points to leaders
        sectors = []
        for tk, label in SECTOR_ETFS.items():
            d = snap.get(tk)
            if not d:
                continue
            sectors.append((tk, label, d.get("pct_5d") or 0.0))
        sectors.sort(key=lambda x: x[2], reverse=True)

        s.log("─── sector RS (5d) ───")
        for i, (tk, label, pct) in enumerate(sectors):
            sign = "+" if pct >= 0 else ""
            mark = "🟢" if i < 3 else ("🔴" if i >= len(sectors) - 3 else "·")
            s.log(f"{mark} {tk:<5} {label:<24} {sign}{pct * 100:+.2f}%")
            # award small points to leaders, deduct from laggards
            if i == 0:
                s.award(tk, 4, f"sector leader 5d ({pct * 100:+.2f}%)")
            elif i == 1:
                s.award(tk, 2.5, f"#2 sector 5d ({pct * 100:+.2f}%)")
            elif i == 2:
                s.award(tk, 1.5, f"#3 sector 5d ({pct * 100:+.2f}%)")

        # Classify regime + persist macro_state.json
        regime = _classify_regime(snap)
        sign_v = "+" if regime["vix_pct_1d"] >= 0 else ""
        s.log(
            f"REGIME: {regime['regime'].upper()}  "
            f"mult={regime['multiplier']}  "
            f"VIX={regime['vix_level']} ({sign_v}{regime['vix_pct_1d'] * 100:+.2f}%)  "
            f"SPY {'above' if regime['spy_above_sma50'] else 'below'} 50d  "
            f"2s10s={regime['spread_2s10s']}",
            level="notable",
        )

        _write(MACRO_FILE, {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "snapshot": snap,
            **regime,
        })

    return {"ok": True, "regime": regime["regime"]}


if __name__ == "__main__":
    print(run())
