#!/usr/bin/env python3
"""
Tech Scan Slice — rolling per-cycle pass over the precondition-scan
universe. Each invocation processes a slice of ~SLICE_SIZE tickers (next
chunk in a rotating window), pulls 6mo of daily bars via yfinance,
computes a panel of indicators locally, identifies tradable setups, and
awards points to the live score engine.

Designed to be invoked every 10–15 minutes by a GitHub Actions cron. With
SLICE_SIZE=400 and ~6000 tickers, the universe is fully covered every ~15
runs (≈ 4 hours). This produces continuous activity for the dashboard
without any one run exceeding a few minutes wall time.

Rotation cursor is persisted at data/live/_slice_cursor.json.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from research.live_score_engine import Session, LIVE_DIR, _read, _write
from research.scan_weights import weight

SCAN_NAME = "tech_slice"
CURSOR_FILE = os.path.join(LIVE_DIR, "_slice_cursor.json")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN_FILE = os.path.join(REPO_ROOT, "research_output", "scan_results.json")

SLICE_SIZE = 400
BATCH = 80          # yfinance batch size within the slice
PERIOD = "6mo"


# ─── universe ────────────────────────────────────────────────────────

def _load_universe() -> list[str]:
    if not os.path.exists(SCAN_FILE):
        return []
    try:
        with open(SCAN_FILE) as f:
            data = json.load(f)
    except Exception:
        return []
    scans = data.get("scans") or []
    if not scans:
        return []
    latest = scans[-1].get("stocks") or []
    out = []
    for s in latest:
        tk = s.get("ticker") or s.get("symbol")
        if tk:
            out.append(tk.upper())
    return out


def _next_slice(universe: list[str]) -> tuple[list[str], int, int]:
    cursor = _read(CURSOR_FILE, {"cursor": 0}).get("cursor", 0)
    n = len(universe)
    if n == 0:
        return [], 0, 0
    cursor = cursor % n
    end = cursor + SLICE_SIZE
    if end <= n:
        sl = universe[cursor:end]
    else:
        sl = universe[cursor:] + universe[: end - n]
    next_cursor = end % n
    _write(CURSOR_FILE, {"cursor": next_cursor, "universe_size": n,
                         "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return sl, cursor, next_cursor


# ─── indicators (vectorised) ─────────────────────────────────────────

def _rsi14(c: pd.Series) -> float:
    if len(c) < 16:
        return float("nan")
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _atr14_pct(h: pd.Series, l: pd.Series, c: pd.Series) -> float:
    if len(c) < 16:
        return float("nan")
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    return float(atr / c.iloc[-1])


def _features(df: pd.DataFrame) -> dict:
    c = df["Close"].astype(float).dropna()
    h = df["High"].astype(float).dropna()
    l = df["Low"].astype(float).dropna()
    v = df["Volume"].astype(float).dropna()
    if len(c) < 30:
        return {}

    price = float(c.iloc[-1])
    pct_1d = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) >= 2 else 0.0
    pct_5d = float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else 0.0
    pct_20d = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else 0.0

    sma20 = float(c.tail(20).mean())
    sma50 = float(c.tail(50).mean()) if len(c) >= 50 else None
    high20 = float(c.tail(20).max())

    v5 = float(v.tail(5).mean()) if len(v) >= 5 else 0.0
    v60_prior = float(v.iloc[-60:-5].mean()) if len(v) >= 60 else 0.0
    vol_ratio = (v5 / v60_prior) if v60_prior > 0 else None

    # Daily RVOL: today's session volume vs prior 20-day average.
    # This is the broad-universe relative-volume signal that lets us
    # discover unusual volume on tickers that aren't yet on the watchlist
    # (the Alpaca rvol_stream daemon only covers the top-25 watchlist).
    today_vol = float(v.iloc[-1]) if len(v) >= 1 else 0.0
    avg20_prior = float(v.iloc[-21:-1].mean()) if len(v) >= 21 else 0.0
    rvol_daily = (today_vol / avg20_prior) if avg20_prior > 0 else None

    dollar_vol = price * (float(v.tail(20).mean()) if len(v) >= 20 else float(v.mean()))

    rsi = _rsi14(c)
    atr_pct = _atr14_pct(h, l, c)

    # Bollinger width (20-period)
    if len(c) >= 20:
        m = c.tail(20).mean()
        s = c.tail(20).std()
        bb_width = (4 * s / m) if m > 0 else None
    else:
        bb_width = None

    # squeeze: BB width below 6th percentile of last 60d widths
    squeeze = False
    if len(c) >= 60:
        widths = []
        for i in range(20, len(c) + 1):
            window = c.iloc[i - 20:i]
            mu = window.mean()
            sd = window.std()
            if mu > 0 and not math.isnan(sd):
                widths.append(4 * sd / mu)
        if widths:
            p = np.percentile(widths[-60:], 12)
            squeeze = (bb_width is not None) and (bb_width <= p)

    return {
        "price": price,
        "pct_1d": pct_1d,
        "pct_5d": pct_5d,
        "pct_20d": pct_20d,
        "sma20": sma20,
        "sma50": sma50,
        "high20": high20,
        "vol_ratio_5_60": vol_ratio,
        "rvol_daily": rvol_daily,
        "dollar_vol": dollar_vol,
        "rsi14": rsi,
        "atr14_pct": atr_pct,
        "bb_width": bb_width,
        "bb_squeeze": squeeze,
    }


# ─── setup scoring ───────────────────────────────────────────────────

def _evaluate(tk: str, f: dict, mult: float) -> list[tuple[float, str, str]]:
    """Return list of (points, reason, attr_key) for setups detected on
    this ticker. ``attr_key`` is the scan_weights.json key, which the
    engine uses to tag the ticker on the dashboard marquee.

    Multiplier comes from macro_state.json so risk-off shaves all
    positive awards.
    """
    out: list[tuple[float, str, str]] = []

    if not f or f.get("dollar_vol", 0) < 1_000_000:
        return out  # illiquid

    price = f["price"]
    pct_1d = f.get("pct_1d", 0.0)
    pct_5d = f.get("pct_5d", 0.0)
    pct_20d = f.get("pct_20d", 0.0)
    rsi = f.get("rsi14") or 50
    atr_pct = f.get("atr14_pct") or 0.02
    vol_ratio = f.get("vol_ratio_5_60")
    rvol_daily = f.get("rvol_daily")
    sma50 = f.get("sma50")
    high20 = f.get("high20")
    bb_squeeze = f.get("bb_squeeze")

    # 1) 20-day-high breakout on volume
    if high20 and price >= high20 * 0.999 and vol_ratio and vol_ratio > 1.5:
        base = weight(SCAN_NAME, "breakout_20d_high_vol", 10.0)
        pts = base + min(8, (vol_ratio - 1.5) * 4)
        out.append((pts, f"20d high break on {vol_ratio:.1f}× vol", "breakout_20d_high_vol"))

    # 2) Bollinger squeeze release with momentum
    if bb_squeeze and abs(pct_1d) > atr_pct * 1.2:
        pts = weight(SCAN_NAME, "squeeze_release_momentum", 8.0)
        out.append((pts, f"squeeze release ({pct_1d * 100:+.1f}%, ATR={atr_pct * 100:.1f}%)", "squeeze_release_momentum"))

    # 3) uptrend pullback to SMA50 with bullish RSI cross
    if sma50 and abs(price - sma50) / sma50 < 0.02 and 45 <= rsi <= 55 and pct_20d > 0.02:
        pts = weight(SCAN_NAME, "pullback_sma50_rsi_cross", 6.0)
        out.append((pts, f"pullback to 50d, RSI {rsi:.0f}, +{pct_20d * 100:.1f}% 20d", "pullback_sma50_rsi_cross"))

    # 4) strong stepped-up volume w/o blowoff
    if vol_ratio and vol_ratio > 2.0 and abs(pct_5d) < 0.20:
        pts = weight(SCAN_NAME, "stepped_volume_no_blowoff", 5.0)
        out.append((pts, f"volume {vol_ratio:.1f}× without blowoff", "stepped_volume_no_blowoff"))

    # 5) coil: low ATR but in uptrend
    if atr_pct and atr_pct < 0.015 and sma50 and price > sma50 and pct_20d > 0:
        pts = weight(SCAN_NAME, "coiling_above_sma50", 4.0)
        out.append((pts, f"coiling above 50d, ATR {atr_pct * 100:.2f}%", "coiling_above_sma50"))

    # 6) clean momentum, not yet extended
    if 55 <= rsi <= 68 and pct_5d > 0.03 and pct_20d > 0.05 and pct_20d < 0.30:
        pts = weight(SCAN_NAME, "clean_momentum_trend", 4.0)
        out.append((pts, f"clean trend RSI {rsi:.0f}, {pct_20d * 100:+.1f}% 20d", "clean_momentum_trend"))

    # 7-9) Daily RVOL tiers — broad-universe relative volume detection.
    # rvol_stream (Alpaca IEX) only covers the top-25 watchlist; this is
    # where rvol gets discovered on the other 5700+ tickers.
    if rvol_daily and rvol_daily >= 5.0:
        pts = weight(SCAN_NAME, "rvol_daily_mega", 12.0)
        out.append((pts, f"daily RVOL {rvol_daily:.1f}× (mega volume vs 20d avg)", "rvol_daily_mega"))
    elif rvol_daily and rvol_daily >= 3.0:
        pts = weight(SCAN_NAME, "rvol_daily_strong", 8.0)
        out.append((pts, f"daily RVOL {rvol_daily:.1f}× (strong volume vs 20d avg)", "rvol_daily_strong"))
    elif rvol_daily and rvol_daily >= 2.0:
        pts = weight(SCAN_NAME, "rvol_daily_notable", 5.0)
        out.append((pts, f"daily RVOL {rvol_daily:.1f}× vs 20d avg", "rvol_daily_notable"))

    # Penalties — subtract points from clearly blown-up names
    if pct_5d > 0.30 or rsi > 80:
        pts = weight(SCAN_NAME, "overextended_penalty", -6.0)
        out.append((pts, f"overextended (RSI {rsi:.0f}, +{pct_5d * 100:.0f}% 5d)", "overextended_penalty"))

    # Apply macro multiplier to positive awards only
    scaled = []
    for pts, reason, attr_key in out:
        if pts > 0:
            scaled.append((round(pts * mult, 2), reason, attr_key))
        else:
            scaled.append((round(pts, 2), reason, attr_key))
    return scaled


# ─── runner ──────────────────────────────────────────────────────────

def _macro_multiplier() -> float:
    # Prefer the new consolidated market_pulse.json; fall back to the
    # legacy macro_state.json for backward compatibility during rollout.
    state = _read(os.path.join(LIVE_DIR, "market_pulse.json"), None)
    if not state:
        state = _read(os.path.join(LIVE_DIR, "macro_state.json"), {})
    return float(state.get("multiplier", 1.0))


def _yf_batch(symbols: list[str]) -> dict:
    """Download once for the batch and split per ticker."""
    if not symbols:
        return {}
    df = yf.download(
        tickers=" ".join(symbols),
        period=PERIOD,
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )
    out = {}
    if df is None or len(df) == 0:
        return out
    for tk in symbols:
        try:
            sub = df[tk] if len(symbols) > 1 else df
            if isinstance(sub, pd.DataFrame) and len(sub.dropna(how="all")) > 0:
                out[tk] = sub.dropna(how="all")
        except Exception:
            continue
    return out


def run() -> dict:
    universe = _load_universe()
    if not universe:
        with Session("tech_slice") as s:
            s.log("no precondition-scan universe found at "
                  "research_output/scan_results.json — skipping",
                  level="notable")
        return {"ok": False, "reason": "no_universe"}

    slice_, start, end = _next_slice(universe)
    mult = _macro_multiplier()

    with Session("tech_slice",
                 note=f"slice {start}..{end} of {len(universe)} mult={mult}") as s:
        s.log(f"scanning slice {start} → {end} of {len(universe)} (×{mult} macro)")

        ok = 0
        fail = 0
        awards = 0

        # process in batches inside the slice
        for i in range(0, len(slice_), BATCH):
            batch = slice_[i:i + BATCH]
            frames = _yf_batch(batch)
            for tk in batch:
                df = frames.get(tk)
                if df is None or len(df) < 30:
                    fail += 1
                    continue
                ok += 1
                f = _features(df)
                if not f:
                    continue
                hits = _evaluate(tk, f, mult)
                for pts, reason, attr_key in hits:
                    s.award(tk, pts, reason, attr_key=attr_key)
                    awards += 1

        s.log(f"slice done — ok={ok} fail={fail} awards={awards}")

    return {"ok": True, "scanned": ok, "failed": fail, "awards": awards,
            "slice_start": start, "slice_end": end}


if __name__ == "__main__":
    print(run())
