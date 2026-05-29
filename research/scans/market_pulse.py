#!/usr/bin/env python3
"""
Market Pulse — consolidated daily macro health scan.

Replaces the old ``macro_regime`` + ``macro_econ`` scans with a single
once-a-day pulse that computes a continuous **Market Health Score (0-100)**
from six weighted components, maps it to a weather label, and exposes a
score-driven multiplier that every per-ticker scan can apply to its
awards.

Inputs
------
- yfinance daily bars for SPY / QQQ / IWM / VIX / 10y / 3m / DXY / 11 SPDR sectors
- FRED daily/monthly series (DFF, DGS10, DGS2, CPI, UNRATE, PAYEMS) — used
  for the "recent release shock" component and the yield-curve component
  (FRED is preferred over yfinance ^TNX/^IRX because it survives weekend
  fetches and gives us the macro release calendar for free).

Components (weights sum to 100)
-------------------------------
    Trend       25   SPY position vs SMA50 / SMA200, 5-day slope
    Volatility  25   VIX level (inverse) + 5-day VIX change
    Breadth     20   % of 11 SPDR sectors trading above their own SMA50
    Yield curve 15   10y-2y spread (steep = healthy, inverted = recession risk)
    Dollar       5   DXY 20-day % change (inverse — strong USD = headwind)
    Releases    10   Recent CPI / NFP / UNRATE / Fed-rate surprises (decaying)

Score → weather label + multiplier
----------------------------------
    80-100  Clear skies   ×1.30
    60-80   Fair weather  ×1.15
    40-60   Overcast      ×1.00
    20-40   Stormy        ×0.85
     0-20   Severe        ×0.70
The multiplier itself is continuous: ``0.70 + 0.006 × score`` (0.70-1.30),
the bands are just labels for the dashboard banner.

Outputs
-------
- ``data/live/market_pulse.json`` — full snapshot consumed by the dashboard
- ``data/live/macro_state.json`` — slim multiplier file consumed by
  ``tech_slice._macro_multiplier()`` (and any future scan that wants to scale
  awards by the regime)
- A small set of log events (snapshot, sector awards, regime line).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yfinance as yf

from research.live_score_engine import Session, LIVE_DIR, _write
from research.scan_weights import weight

SCAN_NAME = "market_pulse"

PULSE_FILE = os.path.join(LIVE_DIR, "market_pulse.json")
MACRO_STATE_FILE = os.path.join(LIVE_DIR, "macro_state.json")
RELEASE_CACHE = Path(LIVE_DIR) / "_market_pulse_releases.json"

INDICES = {
    "SPY":      "S&P 500",
    "QQQ":      "Nasdaq 100",
    "IWM":      "Russell 2000",
    "^VIX":     "VIX",
    "DX-Y.NYB": "Dollar Index",
}

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discretionary",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Communications",
}

# FRED series fetched for yield-curve + release-shock components
FRED_SERIES = {
    "DFF":      ("Fed Funds Rate",       "%",   0.25),
    "DGS10":    ("10y Treasury Yield",   "%",   None),
    "DGS2":     ("2y Treasury Yield",    "%",   None),
    "CPIAUCSL": ("CPI",                  "idx", 0.30),
    "UNRATE":   ("Unemployment Rate",    "%",   0.20),
    "PAYEMS":   ("Nonfarm Payrolls",     "k",  50.00),
}

# ─── data fetch ──────────────────────────────────────────────────────


def _fetch_snapshot(tickers: list[str]) -> dict:
    """Per ticker: {price, pct_1d, pct_5d, pct_20d, sma50, sma200, above_sma50, above_sma200}."""
    out: dict[str, dict] = {}
    df = yf.download(
        tickers=tickers,
        period="1y",
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
            pct_1d  = float(closes.pct_change().iloc[-1])
            pct_5d  = float(closes.iloc[-1] / closes.iloc[-min(6, len(closes))] - 1)
            pct_20d = float(closes.iloc[-1] / closes.iloc[-min(21, len(closes))] - 1) if len(closes) >= 21 else 0.0
            sma50  = float(closes.tail(50).mean())  if len(closes) >= 50  else None
            sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None
            out[tk] = {
                "price": round(price, 4),
                "pct_1d":  round(pct_1d, 5),
                "pct_5d":  round(pct_5d, 5),
                "pct_20d": round(pct_20d, 5),
                "sma50":  round(sma50, 4)  if sma50  else None,
                "sma200": round(sma200, 4) if sma200 else None,
                "above_sma50":  (sma50  is not None) and (price > sma50),
                "above_sma200": (sma200 is not None) and (price > sma200),
            }
        except Exception:
            continue
    return out


def _fetch_fred(api_key: str) -> dict:
    """Return {sid: [observations]} (most-recent first, '.' filtered out)."""
    out: dict[str, list[dict]] = {}
    if not api_key:
        return out
    for sid in FRED_SERIES:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": sid,
                    "api_key": api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 6,
                },
                timeout=15,
            )
            if r.status_code != 200:
                continue
            obs = [o for o in (r.json().get("observations") or [])
                   if o.get("value") not in (None, "", ".")]
            if obs:
                out[sid] = obs
        except Exception:
            continue
    return out


# ─── component scoring (each returns 0-100) ──────────────────────────


def _score_trend(snap: dict) -> tuple[float, str]:
    spy = snap.get("SPY") or {}
    score = 0.0
    if spy.get("above_sma200"): score += 50
    if spy.get("above_sma50"):  score += 30
    if (spy.get("pct_5d") or 0) > 0: score += 20
    label = "trend ↑" if score >= 70 else ("trend mixed" if score >= 40 else "trend ↓")
    return min(100.0, score), label


def _score_vol(snap: dict) -> tuple[float, str]:
    vix = (snap.get("^VIX") or {}).get("price")
    if vix is None:
        return 50.0, "VIX —"
    # Linear ramp: VIX 12 → 100, VIX 35 → 0
    raw = max(0.0, min(100.0, 100.0 * (35.0 - vix) / 23.0))
    if   vix <= 14: tag = f"VIX {vix:.1f} calm"
    elif vix <= 20: tag = f"VIX {vix:.1f} mild"
    elif vix <= 28: tag = f"VIX {vix:.1f} elevated"
    else:           tag = f"VIX {vix:.1f} high"
    return raw, tag


def _score_breadth(snap: dict) -> tuple[float, str, int]:
    above = sum(1 for tk in SECTOR_ETFS if (snap.get(tk) or {}).get("above_sma50"))
    total = len(SECTOR_ETFS)
    return (100.0 * above / total if total else 50.0), f"breadth {above}/{total}", above


def _score_curve(fred: dict) -> tuple[float, str, float | None]:
    try:
        ten = float(fred["DGS10"][0]["value"])
        two = float(fred["DGS2"][0]["value"])
        spread = ten - two
    except (KeyError, IndexError, ValueError, TypeError):
        return 50.0, "curve —", None
    # spread = +1.5 → 100, spread = 0 → 40, spread = -1.0 → 0
    raw = max(0.0, min(100.0, 40.0 + spread * 40.0))
    bp = int(round(spread * 100))
    if spread < 0:    tag = f"curve {bp:+d}bp inverted"
    elif spread < 0.25: tag = f"curve {bp:+d}bp flat"
    else:             tag = f"curve {bp:+d}bp normal"
    return raw, tag, round(spread, 3)


def _score_dollar(snap: dict) -> tuple[float, str]:
    dxy_pct_20d = (snap.get("DX-Y.NYB") or {}).get("pct_20d")
    if dxy_pct_20d is None:
        return 50.0, "USD —"
    # +5% in 20d (strong dollar) → 10, -2% (weak dollar) → 80
    raw = max(0.0, min(100.0, 60.0 - dxy_pct_20d * 1000.0))
    if   dxy_pct_20d >  0.02: tag = f"USD +{dxy_pct_20d*100:.1f}% strengthening"
    elif dxy_pct_20d < -0.02: tag = f"USD {dxy_pct_20d*100:.1f}% weakening"
    else:                     tag = f"USD steady"
    return raw, tag


def _score_releases(fred: dict, cache: dict) -> tuple[float, str, list[dict]]:
    """Compare latest observation per series to the cached prior; reward
    growth-positive surprises, penalise growth-negative ones. Returns the
    score, a short summary tag, and the list of fresh release events."""
    score = 70.0          # neutral baseline
    fresh: list[dict] = []
    notable_tag = "no recent surprises"
    biggest_abs = 0.0

    for sid, (label, unit, threshold) in FRED_SERIES.items():
        obs = fred.get(sid)
        if not obs:
            continue
        try:
            latest_val  = float(obs[0]["value"])
            latest_date = obs[0]["date"]
        except (KeyError, ValueError, TypeError):
            continue
        prev = cache.get(sid) or {}
        prev_val  = prev.get("value")
        prev_date = prev.get("date")
        if prev_date and prev_date != latest_date and prev_val is not None:
            delta = latest_val - float(prev_val)
            fresh.append({
                "sid": sid, "label": label, "unit": unit,
                "date": latest_date, "value": latest_val,
                "prev_value": prev_val, "delta": round(delta, 3),
            })
            # apply per-series effect on the score
            if sid == "CPIAUCSL" and threshold and abs(delta) >= threshold:
                score -= 15 if delta > 0 else -5      # hot CPI bad, cool CPI mildly good
                if abs(delta) > biggest_abs:
                    biggest_abs = abs(delta)
                    notable_tag = "CPI hot" if delta > 0 else "CPI cool"
            elif sid == "DFF" and threshold and abs(delta) >= threshold:
                score += 15 if delta < 0 else -10     # rate cut good, hike bad
                notable_tag = "RATE CUT" if delta < 0 else "RATE HIKE"
            elif sid == "UNRATE" and threshold and abs(delta) >= threshold:
                score -= 15 if delta > 0 else -10     # unemployment rising bad, falling good
                notable_tag = "UNRATE up" if delta > 0 else "UNRATE down"
            elif sid == "PAYEMS" and threshold and abs(delta) >= threshold:
                score += 10 if delta > 0 else -10     # strong NFP good
                notable_tag = "NFP strong" if delta > 0 else "NFP weak"
        elif not prev_date:
            # first-ever read; just seed the cache (no scoring effect)
            pass
    score = max(0.0, min(100.0, score))
    return score, notable_tag, fresh


# ─── label + multiplier ──────────────────────────────────────────────


WEATHER_BANDS = [
    (80, "clear",    "Clear skies"),
    (60, "fair",     "Fair weather"),
    (40, "overcast", "Overcast"),
    (20, "stormy",   "Stormy"),
    ( 0, "severe",   "Severe"),
]


def _label_for(score: float) -> tuple[str, str]:
    for cutoff, key, name in WEATHER_BANDS:
        if score >= cutoff:
            return key, name
    return "severe", "Severe"


def _multiplier_for(score: float) -> float:
    """Linear ramp 0 → 0.70, 100 → 1.30."""
    return round(0.70 + 0.006 * score, 3)


# ─── runner ──────────────────────────────────────────────────────────


def _load_release_cache() -> dict:
    if RELEASE_CACHE.exists():
        try:
            return json.loads(RELEASE_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_release_cache(fred: dict) -> None:
    snap: dict = {}
    for sid, obs in fred.items():
        if obs:
            try:
                snap[sid] = {"value": float(obs[0]["value"]), "date": obs[0]["date"]}
            except (KeyError, ValueError, TypeError):
                continue
    RELEASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RELEASE_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap))
    tmp.replace(RELEASE_CACHE)


def run() -> dict:
    with Session(SCAN_NAME, note="daily market health pulse") as s:
        all_tickers = list(INDICES.keys()) + list(SECTOR_ETFS.keys())
        snap = _fetch_snapshot(all_tickers)
        if not snap:
            s.log("yfinance pull empty — skipping", level="notable")
            return {"ok": False, "reason": "no_snapshot"}

        api_key = (os.environ.get("FRED_API_KEY") or "").strip()
        fred = _fetch_fred(api_key)
        cache = _load_release_cache()

        # ── component scores ────────────────────────────────────────
        trend_s,   trend_tag   = _score_trend(snap)
        vol_s,     vol_tag     = _score_vol(snap)
        breadth_s, breadth_tag, sectors_above = _score_breadth(snap)
        curve_s,   curve_tag,   spread        = _score_curve(fred)
        usd_s,     usd_tag                    = _score_dollar(snap)
        rel_s,     rel_tag,     fresh_rels    = _score_releases(fred, cache)

        components = {
            "trend":      {"score": round(trend_s, 1),   "weight": 25, "tag": trend_tag},
            "volatility": {"score": round(vol_s, 1),     "weight": 25, "tag": vol_tag},
            "breadth":    {"score": round(breadth_s, 1), "weight": 20, "tag": breadth_tag},
            "yield_curve":{"score": round(curve_s, 1),   "weight": 15, "tag": curve_tag},
            "dollar":     {"score": round(usd_s, 1),     "weight":  5, "tag": usd_tag},
            "releases":   {"score": round(rel_s, 1),     "weight": 10, "tag": rel_tag},
        }
        total_weight = sum(c["weight"] for c in components.values())
        health = sum(c["score"] * c["weight"] for c in components.values()) / total_weight
        health = round(max(0.0, min(100.0, health)), 1)

        label_key, label_name = _label_for(health)
        mult = _multiplier_for(health)

        # ── essential log lines (kept deliberately short) ───────────
        spy = snap.get("SPY", {})
        vix = snap.get("^VIX", {})
        s.log(
            f"SPY {spy.get('price', 0):.2f} "
            f"({(spy.get('pct_1d') or 0)*100:+.2f}% 1d, "
            f"{(spy.get('pct_5d') or 0)*100:+.2f}% 5d) · "
            f"VIX {vix.get('price', 0):.2f} · "
            f"curve {curve_tag.split()[1] if 'curve' in curve_tag else '—'} · "
            f"breadth {sectors_above}/{len(SECTOR_ETFS)}",
        )
        s.log(
            f"HEALTH {health:.0f}/100 · {label_name.upper()} · ×{mult}",
            level="notable",
        )

        # ── sector ETF awards (top 3 by 5d return) ──────────────────
        sectors = []
        for tk in SECTOR_ETFS:
            d = snap.get(tk)
            if not d:
                continue
            sectors.append((tk, SECTOR_ETFS[tk], d.get("pct_5d") or 0.0))
        sectors.sort(key=lambda x: x[2], reverse=True)

        leader_pts  = weight(SCAN_NAME, "sector_leader",  4.0)
        second_pts  = weight(SCAN_NAME, "sector_second",  2.5)
        third_pts   = weight(SCAN_NAME, "sector_third",   1.5)

        if len(sectors) >= 3:
            (tk1, lbl1, p1), (tk2, lbl2, p2), (tk3, lbl3, p3) = sectors[:3]
            s.award(tk1, leader_pts, f"sector leader 5d ({p1*100:+.2f}%) {lbl1}", attr_key="sector_leader")
            s.award(tk2, second_pts, f"#2 sector 5d ({p2*100:+.2f}%) {lbl2}",     attr_key="sector_second")
            s.award(tk3, third_pts,  f"#3 sector 5d ({p3*100:+.2f}%) {lbl3}",     attr_key="sector_third")
            s.log(f"sector RS: 🟢 {tk1} {p1*100:+.2f}% · {tk2} {p2*100:+.2f}% · {tk3} {p3*100:+.2f}%")

        # ── persist outputs ─────────────────────────────────────────
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        pulse = {
            "updated_at":   now_iso,
            "health_score": health,
            "label_key":    label_key,        # "clear" | "fair" | "overcast" | "stormy" | "severe"
            "label":        label_name,       # "Clear skies" | …
            "multiplier":   mult,
            "components":   components,
            "left_tags":    [trend_tag, breadth_tag, vol_tag],
            "right_tags":   [curve_tag, usd_tag, rel_tag],
            "spy_above_sma50":  spy.get("above_sma50", False),
            "spy_above_sma200": spy.get("above_sma200", False),
            "vix_level":   (vix.get("price")),
            "spread_2s10s": spread,
            "fresh_releases": fresh_rels,
            "sectors_above": sectors_above,
            "snapshot":    snap,
        }
        _write(PULSE_FILE, pulse)

        # slim file for any scan that just wants the multiplier
        _write(MACRO_STATE_FILE, {
            "updated_at":   now_iso,
            "multiplier":   mult,
            "health_score": health,
            "label_key":    label_key,
            "label":        label_name,
            # legacy keys kept so older consumers don't break before they
            # migrate to market_pulse.json:
            "regime":      label_key,
            "vix_level":   vix.get("price"),
            "spy_above_sma50": spy.get("above_sma50", False),
        })
        _save_release_cache(fred)

        return {"ok": True, "health_score": health, "label": label_name, "multiplier": mult}


if __name__ == "__main__":
    print(run())
