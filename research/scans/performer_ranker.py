"""
performer_ranker.py — broad-universe top-performer attribution scan.

This is the heart of the **market-feedback weight system**. It ranks the
entire scanned universe by trailing 30-day return (sourced from
``data/live/_broad_prices.json``, which ``tech_slice`` keeps fresh as
it rotates through ~6000 tickers) and detects the moment a stock
**enters** the top 5, 20, 100, or 250 performer tier. At each
transition, every attribute the stock currently exhibits (from
``data/live/ticker_attrs.json``) is awarded a tier bonus:

    enters top 250 → +1 per current attribute
    enters top 100 → +3 per current attribute
    enters top  20 → +9 per current attribute
    enters top   5 → +27 per current attribute

Bonuses are stamped onto ``data/scan_weights.json`` via
:func:`research.scan_weights.award_attribute_bonus`, which applies a
1-week half-life decay to the previously-stored value before adding
the bonus and bumps the per-attribute timestamp so decay restarts from
the moment of the award. The net effect: attribute reward values
bootstrap from 0 and drift toward the levels that genuinely
characterize recent top performers, with old market regimes decaying
away on a 1-week half-life.

Tier-entry detection is anchored to the previous tier snapshot at
``data/live/_performer_ranks.json``. To avoid awarding the same
stock multiple bonuses for the same upward move (e.g. a stock that
goes from rank #300 to #4 should only get the +27 top-5 bonus, not
+1+3+9+27), tier membership is tracked **inclusively** (top-5 ⊂ top-20
⊂ top-100 ⊂ top-250) and entry is detected as ``in current AND not in
previous``. The bonus awarded per entrant is the largest tier they
just entered.

Cadence: every 15 minutes, offset to fire after ``tech_slice`` has
likely refreshed at least one new batch of prices.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from research.live_score_engine import Session, LIVE_DIR, _read, _write
from research.scan_weights import award_attribute_bonus

SCAN_NAME = "performer_ranker"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BROAD_PRICES_FILE   = os.path.join(LIVE_DIR, "_broad_prices.json")
TICKER_ATTRS_FILE   = os.path.join(LIVE_DIR, "ticker_attrs.json")
PERFORMER_RANKS_FILE = os.path.join(LIVE_DIR, "_performer_ranks.json")
PERFORMERS_FILE     = os.path.join(LIVE_DIR, "performers.json")

# Tier bonus mapping: { tier_size : bonus_per_attribute }
TIERS: list[tuple[int, float]] = [
    (5,    27.0),
    (20,    9.0),
    (100,   3.0),
    (250,   1.0),
]

# Minimum number of priced tickers required before we'll publish a
# ranking. Below this the universe coverage is too thin to be a
# meaningful "top of the market" signal (the broad_prices file is
# accumulated across slice runs, so it starts small after a daemon
# restart).
MIN_RANKED = 200

# Maximum age of a price observation that's still allowed to count
# toward the ranking. Anything older is treated as missing.
PRICE_FRESHNESS_SEC = 3 * 24 * 3600  # 3 days


def _load_broad_prices(now_ts: float) -> dict[str, float]:
    """Return ``{ticker: pct_30d}`` for all tickers with fresh prices."""
    blob = _read(BROAD_PRICES_FILE, {"tickers": {}})
    out: dict[str, float] = {}
    cutoff = now_ts - PRICE_FRESHNESS_SEC
    for tk, row in (blob.get("tickers") or {}).items():
        try:
            pct = float(row.get("pct_30d"))
            ts = float(row.get("fetched_ts") or 0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pct) or ts < cutoff:
            continue
        out[tk.upper()] = pct
    return out


def _load_ticker_attrs() -> dict[str, list[dict]]:
    """Return ``{ticker: [ {scan, key, last_seen_ts, ...}, ... ]}``."""
    blob = _read(TICKER_ATTRS_FILE, {"tickers": {}})
    out: dict[str, list[dict]] = {}
    for tk, row in (blob.get("tickers") or {}).items():
        attrs = row.get("attrs") or []
        if attrs:
            out[tk.upper()] = attrs
    return out


def _load_prev_ranks() -> dict[str, list[str]]:
    """Previous tier memberships keyed by tier size as a string.
    Returns ``{"5": [...], "20": [...], "100": [...], "250": [...]}``."""
    blob = _read(PERFORMER_RANKS_FILE, {})
    tiers = blob.get("tiers") or {}
    out = {}
    for size, _ in TIERS:
        out[str(size)] = list(tiers.get(str(size)) or [])
    return out


def _save_ranks(tier_sets: dict[int, list[str]], universe_size: int,
                now_iso: str) -> None:
    blob = {
        "updated_at": now_iso,
        "universe_size": universe_size,
        "tiers": {str(size): list(members) for size, members in tier_sets.items()},
    }
    _write(PERFORMER_RANKS_FILE, blob)


def _save_performers_dashboard(ranked: list[tuple[str, float]],
                                broad: dict[str, float],
                                ticker_attrs: dict[str, list[dict]],
                                now_iso: str) -> None:
    """Write the top-30 performer leaderboard for the dashboard's
    Top Performers column."""
    top = ranked[:30]
    rows = []
    for rank, (tk, pct) in enumerate(top, 1):
        attrs = ticker_attrs.get(tk, [])
        # condense attrs for the dashboard payload — only scan + key
        # are needed for color coding and tag rendering
        rows.append({
            "rank": rank,
            "ticker": tk,
            "pct_30d": round(pct, 5),
            "attrs": [
                {"scan": a.get("scan"), "key": a.get("key"),
                 "label": (a.get("key") or "").replace("_", " ")}
                for a in attrs
            ],
        })
    _write(PERFORMERS_FILE, {
        "updated_at": now_iso,
        "universe_size": len(broad),
        "performers": rows,
    })


def run() -> dict:
    now_ts = datetime.now(timezone.utc).timestamp()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with Session(SCAN_NAME, note="broad-universe performer ranker") as s:
        broad = _load_broad_prices(now_ts)
        if len(broad) < MIN_RANKED:
            s.log(
                f"broad price universe still warming up "
                f"({len(broad)}/{MIN_RANKED} priced tickers) — skipping rank update",
                level="info",
            )
            return {"ok": False, "reason": "warming_up",
                    "priced": len(broad), "required": MIN_RANKED}

        # 1) rank the universe by trailing 30d return
        ranked = sorted(broad.items(), key=lambda kv: kv[1], reverse=True)

        # 2) build current tier membership (inclusive: top-5 ⊂ top-20 ⊂ …)
        tier_sets: dict[int, list[str]] = {}
        for size, _ in TIERS:
            tier_sets[size] = [tk for tk, _ in ranked[:size]]

        # 3) load previous tier snapshot
        prev = _load_prev_ranks()
        prev_sets: dict[int, set[str]] = {
            size: set(prev.get(str(size), [])) for size, _ in TIERS
        }
        cur_sets: dict[int, set[str]] = {
            size: set(members) for size, members in tier_sets.items()
        }

        # 4) per-ticker "best tier they just entered" (largest = top-5 = +27)
        # Walk tiers from smallest size (top 5) to largest (top 250) so the
        # first match per ticker is the highest tier they newly entered.
        best_tier_for_entrant: dict[str, tuple[int, float]] = {}
        for size, bonus in TIERS:  # already ordered 5,20,100,250
            entrants = cur_sets[size] - prev_sets[size]
            for tk in entrants:
                if tk not in best_tier_for_entrant:
                    best_tier_for_entrant[tk] = (size, bonus)

        if not best_tier_for_entrant:
            s.log(
                f"no tier transitions this run (universe={len(broad)}, "
                f"top-5: {', '.join(tier_sets[5])})",
                level="info",
            )
            # still persist snapshot so deltas next run are correct
            _save_ranks(tier_sets, len(broad), now_iso)
            _save_performers_dashboard(
                ranked, broad, _load_ticker_attrs(), now_iso
            )
            return {"ok": True, "universe_size": len(broad), "entrants": 0, "bonuses_awarded": 0}

        # 5) for each new entrant, look up their attrs and award bonuses
        ticker_attrs = _load_ticker_attrs()
        bonuses_awarded = 0
        attrs_credited = 0
        entrants_without_attrs = 0
        per_attr_summary: dict[str, float] = {}

        for tk, (tier_size, bonus) in best_tier_for_entrant.items():
            attrs = ticker_attrs.get(tk, [])
            if not attrs:
                entrants_without_attrs += 1
                s.log(
                    f"{tk} entered top-{tier_size} (+{bonus:.0f} per attr) "
                    f"but has no attributes on record — no bonus credited",
                    level="info",
                    ticker=tk,
                )
                continue
            credited_keys = []
            for a in attrs:
                scan_id = a.get("scan")
                attr_key = a.get("key")
                if not scan_id or not attr_key:
                    continue
                reason = (f"{tk} entered top-{tier_size} performers "
                          f"({broad[tk]*100:+.1f}% 30d) — possessing this attribute "
                          f"earned +{bonus:.0f}")
                result = award_attribute_bonus(scan_id, attr_key, bonus, reason=reason)
                if result is None:
                    continue
                bonuses_awarded += 1
                attrs_credited += 1
                credited_keys.append(f"{scan_id}.{attr_key}")
                per_attr_summary[f"{scan_id}.{attr_key}"] = (
                    per_attr_summary.get(f"{scan_id}.{attr_key}", 0.0) + bonus
                )
            if credited_keys:
                s.log(
                    f"{tk} entered top-{tier_size} ({broad[tk]*100:+.1f}% 30d) "
                    f"→ +{bonus:.0f} to {len(credited_keys)} attr(s): "
                    f"{', '.join(credited_keys[:6])}"
                    + ("…" if len(credited_keys) > 6 else ""),
                    level="notable" if tier_size <= 20 else "info",
                    ticker=tk,
                    points=bonus,
                )

        # 6) persist new tier snapshot + dashboard payload
        _save_ranks(tier_sets, len(broad), now_iso)
        _save_performers_dashboard(ranked, broad, ticker_attrs, now_iso)

        s.log(
            f"rank update complete — universe={len(broad)} "
            f"entrants={len(best_tier_for_entrant)} "
            f"bonuses_awarded={bonuses_awarded} "
            f"entrants_without_attrs={entrants_without_attrs} "
            f"top-5: {', '.join(tier_sets[5])}",
            level="notable" if bonuses_awarded else "info",
        )

        return {
            "ok": True,
            "universe_size": len(broad),
            "entrants": len(best_tier_for_entrant),
            "bonuses_awarded": bonuses_awarded,
            "attrs_credited": attrs_credited,
            "entrants_without_attrs": entrants_without_attrs,
        }


if __name__ == "__main__":
    print(run())
