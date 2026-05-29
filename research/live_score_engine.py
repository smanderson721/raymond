#!/usr/bin/env python3
"""
Live Score Engine — scoring substrate for the raymond livestream dashboard.

Every scan that runs on raymond calls into this module to:
  - emit per-event log entries  (renders as the terminal feed)
  - award points to tickers     (drives the top-10 watchlist)
  - flag "hits"                 (drives the big banner)
  - record scan-run metadata    (drives the status footer)

Scores decay exponentially (12-hour half-life by default) so the watchlist
breathes — old signals fade unless they're reinforced by fresh scans.

All output writes to ``data/live/*.json`` at the repo root so a GitHub
Pages dashboard can poll them with plain ``fetch()``.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR = os.path.join(REPO_ROOT, "data", "live")

SCORES_FILE = os.path.join(LIVE_DIR, "scores.json")
EVENTS_FILE = os.path.join(LIVE_DIR, "events.json")
HITS_FILE = os.path.join(LIVE_DIR, "hits.json")
WATCHLIST_FILE = os.path.join(LIVE_DIR, "watchlist.json")
STATUS_FILE = os.path.join(LIVE_DIR, "status.json")

# Tuning
HALF_LIFE_HOURS = 12.0
HALF_LIFE_SEC = HALF_LIFE_HOURS * 3600
HIT_THRESHOLD = 12          # single-event points >= this triggers a banner
MAX_EVENTS = 600            # rolling event log size
MAX_HITS = 60               # rolling hits log size
WATCHLIST_SIZE = 30         # we emit top-30, page can show top-10
ATTR_TAG_HALF_LIFE_HOURS = 24.0   # per-ticker attribute tags decay faster than score
ATTR_TAG_HALF_LIFE_SEC = ATTR_TAG_HALF_LIFE_HOURS * 3600
ATTR_TAG_MIN_WEIGHT = 0.15        # tags below this normalized weight get pruned
MAX_ATTR_TAGS_PER_TICKER = 24     # cap the per-ticker tag list

LEVELS = {"info": 1, "notable": 2, "hit": 3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_ts() -> float:
    return time.time()


def _ensure_dir():
    os.makedirs(LIVE_DIR, exist_ok=True)


def _read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path: str, data) -> None:
    _ensure_dir()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


def _decay(score: float, last_ts: float, now_ts: float) -> float:
    if last_ts <= 0 or score <= 0:
        return 0.0
    dt = max(0.0, now_ts - last_ts)
    return score * math.pow(0.5, dt / HALF_LIFE_SEC)


def _decay_all_scores(scores: dict, now_ts: float) -> dict:
    """Return a fresh scores dict with all values decayed to now_ts."""
    out = {}
    for tk, entry in scores.items():
        decayed = _decay(entry.get("score", 0.0), entry.get("last_ts", 0.0), now_ts)
        if decayed < 0.1:
            continue
        # also decay any attached attribute tags
        attrs = _decay_attrs(entry.get("attrs", []), now_ts)
        out[tk] = {
            "score": round(decayed, 3),
            "last_ts": entry.get("last_ts", now_ts),
            "events": entry.get("events", 0),
            "last_reason": entry.get("last_reason", ""),
            "last_scan": entry.get("last_scan", ""),
            "attrs": attrs,
        }
    return out


def _decay_attrs(attrs: list, now_ts: float) -> list:
    """Decay per-ticker attribute tags. Each tag is
    ``{key, label, scan, points, ts, weight}``. We rescale ``weight`` to the
    decay since each tag's ``ts`` and prune tags whose normalized weight
    falls below ``ATTR_TAG_MIN_WEIGHT``."""
    if not attrs:
        return []
    out = []
    for a in attrs:
        ts = a.get("ts", 0)
        base = max(0.0, float(a.get("points", 0)))
        if ts <= 0 or base <= 0:
            continue
        dt = max(0.0, now_ts - ts)
        w = base * math.pow(0.5, dt / ATTR_TAG_HALF_LIFE_SEC)
        if w < ATTR_TAG_MIN_WEIGHT:
            continue
        a2 = dict(a)
        a2["weight"] = round(w, 3)
        out.append(a2)
    # sort strongest tag first, cap list length
    out.sort(key=lambda x: x.get("weight", 0), reverse=True)
    return out[:MAX_ATTR_TAGS_PER_TICKER]


# ─── Public API ─────────────────────────────────────────────────────────


class Session:
    """One scan-run session. Accumulates events and score deltas, flushes on
    ``close()``. Always use as a context manager:

        with Session("market_pulse") as s:
            s.log("scanning VIX…")
            s.award("AAPL", 3, "rsi cross 50")
    """

    def __init__(self, scan: str, note: str = ""):
        self.scan = scan
        self.note = note
        self.started_at = _now_iso()
        self._t0 = _now_ts()
        self._events: list[dict] = []
        self._awards: dict[str, dict] = {}   # ticker -> {points, reason, scan}
        self._hits: list[dict] = []          # banner-eligible awards

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(error=str(exc) if exc else None)
        return False

    # ── log events ────────────────────────────────────────────
    def log(self, message: str, level: str = "info", ticker: Optional[str] = None,
            points: float = 0.0) -> None:
        ev = {
            "ts": _now_iso(),
            "scan": self.scan,
            "level": level,
            "ticker": ticker,
            "points": round(points, 2) if points else 0,
            "message": message,
        }
        self._events.append(ev)

    # ── award points ──────────────────────────────────────────
    def award(self, ticker: str, points: float, reason: str,
              level: Optional[str] = None,
              attr_key: Optional[str] = None) -> None:
        """Award ``points`` to ``ticker`` for ``reason``. ``attr_key`` (when
        supplied) is the short snake_case key from ``scan_weights.json`` —
        used to render the ticker's per-row attribute tag chips on the
        dashboard and to attribute the award back to a tunable weight."""
        if not ticker or points == 0:
            return
        ticker = ticker.upper()
        cur = self._awards.setdefault(
            ticker,
            {"points": 0.0, "reasons": [], "scan": self.scan,
             "best_points": float("-inf"), "best_reason": "",
             "attr_keys": []},
        )
        cur["points"] += float(points)
        cur["reasons"].append(reason)
        if attr_key:
            cur["attr_keys"].append((attr_key, float(points), reason))
        if float(points) > cur["best_points"]:
            cur["best_points"] = float(points)
            cur["best_reason"] = reason

        # auto-classify level
        lvl = level or ("hit" if points >= HIT_THRESHOLD
                        else "notable" if points >= 5
                        else "info")

        # log this as an event too
        sign = "+" if points >= 0 else ""
        self.log(
            f"{ticker} {sign}{points:.1f}  {reason}",
            level=lvl,
            ticker=ticker,
            points=points,
        )

        if points >= HIT_THRESHOLD:
            self._hits.append({
                "ts": _now_iso(),
                "ticker": ticker,
                "points": round(float(points), 2),
                "reason": reason,
                "scan": self.scan,
            })

    # ── flush ─────────────────────────────────────────────────
    def close(self, error: Optional[str] = None) -> dict:
        now = _now_ts()
        now_iso = _now_iso()
        duration = round(now - self._t0, 2)

        # 1) merge awards into scores.json (with decay)
        scores = _read(SCORES_FILE, {}).get("tickers", {})
        scores = _decay_all_scores(scores, now)
        for tk, a in self._awards.items():
            cur = scores.get(tk, {"score": 0.0, "events": 0,
                                  "last_reason": "", "last_scan": "",
                                  "attrs": []})
            cur["score"] = round(cur.get("score", 0.0) + a["points"], 3)
            cur["last_ts"] = now
            cur["events"] = cur.get("events", 0) + 1
            new_reason = a.get("best_reason") or (a["reasons"][-1] if a["reasons"] else "")
            # only overwrite last_reason if this run's best is a positive contribution,
            # otherwise keep the prior reason so the watchlist label stays informative
            if a.get("best_points", 0) > 0 or not cur.get("last_reason"):
                cur["last_reason"] = new_reason
            cur["last_scan"] = a["scan"]
            # append fresh attribute tags from this run (already decay-pruned
            # in _decay_all_scores above). De-dupe: if a tag with the same
            # attr_key + scan already exists, refresh its ts/points instead
            # of stacking duplicates.
            tags = cur.get("attrs", [])
            scan_id = a["scan"]
            for (attr_key, pts, reason_txt) in a.get("attr_keys", []):
                existing = next((t for t in tags
                                 if t.get("key") == attr_key and t.get("scan") == scan_id),
                                None)
                if existing:
                    existing["points"] = max(float(existing.get("points", 0)), float(pts))
                    existing["ts"] = now
                    existing["weight"] = float(existing["points"])
                else:
                    tags.append({
                        "key": attr_key,
                        "label": attr_key.replace("_", " "),
                        "scan": scan_id,
                        "points": float(pts),
                        "ts": now,
                        "weight": float(pts),
                    })
            cur["attrs"] = tags
            scores[tk] = cur
        _write(SCORES_FILE, {
            "updated_at": now_iso,
            "tickers": scores,
            "half_life_hours": HALF_LIFE_HOURS,
        })

        # 2) append events (rolling tail)
        events_blob = _read(EVENTS_FILE, {"events": []})
        events_blob["events"] = (events_blob.get("events", []) + self._events)[-MAX_EVENTS:]
        events_blob["updated_at"] = now_iso
        _write(EVENTS_FILE, events_blob)

        # 3) append hits (rolling tail)
        if self._hits:
            hits_blob = _read(HITS_FILE, {"hits": []})
            hits_blob["hits"] = (hits_blob.get("hits", []) + self._hits)[-MAX_HITS:]
            hits_blob["updated_at"] = now_iso
            _write(HITS_FILE, hits_blob)

        # 4) recompute top-N watchlist
        ranked = sorted(
            scores.items(),
            key=lambda kv: kv[1].get("score", 0.0),
            reverse=True,
        )[:WATCHLIST_SIZE]
        watchlist = [
            {
                "rank": i + 1,
                "ticker": tk,
                "score": entry.get("score", 0.0),
                "events": entry.get("events", 0),
                "last_reason": entry.get("last_reason", ""),
                "last_scan": entry.get("last_scan", ""),
                "last_ts": datetime.fromtimestamp(entry.get("last_ts", now),
                                                  tz=timezone.utc).isoformat(timespec="seconds"),
                "attrs": entry.get("attrs", []),
            }
            for i, (tk, entry) in enumerate(ranked)
        ]
        _write(WATCHLIST_FILE, {
            "updated_at": now_iso,
            "size": len(watchlist),
            "watchlist": watchlist,
        })

        # 5) status footer — append a row per scan
        status = _read(STATUS_FILE, {"runs": []})
        status["runs"] = (status.get("runs", []) + [{
            "scan": self.scan,
            "started_at": self.started_at,
            "duration_sec": duration,
            "events": len(self._events),
            "awards": len(self._awards),
            "hits": len(self._hits),
            "error": error,
            "note": self.note,
        }])[-200:]
        status["updated_at"] = now_iso
        _write(STATUS_FILE, status)

        return {
            "scan": self.scan,
            "duration": duration,
            "events": len(self._events),
            "awards": len(self._awards),
            "hits": len(self._hits),
        }


# ─── convenience for one-off non-session writes ────────────────────────

def emit_event(scan: str, message: str, level: str = "info") -> None:
    """Write a single event without opening a session (e.g. for startup logs)."""
    with Session(scan) as s:
        s.log(message, level=level)
