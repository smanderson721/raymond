#!/usr/bin/env python3
"""
scan_weights.py — central loader for per-scan attribute reward values.

Every scan calls ``weight(scan_id, attribute_key, default)`` to look up
the live award value for one of its attributes. The first call (per
process) reads ``data/scan_weights.json``; subsequent calls re-read
only if the file's mtime has changed, so an edit via the
``PUT /api/scan-weights`` endpoint takes effect on the very next scan
run — no daemon restart required.

The JSON shape is::

    {
      "_meta": { ... },
      "scans": {
        "<scan_id>": {
          "label": "...",
          "live_editable": true|false,
          "attributes": {
            "<key>": {
              "points": 5.0,
              "points_updated_ts": 1234567890.0,
              "lifetime_bonus": 27.0,
              "description": "...",
              "status": "active"
            },
            ...
          },
          "multipliers": { "<key>": { "value": 1.1, ... } },
          "thresholds":  { "<key>": { "value": 3.0, ... } }
        }
      }
    }

If the file is missing, malformed, or the requested key is absent, the
caller's ``default`` is returned — so removing or breaking the JSON
file never crashes a scan; it just falls back to compile-time defaults.

Lazy half-life decay
====================

As of the market-feedback weight system (2026), attribute reward values
are no longer hand-tuned — they bootstrap from 0 and only grow when
top-performing stocks (top 5/20/100/250 by trailing 30-day return)
possess the attribute at the moment they enter a tier. Bonuses awarded
to an attribute decay continuously with a **1-week half-life** so the
system stays responsive to recent market dynamics.

Decay is computed lazily on every read (and on every bonus award) using
each attribute's ``points_updated_ts`` stamp::

    decayed = stored_points * 0.5 ** ((now - points_updated_ts) / 1_week)

Reading via :func:`weight` returns the decayed value without rewriting
the file. Writing via :func:`award_attribute_bonus` applies decay
first, then adds the bonus, then bumps ``points_updated_ts`` to
``now`` so the next decay window starts fresh.

PUT /api/scan-weights manual edits stamp ``points_updated_ts`` in the
daemon handler so an operator-typed value is treated as the current
freshly-decayed value (it then decays from there).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = os.path.join(REPO_ROOT, "data", "scan_weights.json")

# 1-week half-life for attribute reward decay.
ATTR_HALF_LIFE_SEC = 7 * 24 * 3600  # 604800

_lock = threading.Lock()
_cache: dict = {}
_mtime: float = 0.0


def _load() -> dict:
    """Re-read the JSON file if its mtime has changed since last load.
    Thread-safe; on any IO/parse error returns the last successful cache
    (or {} if nothing has loaded yet) so callers always get a usable map."""
    global _cache, _mtime
    try:
        st = os.stat(WEIGHTS_FILE)
    except FileNotFoundError:
        return _cache
    if st.st_mtime == _mtime and _cache:
        return _cache
    with _lock:
        # re-check inside lock to avoid duplicate reads
        try:
            st = os.stat(WEIGHTS_FILE)
        except FileNotFoundError:
            return _cache
        if st.st_mtime == _mtime and _cache:
            return _cache
        try:
            with open(WEIGHTS_FILE, "r") as f:
                fresh = json.load(f)
            _cache = fresh
            _mtime = st.st_mtime
        except Exception:
            # keep prior cache on parse error
            pass
    return _cache


def _decayed_points(entry: dict, now_ts: float | None = None) -> float:
    """Apply lazy 1-week half-life decay to an attribute entry.

    Returns the decayed point value. Does **not** mutate the entry.
    An entry without ``points_updated_ts`` is treated as fresh (no decay).
    """
    try:
        pts = float(entry.get("points", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if pts <= 0:
        return 0.0
    upd = entry.get("points_updated_ts")
    if upd in (None, 0, 0.0):
        return pts
    try:
        upd_ts = float(upd)
    except (TypeError, ValueError):
        return pts
    if now_ts is None:
        now_ts = time.time()
    dt = max(0.0, now_ts - upd_ts)
    if dt == 0.0:
        return pts
    return pts * math.pow(0.5, dt / ATTR_HALF_LIFE_SEC)


def weight(scan_id: str, key: str, default: float) -> float:
    """Return decayed points for ``scan_id.attributes[key]``, or ``default`` if missing."""
    data = _load()
    try:
        entry = data["scans"][scan_id]["attributes"][key]
    except (KeyError, TypeError):
        return float(default)
    if not isinstance(entry, dict):
        return float(default)
    try:
        return float(_decayed_points(entry))
    except (TypeError, ValueError):
        return float(default)


def threshold(scan_id: str, key: str, default: float) -> float:
    """Return a threshold/tuning value from ``scan_id.thresholds[key]``."""
    data = _load()
    try:
        return float(data["scans"][scan_id]["thresholds"][key]["value"])
    except (KeyError, TypeError, ValueError):
        return float(default)


def multiplier(scan_id: str, key: str, default: float) -> float:
    """Return a multiplier value from ``scan_id.multipliers[key]``."""
    data = _load()
    try:
        return float(data["scans"][scan_id]["multipliers"][key]["value"])
    except (KeyError, TypeError, ValueError):
        return float(default)


def all_weights() -> dict:
    """Return the full weights dict (for the GET /api/scan-weights endpoint).

    Each ``attributes[key]`` block is augmented with two derived (non-
    persisted) fields so the dashboard can show real-time decayed values
    without re-implementing the decay math client-side:

      - ``points_current`` — decayed point value at the time of the request
      - ``decay_factor``   — current decay multiplier (1.0 = fresh, 0.5 = 1wk old)
    """
    src = _load()
    if not src or not isinstance(src, dict):
        return src
    now_ts = time.time()
    out = {k: v for k, v in src.items() if k != "scans"}
    out["scans"] = {}
    for scan_id, scan_blob in (src.get("scans") or {}).items():
        if not isinstance(scan_blob, dict):
            out["scans"][scan_id] = scan_blob
            continue
        new_scan = {k: v for k, v in scan_blob.items() if k != "attributes"}
        attrs_out = {}
        for key, entry in (scan_blob.get("attributes") or {}).items():
            if not isinstance(entry, dict):
                attrs_out[key] = entry
                continue
            decayed = _decayed_points(entry, now_ts)
            stored = float(entry.get("points", 0) or 0)
            new_entry = dict(entry)
            new_entry["points_current"] = round(decayed, 4)
            new_entry["decay_factor"] = (
                round(decayed / stored, 4) if stored > 0 else 1.0
            )
            attrs_out[key] = new_entry
        new_scan["attributes"] = attrs_out
        out["scans"][scan_id] = new_scan
    return out


def write_weights(new_data: dict) -> dict:
    """Persist a new weights dict (called from PUT /api/scan-weights).
    Preserves the on-disk structure; bumps ``_meta.updated_at``.
    Returns the freshly-written dict."""
    global _cache, _mtime
    if not isinstance(new_data, dict) or "scans" not in new_data:
        raise ValueError("weights payload must contain a 'scans' key")
    # Drop derived fields if a previous all_weights() output is being
    # round-tripped back through write_weights.
    for scan_blob in (new_data.get("scans") or {}).values():
        if not isinstance(scan_blob, dict):
            continue
        for entry in (scan_blob.get("attributes") or {}).values():
            if isinstance(entry, dict):
                entry.pop("points_current", None)
                entry.pop("decay_factor", None)
    # ensure _meta exists & stamp updated_at
    meta = new_data.setdefault("_meta", {})
    meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    tmp = WEIGHTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new_data, f, indent=2)
    os.replace(tmp, WEIGHTS_FILE)
    with _lock:
        _cache = new_data
        try:
            _mtime = os.stat(WEIGHTS_FILE).st_mtime
        except FileNotFoundError:
            _mtime = 0.0
    return new_data


def award_attribute_bonus(scan_id: str, key: str, bonus: float,
                           reason: str | None = None) -> dict | None:
    """Apply lazy decay to the current stored points, add ``bonus``, then
    persist. Bumps ``points_updated_ts`` to ``now`` so decay starts fresh
    from this moment.

    Used by the performer-ranker scan: when a stock enters the top
    5/20/100/250 by 30-day return, every attribute that stock currently
    exhibits is awarded a tier bonus (+27/+9/+3/+1).

    Returns the freshly-written full weights dict, or ``None`` if the
    ``scan_id``/``key`` doesn't exist in the catalog (so unknown bonuses
    don't silently create dead attributes — the owning scan must seed
    the key first).
    """
    if not scan_id or not key or bonus == 0:
        return None
    with _lock:
        data = _load()
        if not isinstance(data, dict):
            return None
        scans = data.get("scans") or {}
        scan_blob = scans.get(scan_id)
        if not isinstance(scan_blob, dict):
            return None
        attrs = scan_blob.get("attributes")
        if not isinstance(attrs, dict) or key not in attrs:
            return None
        entry = attrs[key]
        if not isinstance(entry, dict):
            return None
        now_ts = time.time()
        cur = _decayed_points(entry, now_ts)
        new_pts = cur + float(bonus)
        entry["points"] = round(new_pts, 4)
        entry["points_updated_ts"] = now_ts
        entry["lifetime_bonus"] = round(
            float(entry.get("lifetime_bonus", 0) or 0) + float(bonus), 4
        )
        entry["last_bonus_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry["last_bonus_value"] = float(bonus)
        if reason:
            entry["last_bonus_reason"] = reason
    # release lock before write_weights re-acquires
    return write_weights(data)


def stamp_updated_ts(scan_id: str, key: str) -> None:
    """Set ``points_updated_ts`` to now without changing ``points``. Used
    when an operator manually edits an attribute's points via the
    Attributes tab — the typed value is treated as the freshly-decayed
    current value and starts decaying from now."""
    with _lock:
        data = _load()
        if not isinstance(data, dict):
            return
        try:
            entry = data["scans"][scan_id]["attributes"][key]
        except (KeyError, TypeError):
            return
        if not isinstance(entry, dict):
            return
        entry["points_updated_ts"] = time.time()
    write_weights(data)
