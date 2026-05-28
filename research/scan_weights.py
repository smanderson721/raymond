#!/usr/bin/env python3
"""
scan_weights.py — central loader for per-scan attribute reward values.

Each scan module that has been refactored to use this loader calls
``weight(scan_id, attribute_key, default)``. The first call (per process)
reads ``data/scan_weights.json``; subsequent calls re-read only if
the file's mtime has changed, so an edit via the
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
            "<key>": { "points": 5.0, "description": "..." },
            ...
          },
          "multipliers": { "<key>": { "value": 1.1, ... } },
          "thresholds":  { "<key>": { "value": 3.0, ... } }
        }
      }
    }

If the file is missing, malformed, or the requested key is absent, the
caller's ``default`` is returned — so removing or breaking the JSON file
never crashes a scan; it just falls back to compile-time defaults.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = os.path.join(REPO_ROOT, "data", "scan_weights.json")

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


def weight(scan_id: str, key: str, default: float) -> float:
    """Return points for ``scan_id.attributes[key]``, or ``default`` if missing."""
    data = _load()
    try:
        return float(data["scans"][scan_id]["attributes"][key]["points"])
    except (KeyError, TypeError, ValueError):
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
    """Return the full weights dict (for the GET /api/scan-weights endpoint)."""
    return _load()


def write_weights(new_data: dict) -> dict:
    """Persist a new weights dict (called from PUT /api/scan-weights).
    Preserves the on-disk structure; bumps ``_meta.updated_at``.
    Returns the freshly-written dict."""
    global _cache, _mtime
    from datetime import datetime, timezone
    if not isinstance(new_data, dict) or "scans" not in new_data:
        raise ValueError("weights payload must contain a 'scans' key")
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
