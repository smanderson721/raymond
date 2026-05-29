#!/usr/bin/env python3
"""
scan_runner — single CLI entry for all live scans.

Usage:
    python scan_runner.py macro_regime
    python scan_runner.py tech_slice
    python scan_runner.py news_poll
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback


SCANS = {
    # Phase A
    "market_pulse": "research.scans.market_pulse",
    "tech_slice": "research.scans.tech_slice",
    "reg_sho": "research.scans.reg_sho",
    "insider_cluster": "research.scans.insider_cluster",
    # Phase B
    "options_unusual": "research.scans.options_unusual",
    "material_8k": "research.scans.material_8k",
    "fundamentals_snap": "research.scans.fundamentals_snap",
    # add new scans here as they're built
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scan", choices=sorted(SCANS.keys()))
    args = p.parse_args()

    mod_path = SCANS[args.scan]
    try:
        mod = importlib.import_module(mod_path)
    except Exception as e:
        print(f"FATAL: could not import {mod_path}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    if not hasattr(mod, "run"):
        print(f"FATAL: {mod_path} has no run()", file=sys.stderr)
        return 2

    try:
        result = mod.run()
        print(f"[{args.scan}] {result}")
        return 0
    except Exception as e:
        # Engine already wrote the error into status.json via Session __exit__
        # but ensure we surface it on the runner stdout for GHA logs.
        print(f"[{args.scan}] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
