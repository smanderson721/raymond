"""Run AlpacaTradingAgent over the top-N tickers from raymond's watchlist.

Designed to run under the GitHub Actions `trading-agent.yml` workflow after
the Daily Catalyst Scan publishes a fresh `watchlist_combined.json`.

Inputs:
  - research_output/watchlist_combined.json   (raymond's BUP watchlist)
  - <ATA_DIR>/tradingagents/                  (AlpacaTradingAgent codebase,
                                                cloned by the workflow)

Outputs:
  - research_output/ata_decisions/<DATE>.json (one entry per ticker w/ signal)
  - <ATA_DIR>/eval_results/<TICKER>/...       (full per-analyst reports)

Env:
  GEMINI_API_KEY, GOOGLE_API_KEY, FINNHUB_API_KEY,
  ALPACA_API_KEY (or ALPACA_API_KEY_ID), ALPACA_SECRET_KEY (or
  ALPACA_API_SECRET_KEY), FRED_API_KEY, ATA_QUICK_LLM, ATA_DEEP_LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def build_config() -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG  # type: ignore
    cfg = dict(DEFAULT_CONFIG)
    cfg["llm_provider"] = "google"
    cfg["quick_think_llm"] = os.environ.get("ATA_QUICK_LLM", "gemini-3.1-flash-lite")
    cfg["deep_think_llm"] = os.environ.get("ATA_DEEP_LLM", "gemini-3.1-flash-lite")
    cfg["google_thinking_level"] = "high"
    cfg["allow_shorts"] = os.environ.get("ATA_ALLOW_SHORTS") == "1"
    cfg["max_debate_rounds"] = int(os.environ.get("ATA_DEBATE_ROUNDS", "2"))
    cfg["max_risk_discuss_rounds"] = int(os.environ.get("ATA_RISK_ROUNDS", "2"))
    return cfg


def load_top_tickers(watchlist_path: Path, top: int) -> list[str]:
    """Pull top-N tickers from raymond's watchlist_combined.json, ranked by
    best_bup desc (the BUP score already filters for catalyst quality)."""
    data = json.loads(watchlist_path.read_text())
    tickers = data.get("tickers") or data.get("candidates") or data.get("stocks") or []
    if isinstance(tickers, dict):
        tickers = list(tickers.values())
    ranked = sorted(
        tickers,
        key=lambda t: (t.get("best_bup") if isinstance(t, dict) else 0) or 0,
        reverse=True,
    )
    out: list[str] = []
    for entry in ranked:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            sym = entry.get("ticker") or entry.get("symbol")
            if sym:
                out.append(sym)
        if len(out) >= top:
            break
    return out


def _normalize_alpaca_env() -> None:
    """ATA expects ALPACA_API_KEY / ALPACA_SECRET_KEY. raymond stores them
    as ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY. Mirror the names so the
    same workflow secret block works for both repos."""
    if not os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_API_KEY_ID"):
        os.environ["ALPACA_API_KEY"] = os.environ["ALPACA_API_KEY_ID"]
    if not os.environ.get("ALPACA_SECRET_KEY") and os.environ.get("ALPACA_API_SECRET_KEY"):
        os.environ["ALPACA_SECRET_KEY"] = os.environ["ALPACA_API_SECRET_KEY"]
    # ATA reads GOOGLE_API_KEY; we only set GEMINI_API_KEY in raymond.
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    os.environ.setdefault("ALPACA_USE_PAPER", "True")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", type=Path,
                    default=REPO_ROOT / "research_output" / "watchlist_combined.json")
    ap.add_argument("--ata-dir", type=Path, required=True,
                    help="Path to a checkout of huygiatrng/AlpacaTradingAgent")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "research_output" / "ata_decisions")
    ap.add_argument("--trade-date", default=date.today().isoformat())
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    _normalize_alpaca_env()

    if not args.ata_dir.is_dir():
        print(f"AlpacaTradingAgent missing at {args.ata_dir}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(args.ata_dir))

    from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore

    symbols = load_top_tickers(args.watchlist, args.top)
    if not symbols:
        print(f"No tickers in {args.watchlist}", file=sys.stderr)
        return 2

    cfg = build_config()
    # Route ATA eval_results into the cloned ATA directory.
    cfg["results_dir"] = str(args.ata_dir / "eval_results")

    print(f"ATA run · {args.trade_date} · {len(symbols)} tickers: {symbols}")
    print(f"  provider={cfg['llm_provider']} quick={cfg['quick_think_llm']} "
          f"deep={cfg['deep_think_llm']} debate={cfg['max_debate_rounds']}")

    ta = TradingAgentsGraph(debug=args.debug, config=cfg)

    results: list[dict] = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        try:
            _state, signal = ta.propagate(symbol, args.trade_date)
            results.append({
                "ticker": symbol, "trade_date": args.trade_date,
                "signal": signal, "ok": True,
            })
            print(f"  → {signal!r}")
        except Exception as exc:
            results.append({
                "ticker": symbol, "trade_date": args.trade_date,
                "error": f"{type(exc).__name__}: {exc}", "ok": False,
            })
            print(f"  ! {type(exc).__name__}: {exc}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.trade_date}.json"
    out_path.write_text(json.dumps(
        {"trade_date": args.trade_date, "results": results}, indent=2))
    # Also write/overwrite "latest.json" for the Cloud tab.
    (args.out_dir / "latest.json").write_text(json.dumps(
        {"trade_date": args.trade_date, "results": results}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
