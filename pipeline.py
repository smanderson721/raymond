#!/usr/bin/env python3
"""
Raymond — minimal stocks scanner pipeline.

Subcommands:
    --stocks-scan       Weekly precondition scan (yfinance, all tickers)
    --market-scan       Daily catalyst scan (Finnhub + EDGAR + Gemini)
    --refresh-prices    Daily 30-day price refresh (top N by P + 2C)
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stocks-scan", action="store_true",
                   help="Precondition scan via yfinance (no Gemini)")
    p.add_argument("--market-scan", action="store_true",
                   help="News + EDGAR + Gemini catalyst scoring")
    p.add_argument("--refresh-prices", action="store_true",
                   help="Daily 30-day price refresh for top N stocks by P + 2C")

    p.add_argument("--exchange", type=str, default="all",
                   choices=["all", "nasdaq", "nyse"])
    p.add_argument("--news-days", type=int, default=2,
                   help="EDGAR + Finnhub lookback in days (default: 2)")
    p.add_argument("--finnhub-min-precondition", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--max-cap", type=float, default=10e9)
    p.add_argument("--top-n", type=int, default=250,
                   help="Universe size for --refresh-prices (default: 250)")
    p.add_argument("--fetch-kindling", action="store_true",
                   help="Run precondition scan before --market-scan")
    args = p.parse_args()

    if args.stocks_scan:
        from research.stocks_scanner import run_scan
        run_scan(exchange=args.exchange)
        return

    if args.refresh_prices:
        from research.price_refresher import refresh_prices
        refresh_prices(top_n=args.top_n)
        return

    if args.market_scan:
        from research.stocks_scanner import get_tickers
        from research.news_monitor import (fetch_finnhub_news,
                                           fetch_edgar_filings,
                                           prefilter_finnhub_tickers)
        from research.digest_builder import (build_digest, load_scan_stocks,
                                             save_news_cache)
        from research.catalyst_scorer import score_all_catalysts

        if args.fetch_kindling:
            from research.stocks_scanner import run_scan
            print("\n── fetch_kindling: precondition scan first ──\n")
            run_scan(exchange=args.exchange)

        finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        if not finnhub_key:
            print("Error: FINNHUB_API_KEY not set.")
            sys.exit(1)

        tickers = get_tickers(args.exchange)
        scan_stocks = load_scan_stocks()
        days = args.news_days
        print(f"\n── News catalyst scan: {len(tickers)} tickers ({days}d) ──\n")

        edgar_data = fetch_edgar_filings(tickers, days=days, forms=["8-K", "6-K"])
        finnhub_tickers = prefilter_finnhub_tickers(
            tickers, scan_stocks, edgar_data=edgar_data,
            max_cap=args.max_cap,
            min_precondition_total=args.finnhub_min_precondition,
        )
        finnhub_data = fetch_finnhub_news(finnhub_tickers, finnhub_key, days=days)
        save_news_cache(finnhub_data, edgar_data)

        build_digest(finnhub_data, edgar_data, scan_stocks,
                     batch_size=args.batch_size,
                     max_cap=args.max_cap,
                     full_tickers=tickers)
        score_all_catalysts(finnhub_data, edgar_data, scan_stocks,
                            max_cap=args.max_cap)
        return

    p.print_help()


if __name__ == "__main__":
    main()
