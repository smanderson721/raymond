# Raymond — Stock Catalyst Scanner

Automated daily/weekly stock scanning pipeline. Built to run on GitHub Actions.

## Pipelines

| Workflow | Schedule | Cost | Output |
|---|---|---|---|
| `weekly-precondition-scan.yml` | Sun 23:00 UTC | $0 (yfinance only) | `scan_results.json` |
| `daily-catalyst-scan.yml` | Tue–Sat 02:00 UTC (Mon–Fri evening ET) | ~$0.05/day Gemini | `catalyst_scores.json`, `price_data.json`, `news_cache.json` |

## BUP score formula

```
BUP = (precondition_score + 2 × catalyst_score) − √(max(round(pct_30d × 100), 0))
```

Catalysts weighted 2× because news events matter more than technicals.
The penalty term subtracts the square root of the integer 30-day rise %, with
no penalty if the stock has fallen.

## Required GitHub Actions secrets

- `GEMINI_API_KEY`
- `FINNHUB_API_KEY`

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=..." > .env
echo "FINNHUB_API_KEY=..." >> .env

python pipeline.py --stocks-scan
python pipeline.py --market-scan
python pipeline.py --refresh-prices
```
