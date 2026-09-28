# AutomatedTrader V2 — Trading 212 Demo ML + News Trader

A demo-only automated trading research project combining technical market features, machine learning, financial-news sentiment and company-event detection.

## V2 architecture

Market prices/volume -> technical features
News -> sentiment + event extraction
Both -> Random Forest probability -> deterministic risk controls -> Trading 212 DEMO execution

The ML layer proposes a probability. It never bypasses the risk/execution layer.

## News + company events

Current inference retrieves ticker news and extracts:
- 24-hour and 7-day sentiment
- news volume and sentiment acceleration
- product launches/releases
- software/product updates
- earnings/guidance
- partnerships
- acquisitions/mergers
- regulatory events
- recalls/delays
- management changes

V2 uses VADER as a lightweight local sentiment baseline. Event extraction is transparent keyword/rule-based logic so results are inspectable.

## Leakage-safe historical training

Historical news must be timestamped. Put your dataset at `data/historical_news.csv` with:

`symbol,published_at,text`

An example is included. For every historical price observation, only articles published on or before that timestamp are aggregated. This avoids giving the model future information.

## Prediction target

V2 predicts whether the price will rise by more than 1% over the next 3 trading days (configurable), rather than merely whether tomorrow is fractionally higher.

## Setup

1. `python -m venv .venv`
2. Activate the environment.
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add Trading 212 DEMO credentials.
5. Add a real timestamped historical-news dataset to `data/historical_news.csv`.
6. Train with `python train_model.py`.
7. Run `python backtest.py`.
8. Run the bot with `DRY_RUN=true`: `python main.py`.

## Safety

- Trading 212 URL remains hard-coded to the demo environment.
- Dry-run is on by default.
- No short selling in V2.
- Position size and order-value limits remain deterministic.
- `.env`, trained models and private historical-news data are gitignored.

## Backtesting note

`backtest.py` is deliberately labelled a diagnostic backtest. It compares signal behaviour with buy-and-hold and includes a simple transaction-cost assumption, but overlapping multi-day signals mean it is not yet a portfolio-grade simulator. A future version should add walk-forward retraining, non-overlapping/portfolio-aware execution, slippage models and benchmark statistics.

## Disclaimer

Educational/demo research software. Model probabilities and backtests are not evidence of future profitability.
