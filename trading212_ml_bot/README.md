# Trading 212 Demo ML Trader

A deliberately **demo-only** Python trading bot. A machine-learning model estimates the
probability that the next daily close will be higher; deterministic rules control
position sizing and execution.

## Architecture

Historical market data -> feature engineering -> Random Forest -> probability signal
-> risk checks -> Trading 212 demo order.

The ML model does **not** bypass the risk layer.

## Safety defaults

- Trading 212 base URL is hard-coded to `demo.trading212.com`.
- `DRY_RUN=true` by default.
- No short selling.
- Maximum position fraction and maximum order value are enforced.
- API credentials live in `.env`, not source control.

## Setup

1. Create and activate a virtual environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`.
4. Add Trading 212 **demo** API credentials.
5. Confirm the Trading 212 execution tickers in `config.py`.
6. Train: `python train_model.py`
7. Test with `DRY_RUN=true`: `python main.py`
8. Only after inspecting account/position responses and proposed orders, set
   `DRY_RUN=false` to submit orders to the demo account.

## Model

The initial Random Forest uses:
- 1-day return
- 5-day return
- 5/20 moving-average ratio
- 20/50 moving-average ratio
- 20-day volatility
- RSI(14)
- volume change

The target is whether the following daily close is higher.

The train/test split is chronological rather than random to reduce look-ahead leakage.

## Important limitations

This is an educational/demo trading system, not evidence of a profitable strategy.
Yahoo Finance data and Trading 212 execution symbols/prices can differ. A serious
evaluation should add transaction costs, slippage, walk-forward validation,
benchmark comparison, and a proper backtester before changing the strategy.
