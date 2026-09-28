from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import BUY_PROBABILITY, HISTORICAL_NEWS_CSV, MODEL_PATH, TARGET_DAYS, WATCHLIST
from features import FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS, build_features
from market_data import download_history
from news import NEWS_FEATURE_COLUMNS, aggregate_news, load_historical_news

STARTING_CASH = 10_000.0
COST_RATE = 0.001  # simple slippage/fees assumption for research only


def make_frame(symbol, news):
    data = build_features(download_history(symbol, period="5y"))
    symbol_news = news[news["symbol"] == symbol]
    for idx in data.index:
        as_of = pd.Timestamp(idx)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        for col, value in aggregate_news(symbol_news, as_of).items():
            data.loc[idx, col] = value
    return data.dropna(subset=FEATURE_COLUMNS)


def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Train V2 first: python train_model.py")
    if not Path(HISTORICAL_NEWS_CSV).exists():
        raise FileNotFoundError("Historical news CSV required for backtest")

    model = joblib.load(MODEL_PATH)
    news = load_historical_news(HISTORICAL_NEWS_CSV)

    for _, symbol in WATCHLIST:
        data = make_frame(symbol, news)
        p = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
        data = data.assign(probability=p)
        data["forward_return"] = data["Close"].shift(-TARGET_DAYS) / data["Close"] - 1
        trades = data[data["probability"] >= BUY_PROBABILITY].dropna(subset=["forward_return"])
        strategy_returns = trades["forward_return"] - COST_RATE
        compounded = (1 + strategy_returns).prod() - 1 if len(trades) else 0.0
        buy_hold = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
        hit_rate = float((strategy_returns > 0).mean()) if len(trades) else 0.0
        print(f"\n{symbol}")
        print(f"Signals: {len(trades)}")
        print(f"Signal hit rate: {hit_rate:.1%}")
        print(f"Naive compounded signal return: {compounded:.1%}")
        print(f"Buy & hold over same dataset: {buy_hold:.1%}")
        print("Note: overlapping 3-day signals make this a diagnostic, not a portfolio-grade backtest.")


if __name__ == "__main__":
    main()
