import joblib
import pandas as pd

from features import FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS, build_features
from news import NEWS_FEATURE_COLUMNS, aggregate_news, fetch_current_news


class MLNewsStrategy:
    def __init__(self, model_path: str, buy_threshold: float, sell_threshold: float):
        self.model = joblib.load(model_path)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def evaluate(self, symbol, history):
        technical = build_features(history).dropna(subset=TECHNICAL_FEATURE_COLUMNS)
        if technical.empty:
            raise RuntimeError("Not enough price history to calculate features")

        latest = technical.iloc[[-1]].copy()
        news = fetch_current_news(symbol)
        news_features = aggregate_news(news)
        for column in NEWS_FEATURE_COLUMNS:
            latest[column] = news_features[column]

        probability_up = float(self.model.predict_proba(latest[FEATURE_COLUMNS])[0, 1])
        if probability_up >= self.buy_threshold:
            signal = "BUY"
        elif probability_up <= self.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        return signal, probability_up, float(latest["Close"].iloc[0]), news_features
