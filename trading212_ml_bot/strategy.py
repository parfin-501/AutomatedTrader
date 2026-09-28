import joblib

from features import FEATURE_COLUMNS, build_features


class MLStrategy:
    def __init__(self, model_path: str, buy_threshold: float, sell_threshold: float):
        self.model = joblib.load(model_path)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def evaluate(self, history):
        featured = build_features(history).dropna(subset=FEATURE_COLUMNS)

        if featured.empty:
            raise RuntimeError("Not enough price history to calculate features")

        latest = featured.iloc[[-1]]
        probability_up = float(
            self.model.predict_proba(latest[FEATURE_COLUMNS])[0, 1]
        )

        if probability_up >= self.buy_threshold:
            signal = "BUY"
        elif probability_up <= self.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        return signal, probability_up, float(latest["Close"].iloc[0])
