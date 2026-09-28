import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "ma_ratio_5_20",
    "ma_ratio_20_50",
    "volatility_20",
    "rsi_14",
    "volume_change",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    data["ma_ratio_5_20"] = ma5 / ma20 - 1
    data["ma_ratio_20_50"] = ma20 / ma50 - 1
    data["volatility_20"] = data["return_1d"].rolling(20).std()
    data["rsi_14"] = _rsi(close)
    data["volume_change"] = volume.pct_change()

    return data.replace([np.inf, -np.inf], np.nan)
