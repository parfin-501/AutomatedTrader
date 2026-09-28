"""Feature engineering shared by training, inference, and backtesting."""

import numpy as np
import pandas as pd

from news import NEWS_FEATURE_COLUMNS

TECHNICAL_FEATURE_COLUMNS = [
    "return_1d", "return_5d", "ma_ratio_5_20", "ma_ratio_20_50",
    "volatility_20", "rsi_14", "volume_change",
]

# Keeping one canonical feature order is important: scikit-learn expects inference
# columns to match the order/schema used during training.
FEATURE_COLUMNS = TECHNICAL_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate a simple 14-period Relative Strength Index."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add technical features without removing rows needed by later alignment."""
    data = prices.copy()
    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    # Returns/momentum capture recent direction; MA ratios capture trend regime.
    data["return_1d"] = close.pct_change()
    data["return_5d"] = close.pct_change(5)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    data["ma_ratio_5_20"] = ma5 / ma20 - 1
    data["ma_ratio_20_50"] = ma20 / ma50 - 1

    # Volatility, RSI and volume change provide risk/momentum/activity context.
    data["volatility_20"] = data["return_1d"].rolling(20).std()
    data["rsi_14"] = _rsi(close)
    data["volume_change"] = volume.pct_change()

    # Infinite ratios cannot be consumed by the model; convert them to missing values
    # and let callers drop only the rows required for their particular task.
    return data.replace([np.inf, -np.inf], np.nan)
