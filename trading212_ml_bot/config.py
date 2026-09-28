WATCHLIST = [
    # Trading 212 instrument tickers may differ from Yahoo symbols.
    # Map execution ticker -> market-data symbol.
    ("AAPL_US_EQ", "AAPL"),
    ("MSFT_US_EQ", "MSFT"),
]

CHECK_INTERVAL_SECONDS = 900

# ML decision thresholds
BUY_PROBABILITY = 0.62
SELL_PROBABILITY = 0.38

# Risk controls
MAX_POSITION_FRACTION = 0.10
MAX_ORDER_VALUE = 500.0
MIN_ORDER_VALUE = 10.0

MODEL_PATH = "models/model.joblib"
