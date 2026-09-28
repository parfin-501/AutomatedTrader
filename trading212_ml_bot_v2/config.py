"""Central configuration for the demo trader.

Keep strategy/risk constants here so experiments do not require changing execution code.
"""

# Each tuple maps the Trading 212 execution ticker to the Yahoo Finance symbol
# used for historical/current market data. Confirm execution tickers in your demo account.
WATCHLIST = [
    ("AAPL_US_EQ", "AAPL"),
    ("MSFT_US_EQ", "MSFT"),
]

# The bot reevaluates the watchlist every 15 minutes. The model itself is trained
# on daily features, so increasing this frequency does not create new information.
CHECK_INTERVAL_SECONDS = 900

# Probability bands deliberately leave a HOLD region to avoid trading on weak signals.
BUY_PROBABILITY = 0.62
SELL_PROBABILITY = 0.38

# Training target: classify whether the price gains >1% over the next 3 trading days.
TARGET_RETURN = 0.01
TARGET_DAYS = 3

# Deterministic limits sit outside the ML model: even a high-confidence prediction
# cannot exceed these exposure/order constraints.
MAX_POSITION_FRACTION = 0.10
MAX_ORDER_VALUE = 500.0
MIN_ORDER_VALUE = 10.0

MODEL_PATH = "models/model_v2.joblib"
HISTORICAL_NEWS_CSV = "data/historical_news.csv"
