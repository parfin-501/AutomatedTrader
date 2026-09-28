import logging
import os
import time

from dotenv import load_dotenv

from config import (
    BUY_PROBABILITY,
    CHECK_INTERVAL_SECONDS,
    MAX_ORDER_VALUE,
    MAX_POSITION_FRACTION,
    MIN_ORDER_VALUE,
    MODEL_PATH,
    SELL_PROBABILITY,
    WATCHLIST,
)
from market_data import latest_history
from risk import buy_quantity, sell_quantity
from strategy import MLNewsStrategy
from trading212 import Trading212Demo


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def position_lookup(positions):
    return {p.get("ticker"): p for p in positions}


def run_once(trader, strategy):
    account = trader.account_summary()
    positions = trader.positions()
    by_ticker = position_lookup(positions)

    # Field names can evolve; inspect your demo response before disabling DRY_RUN.
    cash = as_float(account.get("cash", {}).get("availableToTrade"))
    portfolio = as_float(account.get("totalValue"))

    logging.info("Available cash: %.2f | Portfolio value: %.2f", cash, portfolio)

    for execution_ticker, data_symbol in WATCHLIST:
        try:
            history = latest_history(data_symbol)
            signal, probability_up, price, news_features = strategy.evaluate(data_symbol, history)

            position = by_ticker.get(execution_ticker, {})
            current_quantity = as_float(position.get("quantity"))
            current_position_value = current_quantity * price

            logging.info(
                "%s | p(target)=%.3f | %s | approx price=%.2f | held=%.6f | news24h=%.2f | product7d=%.0f | updates7d=%.0f",
                execution_ticker,
                probability_up,
                signal,
                price,
                current_quantity,
                news_features["news_sentiment_24h"],
                news_features["event_product_launch_7d"],
                news_features["event_software_update_7d"],
            )

            if signal == "BUY":
                quantity = buy_quantity(
                    available_cash=cash,
                    portfolio_value=portfolio,
                    current_position_value=current_position_value,
                    price=price,
                    max_position_fraction=MAX_POSITION_FRACTION,
                    max_order_value=MAX_ORDER_VALUE,
                    min_order_value=MIN_ORDER_VALUE,
                )

                if quantity > 0:
                    trader.market_order(execution_ticker, quantity)

            elif signal == "SELL" and current_quantity > 0:
                quantity = sell_quantity(current_quantity)
                trader.market_order(execution_ticker, -quantity)

        except Exception:
            logging.exception("Failed while processing %s", execution_ticker)


def main():
    load_dotenv()

    dry_run = os.getenv("DRY_RUN", "true").lower() != "false"

    trader = Trading212Demo(
        api_key=os.getenv("TRADING212_API_KEY", ""),
        api_secret=os.getenv("TRADING212_API_SECRET", ""),
        dry_run=dry_run,
    )

    strategy = MLNewsStrategy(
        MODEL_PATH,
        buy_threshold=BUY_PROBABILITY,
        sell_threshold=SELL_PROBABILITY,
    )

    logging.info("Starting Trading 212 DEMO ML bot | dry_run=%s", dry_run)

    while True:
        run_once(trader, strategy)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
