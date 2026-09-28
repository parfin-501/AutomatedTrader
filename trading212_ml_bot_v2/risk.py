"""Deterministic position sizing.

The ML model proposes direction; this module controls capital exposure independently.
"""


def buy_quantity(
    available_cash: float,
    portfolio_value: float,
    current_position_value: float,
    price: float,
    max_position_fraction: float,
    max_order_value: float,
    min_order_value: float,
) -> float:
    """Return a buy quantity without breaching cash, position, or order limits."""
    if price <= 0 or portfolio_value <= 0:
        return 0.0

    # Cap total exposure to a single instrument as a fraction of the portfolio.
    max_position_value = portfolio_value * max_position_fraction
    remaining_position_capacity = max(0.0, max_position_value - current_position_value)

    # The tightest constraint wins. This prevents model confidence from controlling size.
    order_value = min(available_cash, remaining_position_capacity, max_order_value)

    # Avoid submitting tiny orders that add noise without meaningful exposure.
    if order_value < min_order_value:
        return 0.0

    return round(order_value / price, 6)


def sell_quantity(current_quantity: float) -> float:
    """Exit an existing long position; V2 intentionally does not short-sell."""
    return round(max(0.0, current_quantity), 6)
