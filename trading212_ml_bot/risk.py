def buy_quantity(
    available_cash: float,
    portfolio_value: float,
    current_position_value: float,
    price: float,
    max_position_fraction: float,
    max_order_value: float,
    min_order_value: float,
) -> float:
    if price <= 0 or portfolio_value <= 0:
        return 0.0

    max_position_value = portfolio_value * max_position_fraction
    remaining_position_capacity = max(0.0, max_position_value - current_position_value)

    order_value = min(
        available_cash,
        remaining_position_capacity,
        max_order_value,
    )

    if order_value < min_order_value:
        return 0.0

    return round(order_value / price, 6)


def sell_quantity(current_quantity: float) -> float:
    # First version exits the position rather than short-selling.
    return round(max(0.0, current_quantity), 6)
