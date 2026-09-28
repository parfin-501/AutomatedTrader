"""Minimal Trading 212 wrapper intentionally restricted to the demo environment."""

import requests


class Trading212Demo:
    # Safety invariant: there is no configuration switch to the live API. Changing to
    # real-money execution requires an explicit source-code change.
    BASE_URL = "https://demo.trading212.com/api/v0"

    def __init__(self, api_key: str, api_secret: str, dry_run: bool = True):
        if not api_key or not api_secret:
            raise ValueError("Trading 212 demo API credentials are required")

        self.auth = (api_key, api_secret)
        self.dry_run = dry_run
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        """Send an authenticated request and fail loudly on HTTP errors."""
        response = self.session.request(
            method, f"{self.BASE_URL}{path}", auth=self.auth, timeout=15, **kwargs
        )
        response.raise_for_status()
        return response.json()

    def account_summary(self):
        return self._request("GET", "/equity/account/summary")

    def positions(self):
        return self._request("GET", "/equity/positions")

    def market_order(self, ticker: str, quantity: float):
        """Submit a demo market order; negative quantity represents a sell."""
        if quantity == 0:
            return None

        payload = {"ticker": ticker, "quantity": quantity}

        # Dry-run is an additional safety layer on top of the demo-only base URL.
        if self.dry_run:
            print(f"[DRY RUN] market order: {payload}")
            return {"dryRun": True, **payload}

        return self._request("POST", "/equity/orders/market", json=payload)
