import pandas as pd
import yfinance as yf


def download_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(f"No market data returned for {symbol}")

    # yfinance can return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def latest_history(symbol: str) -> pd.DataFrame:
    return download_history(symbol, period="1y")
