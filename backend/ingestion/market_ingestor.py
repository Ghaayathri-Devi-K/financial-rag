import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_market_data(ticker, start_date="2023-01-01", end_date=None, interval="1d", save_dir="data/market"):
    """
    Downloads historical OHLCV data for a given ticker from Yahoo Finance
    and saves it as a CSV file.
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (default = today)
        interval (str): Data granularity ('1d', '1h', '15m', etc.)
        save_dir (str): Folder to save CSV
    """
    os.makedirs(save_dir, exist_ok=True)
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"💹 Fetching {ticker} data from {start_date} to {end_date} ({interval})...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    if df.empty:
        print(f"⚠️ No data returned for {ticker}")
        return None

    # Add moving averages for quick analytics
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()

    path = os.path.join(save_dir, f"{ticker}_market.csv")
    df.to_csv(path)
    print(f"✅ Saved {len(df)} rows → {path}")
    return df


def fetch_multiple_tickers(tickers, start_date="2023-01-01"):
    """Fetch data for a list of tickers."""
    results = {}
    for t in tickers:
        df = fetch_market_data(t, start_date)
        if df is not None:
            results[t] = df
    return results


if __name__ == "__main__":
    tickers = ["AAPL", "NVDA"]
    fetch_multiple_tickers(tickers, start_date="2023-01-01")
