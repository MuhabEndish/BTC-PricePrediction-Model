import requests
import pandas as pd
import time
import os

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="1d", limit=1000):
    url = BINANCE_URL
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume",
        "num_trades","taker_buy_volume",
        "taker_buy_quote_volume","ignore"
    ]

    df = pd.DataFrame(data, columns=cols)

    # Convert timestamps to UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Convert numeric columns
    numeric_cols = ["open","high","low","close","volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Keep only essential columns
    df = df[["open_time","open","high","low","close","volume"]]

    return df


def save_data(df, path="data/raw/ohlcv.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved OHLCV to {path}")


if __name__ == "__main__":
    df = fetch_binance_ohlcv()
    save_data(df)
