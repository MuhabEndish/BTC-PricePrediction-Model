import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def load_ohlcv(path=RAW_PATH + "ohlcv.csv"):
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.rename(columns={"open_time": "date"})
    df["date"] = df["date"].dt.floor("D")  # round to day
    return df


def load_cfgi(path=RAW_PATH + "cfgi.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].dt.floor("D")
    return df


def merge_data(ohlcv, cfgi):
    # Merge daily data
    df = pd.merge(ohlcv, cfgi, on="date", how="left")

    # Forward-fill CFGI because some days may be missing
    df["cfgi"] = df["cfgi"].ffill()
    df["cfgi_label"] = df["cfgi_label"].ffill()

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_target_variable(df):
    """
    Target = next-day log return
    log(Close[t+1] / Close[t])

    IMPORTANT:
    We drop 'close_shifted' so it is NOT used as a feature.
    """
    df["close_shifted"] = df["close"].shift(-1)

    df["target_log_return"] = np.log(df["close_shifted"] / df["close"])

    # Remove last row (future price unavailable)
    df = df.dropna(subset=["target_log_return"]).reset_index(drop=True)

    # Drop the helper column to avoid data leakage
    df = df.drop(columns=["close_shifted"])

    return df



def save_dataset(df, path=PROCESSED_PATH + "dataset.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved processed dataset to {path}")


if __name__ == "__main__":
    ohlcv = load_ohlcv()
    cfgi = load_cfgi()

    merged = merge_data(ohlcv, cfgi)
    merged = add_target_variable(merged)

    save_dataset(merged)
    print("Data merging and processing complete.")