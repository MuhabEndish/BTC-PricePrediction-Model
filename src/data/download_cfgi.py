import requests
import pandas as pd
import os

CFGI_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_cfgi():
    response = requests.get(CFGI_URL)
    response.raise_for_status()

    data = response.json()["data"]

    df = pd.DataFrame(data)
    df["value"] = df["value"].astype(int)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(
        df["timestamp"], errors="coerce"), unit="s", utc=True)

    df = df.rename(columns={
        "value": "cfgi",
        "value_classification": "cfgi_label",
        "timestamp": "date"
    })

    df = df[["date", "cfgi", "cfgi_label"]]

    return df


def save_data(df, path="data/raw/cfgi.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved CFGI to {path}")


if __name__ == "__main__":
    df = fetch_cfgi()
    save_data(df)
