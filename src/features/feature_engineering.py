import pandas as pd
import numpy as np
import os

PROCESSED_PATH = "data/processed/"
FEATURE_PATH = "data/processed/features.csv"


# ======================================================
# 1. TECHNICAL INDICATORS
# ======================================================
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Core price-based technical indicators with enhanced microstructure."""

    features = {}

    # === 1) Basic log return ===
    if "close" in df.columns:
        features["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # === 2) Price structure ratios ===
    if {"open", "high", "low", "close"}.issubset(df.columns):
        features["high_low_ratio"] = df["high"] / df["low"]
        features["close_open_ratio"] = df["close"] / df["open"]
        features["high_close_ratio"] = df["high"] / df["close"]
        features["low_close_ratio"] = df["low"] / df["close"]
        features["price_range_pct"] = (df["high"] - df["low"]) / df["open"]

        # --- Microstructure (performance boost) ---
        features["body_ratio"] = abs(
            df["close"] - df["open"]) / (df["high"] - df["low"])
        features["upper_shadow"] = (
            df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"])
        features["lower_shadow"] = (df[["close", "open"]].min(
            axis=1) - df["low"]) / (df["high"] - df["low"])

    # === 3) Moving averages & ratios ===
    if "close" in df.columns:
        for window in [7, 20, 50, 200]:
            ma = df["close"].rolling(window).mean()
            features[f"ma_{window}"] = ma
            features[f"ma_{window}_ratio"] = df["close"] / ma

        # Crossovers (need to create df temporarily to access ma columns)
        df_temp = pd.concat(
            [df, pd.DataFrame(features, index=df.index)], axis=1)
        features["ma_7_20_cross"] = (
            df_temp["ma_7"] > df_temp["ma_20"]).astype(int)
        features["ma_20_50_cross"] = (
            df_temp["ma_20"] > df_temp["ma_50"]).astype(int)

    # === 4) Volatility & risk signals ===
    log_return_col = features.get(
        "log_return") if "log_return" in features else df.get("log_return")
    if log_return_col is not None:
        for window in [5, 10, 20, 30]:
            vol = log_return_col.rolling(window).std()
            features[f"volatility_{window}"] = vol

            # Z-score normalization
            vol_mean = vol.rolling(100).mean()
            vol_std = vol.rolling(100).std().replace(0, np.nan)
            features[f"volatility_{window}_z"] = (vol - vol_mean) / vol_std

        features["vol_20"] = features["volatility_20"]

        # --- Volatility of volatility (big performance boost) ---
        features["vol_of_vol_20"] = features["volatility_20"].rolling(20).std()

        # Parkinson volatility
        if {"high", "low"}.issubset(df.columns):
            features["parkinson_vol_20"] = np.sqrt(
                (np.log(df["high"] / df["low"]) **
                 2).rolling(20).mean() / (4 * np.log(2))
            )

    # === 5) RSI multiple windows ===
    if "close" in df.columns:
        delta = df["close"].diff()

        def _rsi(series: pd.Series, window: int) -> pd.Series:
            gain = (series.where(series > 0, 0)).rolling(window).mean()
            loss = (-series.where(series < 0, 0)).rolling(window).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        features["rsi_14"] = _rsi(delta, 14)
        features["rsi_21"] = _rsi(delta, 21)
        features["rsi_30"] = _rsi(delta, 30)

        features["rsi"] = features["rsi_14"]  # legacy

        # RSI momentum (new)
        features["rsi_14_change"] = features["rsi_14"].diff()

    # === 6) MACD ===
    if "close" in df.columns:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        features["macd"] = ema12 - ema26
        features["macd_signal"] = features["macd"].ewm(
            span=9, adjust=False).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

    # === 7) Bollinger Bands ===
    if "close" in df.columns:
        for window in [20, 50]:
            ma = df["close"].rolling(window).mean()
            std = df["close"].rolling(window).std()
            features[f"bb_upper_{window}"] = ma + 2 * std
            features[f"bb_lower_{window}"] = ma - 2 * std
            features[f"bb_width_{window}"] = (
                features[f"bb_upper_{window}"] - features[f"bb_lower_{window}"]) / ma
            features[f"bb_position_{window}"] = (df["close"] - features[f"bb_lower_{window}"]) / (
                features[f"bb_upper_{window}"] - features[f"bb_lower_{window}"]
            )

    # === 8) Volatility regimes ===
    if "vol_20" in features:
        long_vol_mean = features["vol_20"].rolling(90).mean()
        features["vol_regime"] = (
            features["vol_20"] > long_vol_mean).astype(int)

    # Concatenate all features at once to avoid fragmentation
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


# ======================================================
# 2. STATISTICAL / MOMENTUM FEATURES
# ======================================================
def add_statistical_and_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    features = {}

    # Skew / kurt
    if "log_return" in df.columns:
        for window in [5, 10, 20]:
            features[f"return_skew_{window}"] = df["log_return"].rolling(
                window).skew()
            features[f"return_kurt_{window}"] = df["log_return"].rolling(
                window).kurt()

        # Acceleration
        features["price_acceleration"] = df["log_return"].diff()

    # Momentum
    if "close" in df.columns:
        for window in [5, 10, 20]:
            features[f"momentum_{window}"] = df["close"].pct_change(window)

        features["momentum_acceleration"] = features["momentum_20"].diff()

        # ROC (better than pct change)
        features["roc_7"] = df["close"].pct_change(7)
        features["roc_14"] = df["close"].pct_change(14)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


# ======================================================
# 3. VOLUME FEATURES
# ======================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    features = {}

    if "volume" in df.columns:
        features["volume_ma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_ma_20"]
        features["volume_change"] = df["volume"].pct_change()

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


# ======================================================
# 4. CFGI FEATURES
# ======================================================
def add_cfgi_features(df: pd.DataFrame) -> pd.DataFrame:
    if "cfgi" not in df.columns:
        return df

    features = {}

    # Z-scores + MAs
    for window in [14, 30, 60]:
        mean = df["cfgi"].rolling(window).mean()
        std = df["cfgi"].rolling(window).std().replace(0, np.nan)
        features[f"cfgi_z_{window}"] = (df["cfgi"] - mean) / std

    features["cfgi_ma_7"] = df["cfgi"].rolling(7).mean()
    features["cfgi_ma_30"] = df["cfgi"].rolling(30).mean()

    # --- KEY INTERACTIONS (big boost in v2 results) ---
    if "volatility_20" in df.columns:
        features["cfgi_volatility_interaction"] = features["cfgi_z_30"] * \
            df["volatility_20"]

    if "rsi_14" in df.columns:
        features["cfgi_rsi_interaction"] = df["cfgi"] * df["rsi_14"] / 100.0

    if "momentum_20" in df.columns:
        features["cfgi_momentum_interaction"] = features["cfgi_z_30"] * \
            df["momentum_20"]

    if "volume_ratio" in df.columns:
        features["cfgi_volume_interaction"] = features["cfgi_z_30"] * \
            df["volume_ratio"]

    # Sentiment alignment
    if "momentum_20" in df.columns:
        price_up = (df["momentum_20"] > 0).astype(int)
        sentiment_up = (df["cfgi"] > 50).astype(int)
        features["sentiment_price_alignment"] = (
            price_up == sentiment_up).astype(int)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


# ======================================================
# 5. TARGETS (forward looking)
# ======================================================
def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    features = {}

    if "close" in df.columns:
        features["target_log_return_3d"] = np.log(
            df["close"].shift(-3) / df["close"])
        features["target_direction"] = (
            features["target_log_return_3d"] > 0).astype(int)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


# ======================================================
# 6. CLEAN
# ======================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)

    print(f"Rows before cleaning: {before}, after cleaning: {after}")

    # Remove helper columns safely
    helper_cols = ["high_low", "high_close", "low_close", "true_range"]
    df = df.drop(
        columns=[c for c in helper_cols if c in df.columns], errors="ignore")

    return df


# ======================================================
# 7. MAIN PIPELINE
# ======================================================
def save_features(df: pd.DataFrame, path: str = FEATURE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved engineered features → {path}")


if __name__ == "__main__":

    base_file = os.path.join(PROCESSED_PATH, "dataset_with_sentiment.csv")
    if not os.path.exists(base_file):
        base_file = os.path.join(PROCESSED_PATH, "dataset.csv")

    df = pd.read_csv(base_file)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    df = add_technical_features(df)
    df = add_statistical_and_momentum_features(df)
    df = add_volume_features(df)
    df = add_cfgi_features(df)
    df = create_targets(df)
    df = clean_data(df)

    save_features(df)
    print("🎯 Feature engineering complete.")
