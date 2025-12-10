"""
Backtesting: Trading Simulation
Simulate trading strategy based on model predictions and calculate PnL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TEST_PRED_PATH = "data/processed/test_predictions.csv"
BACKTEST_RESULTS = "data/processed/backtest_results.csv"


def load_predictions():
    df = pd.read_csv(TEST_PRED_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Check if this is classification output (has prob_up column)
    if 'prob_up' in df.columns:
        # For classification: convert binary predictions to expected returns
        # Use probability * expected magnitude
        df['pred'] = (df['prob_up'] - 0.5) * 2 * df['true'].abs().mean()

    return df


def simple_strategy_backtest(df, transaction_cost=0.001):
    """
    Simple long/short strategy:
    - If prediction > 0: Long (buy)
    - If prediction < 0: Short (sell)

    Args:
        df: DataFrame with 'true' and 'pred' columns (log returns)
        transaction_cost: Transaction cost per trade (0.1% = 0.001)

    Returns:
        Dictionary with backtest results
    """

    # Trading signals
    df["signal"] = np.where(df["pred"] > 0, 1, -1)  # 1 = long, -1 = short

    # Strategy returns (before costs)
    df["strategy_return"] = df["signal"] * df["true"]

    # Count trades (position changes)
    df["position_change"] = df["signal"].diff().fillna(0)
    df["is_trade"] = (df["position_change"] != 0).astype(int)
    num_trades = df["is_trade"].sum()

    # Apply transaction costs
    df["transaction_cost"] = df["is_trade"] * transaction_cost
    df["strategy_return_net"] = df["strategy_return"] - df["transaction_cost"]

    # Buy & Hold returns (always long)
    df["buy_hold_return"] = df["true"]

    # Cumulative returns
    df["cumulative_strategy"] = (1 + df["strategy_return_net"]).cumprod()
    df["cumulative_buy_hold"] = (1 + df["buy_hold_return"]).cumprod()

    # Calculate metrics
    total_return_strategy = (df["cumulative_strategy"].iloc[-1] - 1) * 100
    total_return_buy_hold = (df["cumulative_buy_hold"].iloc[-1] - 1) * 100

    # Calculate Sharpe ratios (handle zero std)
    std_strategy = df["strategy_return_net"].std()
    std_buy_hold = df["buy_hold_return"].std()

    sharpe_strategy = (df["strategy_return_net"].mean() /
                       std_strategy * np.sqrt(252)) if std_strategy > 0 else 0
    sharpe_buy_hold = (df["buy_hold_return"].mean() /
                       std_buy_hold * np.sqrt(252)) if std_buy_hold > 0 else 0

    max_drawdown_strategy = ((df["cumulative_strategy"].cummax() - df["cumulative_strategy"]) /
                             df["cumulative_strategy"].cummax()).max() * 100
    max_drawdown_buy_hold = ((df["cumulative_buy_hold"].cummax() - df["cumulative_buy_hold"]) /
                             df["cumulative_buy_hold"].cummax()).max() * 100

    win_rate = (df["strategy_return_net"] > 0).sum() / len(df) * 100

    results = {
        "Strategy Total Return (%)": total_return_strategy,
        "Buy & Hold Total Return (%)": total_return_buy_hold,
        "Outperformance (%)": total_return_strategy - total_return_buy_hold,
        "Strategy Sharpe Ratio": sharpe_strategy,
        "Buy & Hold Sharpe Ratio": sharpe_buy_hold,
        "Strategy Max Drawdown (%)": max_drawdown_strategy,
        "Buy & Hold Max Drawdown (%)": max_drawdown_buy_hold,
        "Win Rate (%)": win_rate,
        "Number of Trades": num_trades,
        "Transaction Cost per Trade (%)": transaction_cost * 100,
    }

    return results, df


def threshold_strategy_backtest(df, threshold=0.005, transaction_cost=0.001):
    """
    Threshold strategy: Only trade when confidence is high
    - If prediction > threshold: Long
    - If prediction < -threshold: Short
    - Otherwise: Hold (neutral)
    """

    # Trading signals with threshold
    df["signal"] = np.where(df["pred"] > threshold, 1,
                            np.where(df["pred"] < -threshold, -1, 0))

    # Forward fill positions (hold when signal = 0)
    df["position"] = df["signal"].replace(
        0, np.nan).ffill().fillna(0)

    # Strategy returns
    df["strategy_return"] = df["position"] * df["true"]

    # Count trades
    df["position_change"] = df["position"].diff().fillna(0)
    df["is_trade"] = (df["position_change"] != 0).astype(int)
    num_trades = df["is_trade"].sum()

    # Apply transaction costs
    df["transaction_cost"] = df["is_trade"] * transaction_cost
    df["strategy_return_net"] = df["strategy_return"] - df["transaction_cost"]

    # Buy & Hold
    df["buy_hold_return"] = df["true"]

    # Cumulative returns
    df["cumulative_strategy"] = (1 + df["strategy_return_net"]).cumprod()
    df["cumulative_buy_hold"] = (1 + df["buy_hold_return"]).cumprod()

    # Metrics
    total_return_strategy = (df["cumulative_strategy"].iloc[-1] - 1) * 100
    total_return_buy_hold = (df["cumulative_buy_hold"].iloc[-1] - 1) * 100

    # Calculate Sharpe ratios (handle zero std)
    std_strategy = df["strategy_return_net"].std()
    std_buy_hold = df["buy_hold_return"].std()

    sharpe_strategy = (df["strategy_return_net"].mean() /
                       std_strategy * np.sqrt(252)) if std_strategy > 0 else 0
    sharpe_buy_hold = (df["buy_hold_return"].mean() /
                       std_buy_hold * np.sqrt(252)) if std_buy_hold > 0 else 0

    max_drawdown_strategy = ((df["cumulative_strategy"].cummax() - df["cumulative_strategy"]) /
                             df["cumulative_strategy"].cummax()).max() * 100
    max_drawdown_buy_hold = ((df["cumulative_buy_hold"].cummax() - df["cumulative_buy_hold"]) /
                             df["cumulative_buy_hold"].cummax()).max() * 100

    win_rate = (df["strategy_return_net"] > 0).sum() / len(df) * 100

    results = {
        "Strategy Total Return (%)": total_return_strategy,
        "Buy & Hold Total Return (%)": total_return_buy_hold,
        "Outperformance (%)": total_return_strategy - total_return_buy_hold,
        "Strategy Sharpe Ratio": sharpe_strategy,
        "Buy & Hold Sharpe Ratio": sharpe_buy_hold,
        "Strategy Max Drawdown (%)": max_drawdown_strategy,
        "Buy & Hold Max Drawdown (%)": max_drawdown_buy_hold,
        "Win Rate (%)": win_rate,
        "Number of Trades": num_trades,
        "Threshold": threshold,
        "Transaction Cost per Trade (%)": transaction_cost * 100,
    }

    return results, df


def run_backtest():
    """Run backtesting analysis"""

    print("="*60)
    print("BACKTESTING: Trading Simulation")
    print("="*60)

    # Load predictions
    df = load_predictions()
    print(
        f"\n📅 Test period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"📊 Number of samples: {len(df)}")
    print(f"📈 Actual return: {(df['true'].sum() * 100):.2f}%")

    # Strategy 1: Simple long/short
    print(f"\n{'='*60}")
    print("STRATEGY 1: Simple Long/Short")
    print(f"{'='*60}")
    results_simple, df_simple = simple_strategy_backtest(df.copy())

    for key, value in results_simple.items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:10.2f}")
        else:
            print(f"{key:40s}: {value}")

    # Strategy 2: Threshold-based
    print(f"\n{'='*60}")
    print("STRATEGY 2: Threshold-Based (threshold=0.005)")
    print(f"{'='*60}")
    results_threshold, df_threshold = threshold_strategy_backtest(
        df.copy(), threshold=0.005)

    for key, value in results_threshold.items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:10.2f}")
        else:
            print(f"{key:40s}: {value}")

    # Save results
    results_summary = pd.DataFrame([
        {"Strategy": "Simple Long/Short", **results_simple},
        {"Strategy": "Threshold-Based", **results_threshold},
    ])
    results_summary.to_csv(BACKTEST_RESULTS, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {BACKTEST_RESULTS}")
    print(f"{'='*60}")

    return results_summary, df_simple, df_threshold


if __name__ == "__main__":
    results_summary, df_simple, df_threshold = run_backtest()
