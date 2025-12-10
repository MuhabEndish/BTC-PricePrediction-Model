"""
Visualization: Plot predicted vs actual prices and model performance
"""
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

TEST_PRED_PATH = "data/processed/test_predictions.csv"
VIZ_OUTPUT_DIR = "data/processed/visualizations/"


def load_predictions():
    """Load predictions and handle both regression and classification formats"""
    df = pd.read_csv(TEST_PRED_PATH)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Check if this is classification output (has prob_up column)
    if 'prob_up' in df.columns:
        print("📊 Detected classification predictions with probabilities")
        # Convert probabilities to expected returns for visualization
        # Center around 0: prob_up=0.5 → 0, prob_up=1.0 → +0.05, prob_up=0 → -0.05
        df['pred'] = (df['prob_up'] - 0.5) * 0.1

    return df


def plot_predictions_vs_actual(df, save_path=None):
    """Plot predicted vs actual log returns"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual and predicted returns
    ax.plot(df["date"], df["true"], label="Actual Returns",
            alpha=0.7, linewidth=1.5)
    ax.plot(df["date"], df["pred"], label="Predicted Returns",
            alpha=0.7, linewidth=1.5)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Log Return", fontsize=12)
    ax.set_title("Predicted vs Actual Log Returns",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")

    plt.close()


def plot_cumulative_prices(df, save_path=None):
    """Plot cumulative price movement from log returns"""

    # Convert log returns to cumulative price index
    # Use exp for log returns: exp(cumsum(log_return)) = cumulative price
    df["actual_price_index"] = np.exp(df["true"].cumsum())
    df["predicted_price_index"] = np.exp(df["pred"].cumsum())

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot cumulative returns
    ax.plot(df["date"], df["actual_price_index"], label="Actual Price",
            alpha=0.8, linewidth=2, color="blue")
    ax.plot(df["date"], df["predicted_price_index"], label="Predicted Price",
            alpha=0.8, linewidth=2, color="orange")

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price Index (Base = 1.0)", fontsize=12)
    ax.set_title("Cumulative Price Movement: Actual vs Predicted",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")

    plt.close()


def plot_prediction_scatter(df, save_path=None):
    """Scatter plot of predictions vs actuals"""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(df["true"], df["pred"], alpha=0.5, s=30)

    # Add perfect prediction line
    min_val = min(df["true"].min(), df["pred"].min())
    max_val = max(df["true"].max(), df["pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--',
            linewidth=2, label="Perfect Prediction")

    # Formatting
    ax.set_xlabel("Actual Log Return", fontsize=12)
    ax.set_ylabel("Predicted Log Return", fontsize=12)
    ax.set_title("Prediction Accuracy: Scatter Plot",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add correlation
    corr = df["true"].corr(df["pred"])
    ax.text(0.05, 0.95, f"Correlation: {corr:.4f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round',
                                               facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")

    plt.close()


def plot_direction_accuracy(df, save_path=None):
    """Plot direction prediction accuracy over time"""

    # Calculate rolling direction accuracy
    window = 20
    df["true_direction"] = (df["true"] > 0).astype(int)
    df["pred_direction"] = (df["pred"] > 0).astype(int)
    df["correct_direction"] = (
        df["true_direction"] == df["pred_direction"]).astype(int)
    df["rolling_accuracy"] = df["correct_direction"].rolling(
        window).mean() * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot rolling accuracy
    ax.plot(df["date"], df["rolling_accuracy"], linewidth=2, color="green")
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2,
               label="Random Baseline (50%)", alpha=0.7)

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(
        f"Direction Accuracy (%) - {window}-day Rolling", fontsize=12)
    ax.set_title("Model Direction Accuracy Over Time",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")

    plt.close()


def plot_residuals(df, save_path=None):
    """Plot prediction residuals"""

    df["residual"] = df["true"] - df["pred"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Residuals over time
    ax1.scatter(df["date"], df["residual"], alpha=0.5, s=20)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax1.set_title("Residuals Over Time", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Residuals histogram
    ax2.hist(df["residual"], bins=50, alpha=0.7,
             color="blue", edgecolor="black")
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Residual", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Saved: {save_path}")

    plt.close()


def create_all_visualizations():
    """Generate all visualization plots"""

    print("="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)

    # Create output directory
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

    # Load predictions
    df = load_predictions()
    print(f"📂 Loaded {len(df)} predictions")

    # Generate plots
    print("\n1️⃣ Predictions vs Actual...")
    plot_predictions_vs_actual(
        df, save_path=f"{VIZ_OUTPUT_DIR}predictions_vs_actual.png")

    print("2️⃣ Cumulative Prices...")
    plot_cumulative_prices(
        df, save_path=f"{VIZ_OUTPUT_DIR}cumulative_prices.png")

    print("3️⃣ Prediction Scatter...")
    plot_prediction_scatter(
        df, save_path=f"{VIZ_OUTPUT_DIR}prediction_scatter.png")

    print("4️⃣ Direction Accuracy Over Time...")
    plot_direction_accuracy(
        df, save_path=f"{VIZ_OUTPUT_DIR}direction_accuracy.png")

    print("5️⃣ Residuals Analysis...")
    plot_residuals(df, save_path=f"{VIZ_OUTPUT_DIR}residuals.png")

    print(f"\n{'='*60}")
    print(f"✅ All visualizations saved to: {VIZ_OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_all_visualizations()
