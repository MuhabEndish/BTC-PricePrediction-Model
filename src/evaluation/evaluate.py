import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Default paths
TEST_PRED_PATH = "data/processed/test_predictions.csv"
EVAL_PATH = "data/processed/eval_results.json"


def compute_metrics(df):
    """Compute comprehensive evaluation metrics."""
    y_true = df["true"].values
    y_pred = df["pred"].values

    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Direction accuracy
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    direction_accuracy = (direction_true == direction_pred).mean()

    # Additional metrics
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    # Mean absolute percentage error (handle division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    # Directional metrics by market condition
    up_mask = y_true > 0
    down_mask = y_true < 0

    up_accuracy = (direction_true[up_mask] == direction_pred[up_mask]).mean(
    ) if up_mask.any() else np.nan
    down_accuracy = (direction_true[down_mask] == direction_pred[down_mask]).mean(
    ) if down_mask.any() else np.nan

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mse": float(mse),
        "correlation": float(correlation),
        "direction_accuracy": float(direction_accuracy),
        "direction_accuracy_up": float(up_accuracy) if not np.isnan(up_accuracy) else None,
        "direction_accuracy_down": float(down_accuracy) if not np.isnan(down_accuracy) else None,
        "mape": float(mape) if not np.isnan(mape) else None,
        "samples_total": int(len(y_true)),
        "samples_up": int(up_mask.sum()),
        "samples_down": int(down_mask.sum()),
    }

    return metrics


def print_metrics(metrics):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    print("\n🎯 Regression Metrics:")
    print(f"   RMSE:              {metrics['rmse']:.6f}")
    print(f"   MAE:               {metrics['mae']:.6f}")
    print(f"   R²:                {metrics['r2']:.6f}")
    print(f"   MSE:               {metrics['mse']:.6f}")
    if metrics['mape'] is not None:
        print(f"   MAPE:              {metrics['mape']:.2f}%")

    print("\n📈 Directional Metrics:")
    print(
        f"   Overall Accuracy:  {metrics['direction_accuracy']:.4f} ({metrics['direction_accuracy']*100:.2f}%)")
    if metrics['direction_accuracy_up'] is not None:
        print(
            f"   Up Market:         {metrics['direction_accuracy_up']:.4f} ({metrics['direction_accuracy_up']*100:.2f}%)")
    if metrics['direction_accuracy_down'] is not None:
        print(
            f"   Down Market:       {metrics['direction_accuracy_down']:.4f} ({metrics['direction_accuracy_down']*100:.2f}%)")
    print(f"   Correlation:       {metrics['correlation']:.6f}")

    print("\n📋 Sample Distribution:")
    print(f"   Total Samples:     {metrics['samples_total']}")
    print(
        f"   Up Markets:        {metrics['samples_up']} ({metrics['samples_up']/metrics['samples_total']*100:.1f}%)")
    print(
        f"   Down Markets:      {metrics['samples_down']} ({metrics['samples_down']/metrics['samples_total']*100:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Alow custom path via command line argument
    pred_path = sys.argv[1] if len(sys.argv) > 1 else TEST_PRED_PATH

    # Determine output path
    eval_path = "data/processed/eval_results.json"

    print(f"\n📂 Loading predictions from: {pred_path}")

    if not os.path.exists(pred_path):
        print(f"❌ Error: File not found: {pred_path}")
        print(f"   Please run training first to generate predictions.")
        sys.exit(1)

    df = pd.read_csv(pred_path)

    # Handle classification predictions (pred is 0/1, convert to direction for evaluation)
    if 'prob_up' in df.columns:
        print("   Detected classification predictions with probabilities")
        # Convert binary predictions (0/1) to signed values for direction accuracy
        df['pred_direction'] = (df['pred'] > 0.5).astype(int)
        # For metrics, we still use the original pred column

    # Validate required columns
    if "true" not in df.columns or "pred" not in df.columns:
        print(f"❌ Error: CSV must contain 'true' and 'pred' columns")
        print(f"   Found columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"   Loaded {len(df)} predictions")

    # Compute metrics
    metrics = compute_metrics(df)

    # Save to JSON
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Saved results to: {eval_path}")

    # Print results
    print_metrics(metrics)
