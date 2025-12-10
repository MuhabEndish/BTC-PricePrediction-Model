import os
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


# ============================================================
# CONFIG
# ============================================================

FEATURE_PATH = "data/processed/features.csv"
MODEL_PATH = "models/crypto_autogluon_v2"


# ============================================================
# LOAD & SPLIT
# ============================================================

def load_data(path=FEATURE_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_time_based(df, test_ratio=0.2):
    """Last test_ratio of data for test (default 20%)."""
    test_size = int(len(df) * test_ratio)
    if test_size < 10:
        raise ValueError("Dataset too short for time-based split")

    train_data = df.iloc[:-test_size].copy()
    test_data = df.iloc[-test_size:].copy()

    return train_data, test_data


def validation_split_time_based(train_data, valid_ratio=0.2):
    """Use last 20% of training data as validation for robust holdout."""
    n = len(train_data)
    split_idx = int(n * (1 - valid_ratio))

    train = train_data.iloc[:split_idx].copy()
    valid = train_data.iloc[split_idx:].copy()

    return train, valid


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(train_data, label, top_k=50):
    """Select top K features using correlation + remove highly correlated pairs."""
    features = [col for col in train_data.columns if col != label]

    # Only use numeric features for correlation
    numeric_features = train_data[features].select_dtypes(
        include=[np.number]).columns.tolist()

    if len(numeric_features) == 0:
        print("\n⚠️  Warning: No numeric features found. Using all features.")
        return features[:top_k]

    # Step 1: Get correlation with target
    correlations = train_data[numeric_features].corrwith(
        train_data[label]).abs()
    correlations = correlations.sort_values(ascending=False)

    # Step 2: Remove highly correlated feature pairs (>0.95)
    selected = []
    feature_corr_matrix = train_data[correlations.index].corr().abs()

    for feat in correlations.index:
        # Check if highly correlated with any already selected feature
        is_redundant = False
        for sel_feat in selected:
            if feature_corr_matrix.loc[feat, sel_feat] > 0.95:
                is_redundant = True
                break

        if not is_redundant:
            selected.append(feat)

        if len(selected) >= top_k:
            break

    print(f"\n📊 Feature Selection:")
    print(f"   Total features: {len(features)}")
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Selected (after removing redundant): {len(selected)}")
    print(f"   Top 10 by correlation:")
    for feat in selected[:10]:
        print(f"   {feat}: {correlations[feat]:.4f}")

    return selected
# ============================================================
# TRAINING
# ============================================================


def train_autogluon(train_data,
                    valid_data,
                    label="target_direction",
                    save_path=MODEL_PATH,
                    top_k_features=70,
                    use_classification=True):

    # ---- Columns we MUST NOT use as predictors ----
    leak_cols = [
        "date",
        "cfgi_label",               # string column, not numeric
        "target_direction" if not use_classification else "target_log_return_3d",
        "target_log_return_3d" if use_classification else "target_direction",
        "target_log_return_7d",
        "target_log_return",        # legacy name (safe to drop if exists)
        "target_smoothed",
        "target_log_return_1d",
        "target_log_return_5d",
        "target_volatility_adjusted",
    ]

    # ---- Drop leakage columns ----
    for col in leak_cols:
        if col in train_data.columns and col != label:
            train_data = train_data.drop(columns=[col])
        if col in valid_data.columns and col != label:
            valid_data = valid_data.drop(columns=[col])

    # ---- Feature Selection ----
    selected_features = select_features(
        train_data, label, top_k=top_k_features)
    train_data = train_data[[label] + selected_features]
    valid_data = valid_data[[label] + selected_features]

    os.makedirs(save_path, exist_ok=True)

    problem_type = "binary" if use_classification else "regression"
    eval_metric = "accuracy" if use_classification else "root_mean_squared_error"

    print("\n🚀 Training Configuration:")
    print(f"   Label: {label}")
    print(f"   Problem Type: {problem_type}")
    print(f"   Features: {len(train_data.columns) - 1}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(valid_data)}")
    print(f"   Time limit: 1800s (30.0 min)")
    print(f"   Preset: best_quality")
    print(f"   Bag folds: 10\n")

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        path=save_path,
        eval_metric=eval_metric,
    )

    # ---- Optimized hyperparameters for financial data (stronger regularization) ----
    hyperparameters = {
        "GBM": [
            {
                "num_boost_round": 500,
                "learning_rate": 0.005,
                "num_leaves": 20,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "min_data_in_leaf": 30,
                "lambda_l1": 1.0,
                "lambda_l2": 1.0,
            },
            {
                "num_boost_round": 300,
                "learning_rate": 0.01,
                "num_leaves": 15,
                "feature_fraction": 0.6,
                "bagging_fraction": 0.6,
                "min_data_in_leaf": 40,
                "lambda_l1": 2.0,
                "lambda_l2": 2.0,
            },
        ],
        "XGB": [
            {
                "n_estimators": 400,
                "learning_rate": 0.005,
                "max_depth": 4,
                "colsample_bytree": 0.7,
                "subsample": 0.7,
                "min_child_weight": 5,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.01,
                "max_depth": 3,
                "colsample_bytree": 0.6,
                "subsample": 0.6,
                "min_child_weight": 7,
                "reg_alpha": 2.0,
                "reg_lambda": 3.0,
            },
        ],
        "CAT": [
            {
                "iterations": 400,
                "learning_rate": 0.01,
                "depth": 4,
                "l2_leaf_reg": 5,
                "bagging_temperature": 1,
            }
        ],
        "RF": [
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": 0.5,
            }
        ],
        "NN_TORCH": [
            {
                "num_epochs": 50,
                "learning_rate": 0.0005,
                "activation": "relu",
                "dropout_prob": 0.3,
                "hidden_size": 64,
                "weight_decay": 0.01,
            }
        ],
    }

    # Combine train and validation data for bagging mode
    combined_data = pd.concat([train_data, valid_data], ignore_index=True)

    predictor.fit(
        train_data=combined_data,
        presets="best_quality",
        time_limit=1800,
        num_bag_folds=10,
        auto_stack=True,
        hyperparameters=hyperparameters,
    )

    print(f"\n✅ Model saved to: {save_path}")

    # Show leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    print(f"\n📊 Top 10 Models:")
    print(leaderboard[["model", "score_val", "pred_time_val", "fit_time"]].head(
        10).to_string(index=False))

    return predictor


# ============================================================
# TEST PREDICTIONS
# ============================================================

def generate_test_predictions(predictor, test_data, label="target_direction", use_classification=True):
    """Generate predictions on test set."""
    print(
        f"\n🔮 Generating predictions on test set ({len(test_data)} samples)...")

    dates = test_data["date"].copy() if "date" in test_data.columns else None
    y_true = test_data[label].copy()

    # For classification, also get regression target for evaluation
    y_true_reg = test_data["target_log_return_3d"].copy(
    ) if use_classification and "target_log_return_3d" in test_data.columns else None

    leak_cols = [
        "date",
        "target_direction",
        "target_log_return",
        "target_log_return_1d",
        "target_log_return_3d",
        "target_log_return_5d",
        "target_log_return_7d",
        "target_smoothed",
        "target_volatility_adjusted",
    ]

    for col in leak_cols:
        if col in test_data.columns and col != label:
            test_data = test_data.drop(columns=[col])

    X = test_data.drop(
        columns=[label]) if label in test_data.columns else test_data
    preds = predictor.predict(X)

    # Save predictions
    if use_classification:
        # Get probability predictions for classification
        pred_proba = predictor.predict_proba(X)
        prob_up = pred_proba[1] if 1 in pred_proba.columns else pred_proba.iloc[:, 1]
        results = pd.DataFrame({
            "true": y_true_reg if y_true_reg is not None else y_true,
            "pred": preds,
            "prob_up": prob_up,
        })
    else:
        results = pd.DataFrame({
            "true": y_true,
            "pred": preds,
        })

    if dates is not None:
        results.insert(0, "date", dates.values)

    save_path = "data/processed/test_predictions.csv"
    results.to_csv(save_path, index=False)
    print(f"   ✅ Saved predictions to: {save_path}")

    return results


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(predictor, df):
    fi = predictor.feature_importance(df)
    out_path = "data/processed/feature_importance.csv"
    fi.to_csv(out_path)
    print(f"\n✔ Saved feature importance → {out_path}")
    return fi


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("📂 Loading data...")
    df = load_data()

    # ---- TIME BASED SPLITS ----
    print("✂️ Splitting data...")
    train_data, test_data = train_test_split_time_based(df, test_ratio=0.2)
    print(f"   Train: {len(train_data)} samples")
    print(f"   Test: {len(test_data)} samples")

    # Use 20% of train for validation
    train, valid = validation_split_time_based(train_data, valid_ratio=0.2)

    # ---- TRAIN ----
    print("\n🏋️ Training model...")
    predictor = train_autogluon(
        train, valid, label="target_log_return_3d", top_k_features=70, use_classification=False)

    # ---- PREDICT ----
    results = generate_test_predictions(
        predictor, test_data, label="target_log_return_3d", use_classification=False)

    # ---- FEATURE IMPORTANCE ----
    save_feature_importance(predictor, train)

    print("\n✅ Training complete!")
    print("   Model saved to: models/crypto_autogluon_v2")
    print("   Predictions saved to: data/processed/test_predictions.csv")
    print("   Feature importance saved to: data/processed/feature_importance.csv")
    print("\n📊 For evaluation metrics, run: python src/evaluation/evaluate.py")
