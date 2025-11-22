import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.train_model import compute_features_and_labels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StrictTrainer")

def train_strict_model(data_dir="data/swing", model_path="src/oracle/model_strict_2024.json", split_date_ts=1735689600):
    """
    Train a model strictly on data BEFORE the split date (Jan 1, 2025).
    """
    logger.info(f"Training STRICT model from {data_dir}")
    logger.info(f"Training Data Cutoff: {split_date_ts} (Jan 1, 2025)")

    all_files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]

    train_dfs = []
    test_dfs = []

    for filename in all_files:
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)

        try:
            df = pd.read_csv(path)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]

            # Compute features
            df = compute_features_and_labels(df)

            # Split by Date
            train_subset = df[df['timestamp'] < split_date_ts].copy()
            test_subset = df[df['timestamp'] >= split_date_ts].copy()

            if not train_subset.empty:
                train_dfs.append(train_subset)

            if not test_subset.empty:
                test_dfs.append(test_subset)

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    if not train_dfs:
        logger.error("No training data found before cutoff!")
        return

    # Combine
    full_train_df = pd.concat(train_dfs, ignore_index=True)
    full_test_df = pd.concat(test_dfs, ignore_index=True)

    logger.info(f"Training Set: {len(full_train_df)} samples")
    logger.info(f"Test Set: {len(full_test_df)} samples")

    # Features
    features = [
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
    ]

    X_train = full_train_df[features]
    y_train = full_train_df['target']

    # Remap labels: -1->0, 0->1, 1->2
    y_train_map = y_train + 1

    logger.info("Training XGBoost model...")

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train, y_train_map)

    # Evaluate on Test Set (Out of Sample)
    if not full_test_df.empty:
        X_test = full_test_df[features]
        y_test = full_test_df['target']
        y_test_map = y_test + 1

        y_pred_map = model.predict(X_test)
        y_pred = y_pred_map - 1

        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"OOS Test Accuracy (2025): {acc:.4f}")
        logger.info(classification_report(y_test, y_pred, target_names=['SELL', 'CHOP', 'BUY']))

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Strict Model saved to {model_path}")

if __name__ == "__main__":
    train_strict_model()
