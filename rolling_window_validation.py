import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RollingWalkForward")

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)

    atr = tr.rolling(window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/window).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx

def compute_time_sin(timestamps):
    seconds_in_day = 24 * 60 * 60
    return np.sin(2 * np.pi * (timestamps % seconds_in_day) / seconds_in_day)

def compute_volume_delta(volume):
    v = volume.replace(0, 1)
    return np.log(v / v.shift(1))

def prepare_features(df):
    """Compute all features for the dataframe"""
    window = 20
    close_col = 'close'

    df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window).std()
    df['ma'] = df[close_col].rolling(window=window).mean()
    df['std'] = df[close_col].rolling(window=window).std()
    df['z_score'] = (df[close_col] - df['ma']) / df['std']
    df['rsi'] = compute_rsi(df[close_col], window=14)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    df['time_sin'] = compute_time_sin(df['timestamp'])
    df['volume_delta'] = compute_volume_delta(df['volume'])

    # Labeling
    horizon = 5
    threshold = 0.0001
    df['future_close'] = df[close_col].shift(-horizon)
    df['target'] = (df['future_close'] > df[close_col] + threshold).astype(int)

    df = df.dropna()
    return df

def rolling_window_validation(data_path="data/raw/eurusd_m1.csv", train_window_weeks=2):
    """
    Rolling Window Walk-Forward Validation
    - Fixed training window size (e.g., 2 weeks)
    - Slides forward to test on subsequent periods
    - Example: Train on Weeks 1-2, test on Week 3
    -          Train on Weeks 2-3, test on Week 4
    """
    logger.info("=" * 80)
    logger.info(f"ROLLING WINDOW WALK-FORWARD VALIDATION (Train Window: {train_window_weeks} weeks)")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")

    # Prepare features
    logger.info("\nPreparing features...")
    df = prepare_features(df)

    # Define rolling windows
    start_date = df['datetime'].min()
    total_days = (df['datetime'].max() - start_date).days

    # Calculate window size in days
    window_days = total_days // 4  # Each week is ~7 days
    train_window_days = train_window_weeks * window_days
    test_window_days = window_days

    logger.info(f"\nTrain window: ~{train_window_days} days")
    logger.info(f"Test window: ~{test_window_days} days")

    # Create rolling splits
    splits = []
    num_folds = 4 - train_window_weeks  # With 4 weeks total and 2-week train, we get 2 folds

    for i in range(num_folds):
        train_start = start_date + timedelta(days=window_days * i)
        train_end = train_start + timedelta(days=train_window_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_window_days)

        train_mask = (df['datetime'] >= train_start) & (df['datetime'] < train_end)
        test_mask = (df['datetime'] >= test_start) & (df['datetime'] < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) > 100 and len(test_df) > 50:
            splits.append({
                'fold': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_start': train_df['datetime'].min(),
                'train_end': train_df['datetime'].max(),
                'test_start': test_df['datetime'].min(),
                'test_end': test_df['datetime'].max(),
                'train_df': train_df,
                'test_df': test_df
            })

    logger.info(f"\nCreated {len(splits)} rolling window splits")

    # Features to use
    features = ['z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta']

    # Store results
    results = []

    # Run rolling window validation
    for split in splits:
        fold = split['fold']
        train_df = split['train_df']
        test_df = split['test_df']

        logger.info("\n" + "=" * 80)
        logger.info(f"FOLD {fold}")
        logger.info("=" * 80)
        logger.info(f"Train: {split['train_start'].date()} to {split['train_end'].date()} ({split['train_size']} samples)")
        logger.info(f"Test:  {split['test_start'].date()} to {split['test_end'].date()} ({split['test_size']} samples)")

        X_train = train_df[features]
        y_train = train_df['target']
        X_test = test_df[features]
        y_test = test_df['target']

        # Class balance
        train_balance = y_train.value_counts(normalize=True)
        test_balance = y_test.value_counts(normalize=True)
        logger.info(f"\nTrain class balance: {train_balance[1]:.2%} positive")
        logger.info(f"Test class balance: {test_balance[1]:.2%} positive")

        # Train model
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # High-confidence signals
        high_conf = y_proba > 0.65
        num_signals = high_conf.sum()
        signal_acc = accuracy_score(y_test[high_conf], y_pred[high_conf]) if num_signals > 0 else 0

        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        logger.info(f"\nHigh-Confidence Signals (>0.65):")
        logger.info(f"  Count:     {num_signals} ({num_signals/len(y_test):.1%} of test set)")
        logger.info(f"  Accuracy:  {signal_acc:.4f}")

        results.append({
            'fold': fold,
            'train_size': split['train_size'],
            'test_size': split['test_size'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'num_signals': num_signals,
            'signal_rate': num_signals / len(y_test),
            'signal_accuracy': signal_acc
        })

    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE RESULTS (Mean ± Std)")
    logger.info("=" * 80)

    results_df = pd.DataFrame(results)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'signal_rate', 'signal_accuracy']

    for metric in metrics:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        logger.info(f"{metric.upper():20s}: {mean:.4f} ± {std:.4f}")

    logger.info(f"\nAverage signals per period: {results_df['num_signals'].mean():.0f}")

    # Check consistency
    logger.info("\n" + "=" * 80)
    logger.info("ROBUSTNESS CHECK")
    logger.info("=" * 80)

    f1_std = results_df['f1'].std()
    f1_mean = results_df['f1'].mean()
    cv = f1_std / f1_mean if f1_mean > 0 else float('inf')

    logger.info(f"F1-Score Coefficient of Variation: {cv:.2f}")
    if cv < 0.2:
        logger.info("✅ Model performance is CONSISTENT across time periods")
    elif cv < 0.5:
        logger.info("⚠️  Model performance is MODERATELY CONSISTENT")
    else:
        logger.info("❌ Model performance is INCONSISTENT - High overfitting risk!")

    # Compare to expanding window
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH EXPANDING WINDOW")
    logger.info("=" * 80)
    logger.info("Rolling Window simulates realistic retraining:")
    logger.info("  ✓ Uses most recent data only (more adaptive)")
    logger.info("  ✓ Mimics production retraining strategy")
    logger.info("  ✓ Can detect if older data becomes less relevant")

    logger.info("\n" + "=" * 80)
    logger.info("ROLLING WINDOW VALIDATION COMPLETE")
    logger.info("=" * 80)

    return results_df

if __name__ == "__main__":
    results = rolling_window_validation(train_window_weeks=2)

    # Save results
    results.to_csv("rolling_window_results.csv", index=False)
    logger.info("\nResults saved to rolling_window_results.csv")
