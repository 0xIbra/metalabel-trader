import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelEvaluator")

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

def evaluate_model(data_path="data/raw/eurusd_m1.csv", model_path="src/oracle/model.json"):
    """
    Comprehensive model evaluation with multiple metrics
    """
    logger.info("=" * 70)
    logger.info("XGBOOST MODEL EVALUATION")
    logger.info("=" * 70)

    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Feature Engineering
    logger.info("Computing features...")
    window = 20
    close_col = 'close' if 'close' in df.columns else 'Close'
    df.columns = [c.lower() for c in df.columns]

    df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window).std()
    df['ma'] = df[close_col].rolling(window=window).mean()
    df['std'] = df[close_col].rolling(window=window).std()
    df['z_score'] = (df[close_col] - df['ma']) / df['std']
    df['rsi'] = compute_rsi(df[close_col], window=14)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    df['time_sin'] = compute_time_sin(df['timestamp'])
    df['volume_delta'] = compute_volume_delta(df['volume'])

    df = df.dropna()

    # Labeling
    horizon = 5
    threshold = 0.0001
    df['future_close'] = df[close_col].shift(-horizon)
    df['target'] = (df['future_close'] > df[close_col] + threshold).astype(int)
    df = df.dropna()

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['target'].value_counts(normalize=True)}")

    # Features
    features = ['z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta']
    X = df[features]
    y = df['target']

    # Train/Test Split (80/20)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Load or train model
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        logger.info("Training new model...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    # Predictions
    logger.info("\nGenerating predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SET PERFORMANCE")
    logger.info("=" * 70)
    logger.info(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
    logger.info(f"F1-Score: {f1_score(y_train, y_train_pred):.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("TEST SET PERFORMANCE")
    logger.info("=" * 70)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    logger.info(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

    # Confusion Matrix
    logger.info("\n" + "=" * 70)
    logger.info("CONFUSION MATRIX (Test Set)")
    logger.info("=" * 70)
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n{cm}")
    logger.info(f"\nTrue Negatives: {cm[0][0]}")
    logger.info(f"False Positives: {cm[0][1]}")
    logger.info(f"False Negatives: {cm[1][0]}")
    logger.info(f"True Positives: {cm[1][1]}")

    # Classification Report
    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION REPORT (Test Set)")
    logger.info("=" * 70)
    logger.info(f"\n{classification_report(y_test, y_test_pred, target_names=['No Move', 'Price Up'])}")

    # Feature Importance
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 70)
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    logger.info(f"\n{feature_importance.to_string(index=False)}")

    # Trading Performance Simulation
    logger.info("\n" + "=" * 70)
    logger.info("SIMULATED TRADING PERFORMANCE (0.65 Threshold)")
    logger.info("=" * 70)

    # Apply 0.65 threshold (as in Oracle)
    high_conf_signals = y_test_proba > 0.65
    num_signals = high_conf_signals.sum()

    if num_signals > 0:
        signal_accuracy = accuracy_score(
            y_test[high_conf_signals],
            y_test_pred[high_conf_signals]
        )
        logger.info(f"Number of high-confidence signals (>0.65): {num_signals}")
        logger.info(f"High-confidence signal accuracy: {signal_accuracy:.4f}")
        logger.info(f"Signal rate: {num_signals / len(y_test):.2%} of all samples")
    else:
        logger.info("No signals above 0.65 threshold!")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)

    return {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }

if __name__ == "__main__":
    evaluate_model()
