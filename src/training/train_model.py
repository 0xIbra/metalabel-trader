import pandas as pd
import numpy as np
import xgboost as xgb
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Trainer")

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
    adx = ((dx.shift(1) * (window - 1)) + dx) / window
    adx = dx.rolling(window).mean()
    return adx

def compute_time_sin(timestamps):
    # timestamp is unix epoch
    seconds_in_day = 24 * 60 * 60
    return np.sin(2 * np.pi * (timestamps % seconds_in_day) / seconds_in_day)

def compute_volume_delta(volume):
    # Log change in volume
    # Handle zero volume by adding small epsilon or replacing 0 with 1
    v = volume.replace(0, 1)
    return np.log(v / v.shift(1))

def compute_roc(series, period=10):
    """Rate of Change - momentum indicator"""
    return ((series - series.shift(period)) / series.shift(period)) * 100

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD - Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram  # Return histogram (most predictive)

def compute_price_velocity(close, period=5):
    """Price velocity - rate of change of returns (acceleration)"""
    returns = close.pct_change()
    velocity = returns.diff(period)
    return velocity

def apply_triple_barrier_labels(prices, take_profit_pips=2, stop_loss_pips=1, timeout_bars=10, pip_value=0.0001):
    """Triple Barrier Method for labeling"""
    from numba import jit

    @jit(nopython=True)
    def _triple_barrier(prices, tp, sl, timeout):
        n = len(prices)
        labels = np.zeros(n, dtype=np.int32)

        for i in range(n - timeout):
            entry_price = prices[i]
            tp_price = entry_price + tp
            sl_price = entry_price - sl

            for j in range(1, timeout + 1):
                if i + j >= n:
                    break
                future_price = prices[i + j]

                if future_price >= tp_price:
                    labels[i] = 1  # BUY signal
                    break
                elif future_price <= sl_price:
                    labels[i] = -1  # SELL/Avoid
                    break
        return labels

    take_profit = take_profit_pips * pip_value
    stop_loss = stop_loss_pips * pip_value
    labels = _triple_barrier(prices, take_profit, stop_loss, timeout_bars)

    logger.info(f"\nTriple Barrier Label Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  Label {label:2d}: {count:6d} ({count/len(labels)*100:5.2f}%)")

    return labels


def train_model(symbol='EURUSD', data_path=None, model_path=None):
    """
    Train XGBoost model with triple barrier labeling.

    Args:
        symbol: Forex symbol (EURUSD, GBPUSD, USDJPY, AUDUSD)
        data_path: Path to CSV data (auto-generated if None)
        model_path: Output model path (auto-generated if None)
    """
    # Auto-generate paths based on symbol
    if data_path is None:
        data_path = f"data/raw/{symbol.lower()}_m1_extended.csv"

    if model_path is None:
        model_path = f"src/oracle/model_{symbol.lower()}.json"

    logger.info(f"Training model for {symbol}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {model_path}")

    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.info("Trying fallback to original dataset...")
        data_path = "data/raw/eurusd_m1.csv"
        if not os.path.exists(data_path):
            logger.error("No data files found!")
            return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Ensure timestamp is sorted
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Feature Engineering
    logger.info("Computing features...")
    window = 20

    # Use 'close' price (EODHD might have 'close' or 'Close')
    # Check columns
    close_col = 'close' if 'close' in df.columns else 'Close'

    df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window).std()
    df['ma'] = df[close_col].rolling(window=window).mean()
    df['std'] = df[close_col].rolling(window=window).std()
    df['z_score'] = (df[close_col] - df['ma']) / df['std']
    df['rsi'] = compute_rsi(df[close_col], window=14)

    # New Features
    # EODHD M1 data has 'high', 'low', 'open', 'close', 'volume'
    # Ensure columns exist and are lower case
    df.columns = [c.lower() for c in df.columns]

    df['adx'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    df['time_sin'] = compute_time_sin(df['timestamp'])
    df['volume_delta'] = compute_volume_delta(df['volume'])

    # === MOMENTUM FEATURES ===
    logger.info("Computing momentum features...")
    df['roc_5'] = compute_roc(df['close'], period=5)
    df['roc_10'] = compute_roc(df['close'], period=10)
    df['roc_20'] = compute_roc(df['close'], period=20)
    df['macd'] = compute_macd(df['close'])
    df['velocity'] = compute_price_velocity(df['close'], period=5)

    # === LAG FEATURES ===
    logger.info("Computing lag features...")
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['close_lag3'] = df['close'].shift(3)
    df['returns_lag1'] = df['log_returns'].shift(1)
    df['returns_lag2'] = df['log_returns'].shift(2)


    # Drop NaN
    df = df.dropna()

    # ===== TRIPLE BARRIER LABELING =====
    logger.info("Applying Triple Barrier labeling...")
    logger.info("Parameters: TP=5 pips, SL=1 pip (5:1 RR), Timeout=60 bars")
    labels = apply_triple_barrier_labels(
        df['close'].values,
        take_profit_pips=5,   # 5 pips for 5:1 risk-reward (optimal expectancy)
        stop_loss_pips=1,     # Keep at 1 pip
        timeout_bars=60       # 60 bars (1 hour) for 5-pip move to develop
    )
    df['target'] = labels

    logger.info(f"Training data shape after labeling: {df.shape}")


    # Features for training (expanded from 6 to 16)
    features = [
        # Technical indicators
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        # Momentum
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        # Lag features
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
    ]

    logger.info(f"Total features: {len(features)}")
    X = df[features]
    y = df['target']

    # Train/Test Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train XGBoost (Multi-class)
    logger.info("Training XGBoost model (3-class classification)...")

    # Check class distribution
    class_counts = y_train.value_counts()
    logger.info(f"Class distribution in training set: {class_counts.to_dict()}")

    # If NO_ACTION class is very rare (< 1%), remove it and use binary classification
    if 0 in class_counts and class_counts[0] / len(y_train) < 0.01:
        logger.warning("NO_ACTION class too rare (<1%), switching to binary classification (SELL vs BUY)")
        # Remove NO_ACTION class
        mask = y_train != 0
        X_train = X_train[mask]
        y_train = y_train[mask]

        mask_test = y_test != 0
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        logger.info(f"After filtering: {len(y_train)} training samples, {len(y_test)} test samples")

        # Remap to binary: -1 → 0, 1 → 1
        y_train_remapped = (y_train + 1) // 2  # -1→0, 1→1
        y_test_remapped = (y_test + 1) // 2

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        model.fit(X_train, y_train_remapped)
        y_pred_remapped = model.predict(X_test)
        y_pred = y_pred_remapped * 2 - 1  # 0→-1, 1→1

        target_names = ['SELL/Avoid (-1)', 'BUY (1)']
    else:
        # Original 3-class approach
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

        y_train_remapped = y_train + 1
        model.fit(X_train, y_train_remapped)

        y_test_remapped = y_test + 1
        y_pred_remapped = model.predict(X_test)
        y_pred = y_pred_remapped - 1

        target_names = ['SELL/Avoid (-1)', 'NO_ACTION (0)', 'BUY (1)']

    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=target_names))


    # Save model
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost model for forex symbol')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Forex symbol (EURUSD, GBPUSD, USDJPY, AUDUSD)')
    args = parser.parse_args()

    train_model(symbol=args.symbol)
