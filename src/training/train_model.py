import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from src.quant_engine.indicators import calculate_bollinger_bands, calculate_atr, calculate_pivot_points

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

def apply_volatility_expansion_labels(prices, atr, horizon=12, multiplier=2.0):
    """
    Volatility Expansion Labeling (Trend Following).
    Target: Does price move > 2 * ATR in the breakout direction within 'horizon' bars?

    Labels:
    1: Bullish Trend (Price moves up > 2*ATR)
    -1: Bearish Trend (Price moves down > 2*ATR)
    0: Chop/Fakeout (Price stays within range)
    """
    from numba import jit

    @jit(nopython=True)
    def _label_trend(prices, atr, horizon, multiplier):
        n = len(prices)
        labels = np.zeros(n, dtype=np.int32)

        for i in range(n - horizon):
            current_price = prices[i]
            current_atr = atr[i]

            if current_atr == 0:
                continue

            threshold = current_atr * multiplier

            # Check future window
            max_price = -1.0
            min_price = 1e9

            for j in range(1, horizon + 1):
                if i + j >= n:
                    break
                p = prices[i + j]
                if p > max_price: max_price = p
                if p < min_price: min_price = p

            # Determine label
            # Priority to the direction of the breakout if both hit?
            # Ideally we want the FIRST hit.

            # Simplified: Check net displacement at end of horizon?
            # No, we want to know if it HIT the target.

            hit_up = (max_price - current_price) >= threshold
            hit_down = (current_price - min_price) >= threshold

            if hit_up and not hit_down:
                labels[i] = 1
            elif hit_down and not hit_up:
                labels[i] = -1
            elif hit_up and hit_down:
                # Volatility explosion in both directions (rare but possible)
                # Check which happened first? For now, treat as Chop (0) or ignore.
                labels[i] = 0
            else:
                labels[i] = 0

        return labels

    labels = _label_trend(prices, atr, horizon, multiplier)

    logger.info(f"\nVolatility Expansion Label Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        logger.info(f"  Label {label:2d}: {count:6d} ({count/len(labels)*100:5.2f}%)")

    return labels


def train_global_model(data_dir="data/swing", model_path="src/oracle/model_swing_global.json"):
    """
    Train a single global model on all available swing data.
    """
    logger.info(f"Training GLOBAL model from {data_dir}")

    all_files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]
    if not all_files:
        logger.error(f"No data files found in {data_dir}")
        return

    dfs = []

    for filename in all_files:
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)

        logger.info(f"Processing {symbol}...")
        try:
            df = pd.read_csv(path)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Feature Engineering
            # Ensure columns exist and are lower case
            df.columns = [c.lower() for c in df.columns]

            # Compute all features (reuse logic from train_model but refactored ideally,
            # for now copy-paste or we should have refactored into a function.
            # To avoid massive code duplication, let's extract feature_engineering function)

            # ... actually, let's just call train_model logic or refactor.
            # Refactoring is better.

            df = compute_features_and_labels(df)

            # Add symbol column for reference (optional, model shouldn't see it)
            # df['symbol'] = symbol

            dfs.append(df)

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    if not dfs:
        logger.error("No valid dataframes created")
        return

    # Combine all data
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined Data Shape: {full_df.shape}")

    # Train
    train_on_dataframe(full_df, model_path)

def compute_features_and_labels(df, currency_strength=None, symbol=None):
    """
    Refactored feature engineering logic
    """
    window = 20

    # Basic Features
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window).std()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['z_score'] = (df['close'] - df['ma']) / df['std']
    df['rsi'] = compute_rsi(df['close'], window=14)

    df['adx'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    df['time_sin'] = compute_time_sin(df['timestamp'])
    df['volume_delta'] = compute_volume_delta(df['volume'])

    # Swing Features
    upper, middle, lower, bandwidth = calculate_bollinger_bands(df['close'].values, window=20, num_std=2)
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    df['bb_width'] = bandwidth
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['atr'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, window=14)
    df['atr_pct'] = df['atr'] / df['close'] * 100

    pivot, r1, s1 = calculate_pivot_points(df['high'].values, df['low'].values, df['close'].values)
    df['dist_pivot'] = (df['close'] - pivot) / df['atr']

    # Momentum
    df['roc_5'] = compute_roc(df['close'], period=5)
    df['roc_10'] = compute_roc(df['close'], period=10)
    df['roc_20'] = compute_roc(df['close'], period=20)
    df['macd'] = compute_macd(df['close'])
    df['velocity'] = compute_price_velocity(df['close'], period=5)

    # Lag
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['close_lag3'] = df['close'].shift(3)
    df['returns_lag1'] = df['log_returns'].shift(1)
    df['returns_lag2'] = df['log_returns'].shift(2)

    # Currency Strength (if provided)
    if currency_strength is not None and symbol is not None:
        # Parse symbol (e.g., EURUSD -> Base: EUR, Quote: USD)
        base = symbol[:3]
        quote = symbol[3:]

        # Get strength series
        base_strength = currency_strength.get(base, pd.Series(0, index=df['timestamp']))
        quote_strength = currency_strength.get(quote, pd.Series(0, index=df['timestamp']))

        # Align timestamps (reindex to match df)
        # Assuming currency_strength is a dict of Series indexed by timestamp
        # We need to map df['timestamp'] to the strength values

        # Efficient mapping
        df['base_strength'] = df['timestamp'].map(base_strength).fillna(0)
        df['quote_strength'] = df['timestamp'].map(quote_strength).fillna(0)

        # Feature: Strength Differential
        df['strength_diff'] = df['base_strength'] - df['quote_strength']
    else:
        df['strength_diff'] = 0.0

    df = df.dropna()

    # Labeling
    labels = apply_volatility_expansion_labels(
        df['close'].values,
        df['atr'].values,
        horizon=12,
        multiplier=2.0
    )
    df['target'] = labels

    return df

def train_on_dataframe(df, model_path):
    """
    Train XGBoost on a prepared dataframe
    """
    features = [
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2',
        'strength_diff' # New Feature
    ]

    X = df[features]
    y = df['target']

    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

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

    # Remap labels for XGBoost (must be 0, 1, 2)
    # Our labels: -1, 0, 1
    # Map: -1->0, 0->1, 1->2
    y_train_map = y_train + 1
    y_test_map = y_test + 1

    model.fit(X_train, y_train_map)

    y_pred_map = model.predict(X_test)
    y_pred = y_pred_map - 1

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(classification_report(y_test, y_pred, target_names=['SELL', 'CHOP', 'BUY']))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='EURUSD')
    parser.add_argument('--global_train', action='store_true', help='Train global model on all data')
    args = parser.parse_args()

    if args.global_train:
        train_global_model()
    else:
        # Legacy single symbol mode (modified to use new functions)
        # For simplicity, just reuse the new logic
        path = f"data/swing/{args.symbol.lower()}_h4.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df = compute_features_and_labels(df)
            train_on_dataframe(df, f"src/oracle/model_swing_{args.symbol.lower()}.json")
        else:
            logger.error(f"Data not found for {args.symbol}")
