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


def train_model(data_path="data/raw/eurusd_m1.csv", model_path="src/oracle/model.json"):
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
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


    # Drop NaN
    df = df.dropna()

    # Labeling
    # Target: 1 if price increases by > 1 pip in next 5 minutes
    horizon = 5
    threshold = 0.0001

    df['future_close'] = df[close_col].shift(-horizon)
    df['target'] = (df['future_close'] > df[close_col] + threshold).astype(int)

    # Drop last 'horizon' rows where target is invalid (NaN future_close)
    df = df.dropna()

    logger.info(f"Training data shape: {df.shape}")
    logger.info(f"Class balance: {df['target'].value_counts(normalize=True)}")

    # Features for training
    # Features for training
    features = ['z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta']

    X = df[features]
    y = df['target']

    # Train/Test Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train XGBoost
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Test Accuracy: {score:.4f}")

    # Save model
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
