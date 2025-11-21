"""
Enhanced training script with Macro Features
Adds sentiment and economic calendar awareness to the model
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('.')

from src.training.train_model import compute_features_and_labels, train_on_dataframe
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MacroTrainer")

def load_macro_features():
    """Load sentiment and economic calendar data"""
    # News Sentiment
    try:
        sentiment_df = pd.read_csv('data/macro/news_sentiment.csv')
        # Parse sentiment from string representation
        sentiment_df['sentiment_norm'] = sentiment_df[sentiment_df.columns[0]].apply(
            lambda x: eval(x)['normalized'] if isinstance(x, str) and 'normalized' in x else 0
        )
        sentiment_df['date'] = sentiment_df[sentiment_df.columns[0]].apply(
            lambda x: eval(x)['date'] if isinstance(x, str) and 'date' in x else ''
        )
        sentiment_df['symbol'] = sentiment_df['symbol']

        logger.info(f"Loaded {len(sentiment_df)} sentiment records")
        return sentiment_df
    except Exception as e:
        logger.warning(f"Failed to load sentiment: {e}")
        return None

def merge_sentiment_features(df, symbol, sentiment_df):
    """Merge sentiment data into price dataframe"""
    if sentiment_df is None:
        # Add dummy sentiment column
        df['sentiment'] = 0
        return df

    # Filter for this symbol
    symbol_sentiment = sentiment_df[sentiment_df['symbol'] == symbol].copy()

    if symbol_sentiment.empty:
        df['sentiment'] = 0
        return df

    # Convert timestamp to date for merging
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date.astype(str)
    symbol_sentiment['date'] = symbol_sentiment['date'].astype(str)

    # Merge on date
    df = df.merge(
        symbol_sentiment[['date', 'sentiment_norm']],
        on='date',
        how='left'
    )

    # Fill missing sentiment with 0
    df['sentiment'] = df['sentiment_norm'].fillna(0)
    df = df.drop(columns=['sentiment_norm', 'date'], errors='ignore')

    return df

def train_global_model_with_macro(
    data_dir="data/swing",
    model_path="src/oracle/model_swing_macro.json"
):
    """
    Train global model with macro features
    """
    logger.info("="*80)
    logger.info("TRAINING WITH MACRO FEATURES")
    logger.info("="*80)

    # Load macro data
    sentiment_df = load_macro_features()

    # Get all symbols
    files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]
    dfs = []

    for filename in files:
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)

        logger.info(f"\nProcessing {symbol}...")
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Compute base features
            df = compute_features_and_labels(df)

            # Add macro features
            df = merge_sentiment_features(df, symbol, sentiment_df)

            # Add rolling sentiment features
            df['sentiment_ma7'] = df['sentiment'].rolling(7, min_periods=1).mean()
            df['sentiment_std7'] = df['sentiment'].rolling(7, min_periods=1).std().fillna(0)

            dfs.append(df)
            logger.info(f"  ✅ {len(df)} bars with macro features")

        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")

    if not dfs:
        logger.error("No data processed")
        return

    # Combine all data
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nCombined Data: {full_df.shape}")
    logger.info(f"Features: {full_df.columns.tolist()}")

    # Update feature list
    features = [
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2',
        # New macro features
        'sentiment', 'sentiment_ma7', 'sentiment_std7'
    ]

    # Filter to only include rows with all features
    full_df = full_df.dropna(subset=features + ['target'])

    logger.info(f"After dropna: {full_df.shape}")
    logger.info(f"\nSentiment Stats:")
    logger.info(f"  Mean: {full_df['sentiment'].mean():.4f}")
    logger.info(f"  Std:  {full_df['sentiment'].std():.4f}")
    logger.info(f"  Range: [{full_df['sentiment'].min():.4f}, {full_df['sentiment'].max():.4f}]")

    # Train with updated features
    train_on_dataframe_macro(full_df, model_path, features)

def train_on_dataframe_macro(df, model_path, features):
    """Modified training function with macro features"""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    X = df[features]
    y = df['target']

    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info("\nTraining XGBoost with macro features...")

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,  # Increased depth for macro features
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    # Remap labels
    y_train_map = y_train + 1
    y_test_map = y_test + 1

    model.fit(X_train, y_train_map)

    y_pred_map = model.predict(X_test)
    y_pred = y_pred_map - 1

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\nTest Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['SELL', 'CHOP', 'BUY']))

    # Feature Importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 10 Features:")
    logger.info(feature_importance.head(10).to_string(index=False))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logger.info(f"\n✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_global_model_with_macro()
