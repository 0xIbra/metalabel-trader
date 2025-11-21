"""
Backtest for H4 Swing Trading Strategy with Macro Features
Compares baseline vs macro-enhanced model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
sys.path.append('.')

from src.training.train_model import compute_features_and_labels
from train_with_macro import load_macro_features, merge_sentiment_features

def backtest_macro_strategy(
    data_dir="data/swing",
    model_path="src/oracle/model_swing_macro.json",
    baseline_path="src/oracle/model_swing_global.json",
    confidence_threshold=0.50,
    test_split=0.8
):
    """
Backtest macro-enhanced model vs baseline
    """
    print("=" * 80)
    print("MACRO-ENHANCED BACKTEST")
    print("=" * 80)

    # Load models
    try:
        macro_model = xgb.Booster()
        macro_model.load_model(model_path)
        print(f"✅ Loaded macro model: {model_path}")

        baseline_model = xgb.Booster()
        baseline_model.load_model(baseline_path)
        print(f"✅ Loaded baseline model: {baseline_path}\n")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return None

    # Load sentiment
    sentiment_df = load_macro_features()

    files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]

    macro_trades = []
    baseline_trades = []

    for filename in files[:10]:  # Test on 10 symbols
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)

        print(f"\n{'─'*80}")
        print(f"Testing {symbol}...")

        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Compute base features
        df = compute_features_and_labels(df)

        # Add macro features
        df = merge_sentiment_features(df, symbol, sentiment_df)
        df['sentiment_ma7'] = df['sentiment'].rolling(7, min_periods=1).mean()
        df['sentiment_std7'] = df['sentiment'].rolling(7, min_periods=1).std().fillna(0)

        # Split
        split_idx = int(len(df) * test_split)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        # Base features
        base_features = [
            'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
            'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
            'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
            'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
        ]

        # Macro features
        macro_features = base_features + ['sentiment', 'sentiment_ma7', 'sentiment_std7']

        # Baseline predictions
        X_base = test_df[base_features]
        dmatrix_base = xgb.DMatrix(X_base, feature_names=base_features)
        baseline_preds = baseline_model.predict(dmatrix_base)
        baseline_probs = baseline_preds[:, 2] if len(baseline_preds.shape) > 1 else baseline_preds

        # Macro predictions
        X_macro = test_df[macro_features]
        dmatrix_macro = xgb.DMatrix(X_macro, feature_names=macro_features)
        macro_preds = macro_model.predict(dmatrix_macro)
        macro_probs = macro_preds[:, 2] if len(macro_preds.shape) > 1 else macro_preds

        # Simulate trades
        baseline_trades.extend(simulate_trades(test_df, baseline_probs, symbol, confidence_threshold))
        macro_trades.extend(simulate_trades(test_df, macro_probs, symbol, confidence_threshold))

    # Compare results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    compare_results("Baseline", baseline_trades)
    print()
    compare_results("Macro-Enhanced", macro_trades)

    # Calculate improvement
    if baseline_trades and macro_trades:
        base_pf = calc_profit_factor(baseline_trades)
        macro_pf = calc_profit_factor(macro_trades)

        print("\n" + "=" * 80)
        print(f"PROFIT FACTOR IMPROVEMENT: {((macro_pf - base_pf) / base_pf * 100):+.1f}%")
        print("=" * 80)

def simulate_trades(df, buy_probs, symbol, threshold):
    """Simulate trades (same logic as original backtest)"""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    tp_price = None
    sl_price = None
    atr_entry = None
    highest_high = None

    for i in range(len(df)):
        if in_position:
            current_price = df.iloc[i]['close']
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']

            if high > highest_high:
                highest_high = high

            trail_sl = highest_high - (1.5 * atr_entry)
            if trail_sl > sl_price:
                sl_price = trail_sl

            exit_reason = None
            exit_price = current_price

            if high >= tp_price:
                exit_reason = 'TP'
                exit_price = tp_price
            elif low <= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
            elif i - entry_idx >= 18:
                exit_reason = 'TIMEOUT'

            if exit_reason:
                pnl = (exit_price - entry_price) * 1000
                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                in_position = False

        else:
            if buy_probs[i] >= threshold:
                entry_price = df.iloc[i]['close']
                atr_entry = df.iloc[i]['atr']
                tp_price = entry_price + (2.0 * atr_entry)
                sl_price = entry_price - (1.0 * atr_entry)
                highest_high = entry_price
                entry_idx = i
                in_position = True

    return trades

def compare_results(name, trades):
    """Print performance metrics"""
    if not trades:
        print(f"{name}: No trades")
        return

    total_pnl = sum([t['pnl'] for t in trades])
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]

    win_rate = len(wins) / len(trades)
    pf = calc_profit_factor(trades)

    print(f"{name}:")
    print(f"  Trades:        {len(trades)}")
    print(f"  Win Rate:      {win_rate:.1%}")
    print(f"  Total PnL:     ${total_pnl:.2f}")
    print(f"  Profit Factor: {pf:.2f}")

def calc_profit_factor(trades):
    """Calculate profit factor"""
    wins = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    losses = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    return wins / losses if losses > 0 else float('inf')

if __name__ == "__main__":
    backtest_macro_strategy()
