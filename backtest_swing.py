"""
Backtest for H4 Swing Trading Strategy
Tests the global model on out-of-sample data with realistic execution
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
sys.path.append('.')

from src.training.train_model import compute_features_and_labels
from src.quant_engine.indicators import calculate_atr

def backtest_swing_strategy(
    data_dir="data/swing",
    model_path="src/oracle/model_swing_global.json",
    confidence_threshold=0.50,
    test_split=0.8
):
    """
    Backtest the swing trading strategy on multiple symbols
    """
    print("=" * 80)
    print("SWING TRADING STRATEGY BACKTEST")
    print("=" * 80)

    # Load model
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"✅ Loaded model: {model_path}\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Get all data files
    files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]

    all_trades = []
    symbol_results = {}

    for filename in files:  # Test on ALL symbols
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)

        print(f"\n{'─'*80}")
        print(f"Testing {symbol}...")
        print('─'*80)

        # Load and prepare data
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Compute features
        df = compute_features_and_labels(df)

        # Split into train/test
        split_idx = int(len(df) * test_split)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        print(f"  Train: {split_idx} bars | Test: {len(test_df)} bars")

        # Prepare features for prediction
        features = [
            'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
            'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
            'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
            'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
        ]

        X_test = test_df[features]
        dmatrix = xgb.DMatrix(X_test, feature_names=features)
        predictions = model.predict(dmatrix)

        # Get BUY probability (class 2)
        if len(predictions.shape) > 1 and predictions.shape[1] == 3:
            buy_probs = predictions[:, 2]
        else:
            buy_probs = predictions

        # Simulate trading
        trades = simulate_trades(
            test_df,
            buy_probs,
            symbol,
            confidence_threshold
        )

        all_trades.extend(trades)

        # Calculate symbol metrics
        if trades:
            symbol_pnl = sum([t['pnl'] for t in trades])
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] < 0]

            symbol_results[symbol] = {
                'trades': len(trades),
                'pnl': symbol_pnl,
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) if trades else 0
            }

            print(f"  Trades: {len(trades)} | PnL: ${symbol_pnl:.2f} | Win Rate: {symbol_results[symbol]['win_rate']:.1%}")

    # Overall Performance
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    if not all_trades:
        print("❌ No trades executed")
        return None

    total_pnl = sum([t['pnl'] for t in all_trades])
    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] < 0]

    win_rate = len(wins) / len(all_trades)
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    profit_factor = abs(sum([t['pnl'] for t in wins]) / sum([t['pnl'] for t in losses])) if losses else float('inf')

    # Calculate max drawdown
    equity_curve = np.cumsum([t['pnl'] for t in all_trades])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Sharpe Ratio (annualized, assuming 252 trading days, ~6 H4 bars per day)
    returns = [t['pnl'] for t in all_trades]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 6) if np.std(returns) > 0 else 0

    print(f"\nTotal Trades:    {len(all_trades)}")
    print(f"Winning Trades:  {len(wins)} ({win_rate:.1%})")
    print(f"Losing Trades:   {len(losses)}")
    print(f"\nTotal PnL:       ${total_pnl:.2f}")
    print(f"Avg Win:         ${avg_win:.2f}")
    print(f"Avg Loss:        ${avg_loss:.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Max Drawdown:    ${max_drawdown:.2f}")
    print(f"Sharpe Ratio:    {sharpe:.2f}")

    print("\n" + "─" * 80)
    print("PER-SYMBOL BREAKDOWN")
    print("─" * 80)
    for sym, res in symbol_results.items():
        print(f"{sym:10s} | Trades: {res['trades']:3d} | PnL: ${res['pnl']:8.2f} | Win Rate: {res['win_rate']:5.1%}")

    # Save results
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv('backtest_swing_results.csv', index=False)
    print(f"\n✅ Results saved to backtest_swing_results.csv")

    return {
        'total_trades': len(all_trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe
    }

def simulate_trades(df, buy_probs, symbol, threshold):
    """
    Simulate trades based on model predictions
    Uses ATR-based TP/SL like the live bot
    """
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

            # Update highest high
            if high > highest_high:
                highest_high = high

            # Trailing stop (1.5x ATR from highest high)
            trail_sl = highest_high - (1.5 * atr_entry)
            if trail_sl > sl_price:
                sl_price = trail_sl

            # Check exit conditions
            exit_reason = None
            exit_price = current_price

            if high >= tp_price:
                exit_reason = 'TP'
                exit_price = tp_price
            elif low <= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
            elif i - entry_idx >= 18:  # 72 hours / 4 = 18 bars
                exit_reason = 'TIMEOUT'

            if exit_reason:
                pnl = (exit_price - entry_price) * 1000  # Simplified: 0.1 lot = 1000 units

                trades.append({
                    'symbol': symbol,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'bars_held': i - entry_idx
                })

                in_position = False

        else:
            # Check for entry signal
            if buy_probs[i] >= threshold:
                entry_price = df.iloc[i]['close']
                atr_entry = df.iloc[i]['atr']

                # Set TP/SL based on ATR
                tp_price = entry_price + (2.0 * atr_entry)
                sl_price = entry_price - (1.0 * atr_entry)
                highest_high = entry_price

                entry_idx = i
                in_position = True

    return trades

if __name__ == "__main__":
    results = backtest_swing_strategy()
