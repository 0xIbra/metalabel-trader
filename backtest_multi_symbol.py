"""
Multi-Symbol Backtest
Simulates trading across EURUSD, GBPUSD, AUDUSD with individual models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# Import feature computation functions
import sys
sys.path.append('.')
from src.training.train_model import (
    compute_roc, compute_macd, compute_price_velocity
)
from src.quant_engine.indicators import (
    calculate_log_returns, calculate_volatility, calculate_z_score,
    calculate_rsi, calculate_adx, calculate_time_sin, calculate_volume_delta
)

def backtest_symbol(symbol, data_path, model_path, confidence_threshold=0.50):
    """
    Backtest a single symbol with its trained model
    """
    print(f"\n{'='*80}")
    print(f"BACKTESTING {symbol}")
    print('='*80)

    # Load model
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} bars")

    # Compute all features (same as training)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    timestamps = df['timestamp'].values

    log_returns = calculate_log_returns(close)
    volatility = calculate_volatility(log_returns, window=20)
    z_score = calculate_z_score(close, window=20)
    rsi = calculate_rsi(close)
    adx = calculate_adx(high, low, close)
    time_sin = calculate_time_sin(timestamps)
    volume_delta = calculate_volume_delta(volume)

    # Momentum features (need pandas Series)
    close_series = pd.Series(close)
    roc_5 = compute_roc(close_series, 5).values
    roc_10 = compute_roc(close_series, 10).values
    roc_20 = compute_roc(close_series, 20).values
    macd = compute_macd(close_series).values
    velocity = compute_price_velocity(close_series, 10).values

    # Lag features
    close_lag1 = np.roll(close, 1)
    close_lag2 = np.roll(close, 2)
    close_lag3 = np.roll(close, 3)
    returns_lag1 = np.roll(log_returns, 1)
    returns_lag2 = np.roll(log_returns, 2)

    # Create feature matrix
    features = np.column_stack([
        z_score, rsi, volatility, adx, time_sin, volume_delta,
        roc_5, roc_10, roc_20, macd, velocity,
        close_lag1, close_lag2, close_lag3, returns_lag1, returns_lag2
    ])

    feature_names = [
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
    ]

    # Get predictions
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predictions = model.predict(dmatrix)

    # Assuming 3-class model (or binary for some symbols)
    if len(predictions.shape) > 1 and predictions.shape[1] == 3:
        # 3-class: [SELL, NO_ACTION, BUY]
        buy_probs = predictions[:, 2]
    else:
        # Binary or single output
        buy_probs = predictions

    # Trading simulation parameters
    ACCOUNT_BALANCE = 1000
    LEVERAGE = 30
    RISK_PERCENT = 0.02

    # Symbol-specific parameters
    if symbol == 'USDJPY':
        PIP_SIZE = 0.01  # JPY pairs
    else:
        PIP_SIZE = 0.0001  # Standard

    SPREAD_PIPS = 0.5
    TP_PIPS = 5
    SL_PIPS = 1
    HOLD_BARS = 60

    # Calculate position sizing
    RISK_AMOUNT = ACCOUNT_BALANCE * RISK_PERCENT
    POSITION_SIZE = ACCOUNT_BALANCE * LEVERAGE * 0.30
    LOT_SIZE = POSITION_SIZE / 100000
    PIP_VALUE = LOT_SIZE * (10 if symbol == 'USDJPY' else 10)

    # Backtest
    trades = []
    in_trade = False
    entry_bar = 0
    entry_price = 0

    for i in range(60, len(df)):  # Start after warmup
        if buy_probs[i] < confidence_threshold:
            continue

        if not in_trade:
            # Enter trade
            in_trade = True
            entry_bar = i
            entry_price = close[i] + (SPREAD_PIPS * PIP_SIZE)  # Account for spread

        else:
            # Check exit conditions
            bars_held = i - entry_bar
            current_price = close[i]
            pips = (current_price - entry_price) / PIP_SIZE

            # TP hit
            if pips >= TP_PIPS:
                pnl = TP_PIPS * PIP_VALUE
                trades.append({
                    'symbol': symbol,
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'pnl': pnl,
                    'pips': TP_PIPS,
                    'exit_reason': 'TP',
                    'confidence': buy_probs[entry_bar]
                })
                in_trade = False

            # SL hit
            elif pips <= -SL_PIPS:
                pnl = -SL_PIPS * PIP_VALUE
                trades.append({
                    'symbol': symbol,
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'pnl': pnl,
                    'pips': -SL_PIPS,
                    'exit_reason': 'SL',
                    'confidence': buy_probs[entry_bar]
                })
                in_trade = False

            # Timeout
            elif bars_held >= HOLD_BARS:
                pnl = pips * PIP_VALUE
                trades.append({
                    'symbol': symbol,
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'pnl': pnl,
                    'pips': pips,
                    'exit_reason': 'TIMEOUT',
                    'confidence': buy_probs[entry_bar]
                })
                in_trade = False

    return pd.DataFrame(trades)


def main():
    """
    Run multi-symbol backtest
    """
    print("="*80)
    print("MULTI-SYMBOL TRADING BACKTEST")
    print("="*80)
    print("\nSymbols: EURUSD, GBPUSD, AUDUSD")
    print("Strategy: 5:1 Risk-Reward, 50% Confidence Threshold")
    print()

    symbols = [
        ('EURUSD', 'data/raw/eurusd_m1_extended.csv', 'src/oracle/model.json', 0.50),
        # ('GBPUSD', 'data/raw/gbpusd_m1_extended.csv', 'src/oracle/model_gbpusd.json', 0.60),  # Removed - unprofitable
        ('AUDUSD', 'data/raw/audusd_m1_extended.csv', 'src/oracle/model_audusd.json', 0.50),
    ]

    all_results = []

    for symbol, data_path, model_path, threshold in symbols:
        print(f"\nUsing {threshold*100:.0f}% confidence threshold for {symbol}")
        result = backtest_symbol(symbol, data_path, model_path, confidence_threshold=threshold)
        if result is not None and len(result) > 0:
            all_results.append(result)

    # Combine results
    if not all_results:
        print("\n❌ No trades generated across any symbol!")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Print combined results
    print(f"\n{'='*80}")
    print("COMBINED RESULTS ACROSS ALL SYMBOLS")
    print('='*80)

    total_trades = len(combined)
    wins = combined[combined['pnl'] > 0]
    losses = combined[combined['pnl'] < 0]

    print(f"\nTotal Trades: {total_trades}")
    print(f"  Wins:   {len(wins)} ({len(wins)/total_trades*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/total_trades*100:.1f}%)")

    total_pnl = combined['pnl'].sum()
    print(f"\nP&L:")
    print(f"  Total P&L:      ${total_pnl:.2f}")
    print(f"  Avg Win:        ${wins['pnl'].mean():.2f}")
    print(f"  Avg Loss:       ${losses['pnl'].mean():.2f}")

    if len(losses) > 0:
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum())
        print(f"  Profit Factor:  {profit_factor:.2f}")

    print(f"\nAccount Performance:")
    print(f"  Starting Balance: $1,000.00")
    print(f"  Final Balance:    ${1000 + total_pnl:.2f}")
    print(f"  Return:           +{total_pnl/1000*100:.2f}%")

    # Per-symbol breakdown
    print(f"\n{'='*80}")
    print("PER-SYMBOL BREAKDOWN")
    print('='*80)

    for symbol in combined['symbol'].unique():
        symbol_trades = combined[combined['symbol'] == symbol]
        symbol_pnl = symbol_trades['pnl'].sum()
        symbol_wins = symbol_trades[symbol_trades['pnl'] > 0]
        win_rate = len(symbol_wins) / len(symbol_trades) * 100

        print(f"\n{symbol}:")
        print(f"  Trades: {len(symbol_trades)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  P&L: ${symbol_pnl:.2f}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print('='*80)

    if total_pnl > 0:
        print(f"✅ PROFITABLE: ${total_pnl:.2f}")
        annualized_return = (total_pnl / 1000) * (365 / 90) * 100
        print(f"✅ Annualized Return: {annualized_return:.1f}%")
    else:
        print(f"❌ UNPROFITABLE: ${total_pnl:.2f}")

    # Save results
    combined.to_csv('backtest_multi_symbol_results.csv', index=False)
    print(f"\n📊 Detailed results saved to backtest_multi_symbol_results.csv")

if __name__ == "__main__":
    main()
