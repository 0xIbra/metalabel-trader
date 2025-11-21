import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

def backtest_trading_strategy(data_path="data/raw/eurusd_m1_extended.csv", model_path="src/oracle/model.json"):
    """
    Backtest the trained model with realistic trading simulation
    """
    print("=" * 80)
    print("TRADING BACKTEST SIMULATION")
    print("=" * 80)

    # Load model
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"✅ Loaded model from {model_path}\n")
    except:
        print(f"❌ Could not load model from {model_path}")
        return

    # Load and prepare data (same as training)
    print("Loading data...")
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Compute features (matching training - NOW WITH 16 FEATURES)
    from src.training.train_model import (
        compute_rsi, compute_adx, compute_time_sin, compute_volume_delta,
        compute_roc, compute_macd, compute_price_velocity
    )

    window = 20
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=window).std()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['z_score'] = (df['close'] - df['ma']) / df['std']
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['adx'] = compute_adx(df['high'], df['low'], df['close'], window=14)
    df['time_sin'] = compute_time_sin(df['timestamp'])
    df['volume_delta'] = compute_volume_delta(df['volume'])

    # Momentum features
    df['roc_5'] = compute_roc(df['close'], period=5)
    df['roc_10'] = compute_roc(df['close'], period=10)
    df['roc_20'] = compute_roc(df['close'], period=20)
    df['macd'] = compute_macd(df['close'])
    df['velocity'] = compute_price_velocity(df['close'], period=5)

    # Lag features
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['close_lag3'] = df['close'].shift(3)
    df['returns_lag1'] = df['log_returns'].shift(1)
    df['returns_lag2'] = df['log_returns'].shift(2)

    df = df.dropna()

    features = [
        'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
        'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
        'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
    ]
    X = df[features]

    # Get predictions AND probabilities
    print("Generating predictions with confidence scores...")
    predictions_remapped = model.predict(X)
    probabilities = model.predict_proba(X)  # Shape: (n_samples, 3) for 3 classes

    predictions = predictions_remapped - 1  # 0→-1, 1→0, 2→1

    # For BUY signals (class 2 after remapping, which is prediction=1), get probability of class 2
    buy_probabilities = probabilities[:, 2]  # Probability of BUY class

    df['prediction'] = predictions
    df['buy_probability'] = buy_probabilities
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Trading simulation parameters
    ACCOUNT_BALANCE = 1000  # $1,000 starting capital
    LEVERAGE = 30  # 1:30 leverage
    RISK_PERCENT = 0.02  # Risk 2% per trade

    CONFIDENCE_THRESHOLD = 0.50  # Lower to 50% for rarer 5-pip targets

    # Calculate position size with proper risk management
    # Risk amount = Account × Risk%
    risk_per_trade = ACCOUNT_BALANCE * RISK_PERCENT  # $20 per trade

    # With 1 pip SL, position size needed for $20 risk:
    # For EURUSD: 1 pip = $10 per standard lot (100,000 units)
    # So for $20 risk with 1 pip SL, we need 2 standard lots
    # But that requires huge margin ($200,000 / 30 = $6,667 > account!)

    # Instead, use conservative position sizing:
    # Use 30% of account as margin = $300
    # With 1:30 leverage: $300 × 30 = $9,000 notional
    # This is 0.09 lot (9,000 units)
    # Pip value for 0.09 lot = $0.90

    POSITION_SIZE = 9000  # 0.09 lot in dollars
    PIP_VALUE = 0.90  # $0.90 per pip for 0.09 lot
    LOT_SIZE = 0.09  # 0.09 lot
    MARGIN_USED = POSITION_SIZE / LEVERAGE  # $300 margin per trade

    SPREAD_PIPS = 0.5  # 0.5 pip spread (conservative)
    TAKE_PROFIT_PIPS = 5  # 5 pips TP (5:1 risk-reward ratio - optimal expectancy)
    STOP_LOSS_PIPS = 1  # 1 pip SL
    HOLD_BARS = 60  # 60 bars (1 hour) to allow 5-pip moves

    # Backtesting
    print("\nRunning backtest...")
    print(f"Account Balance: ${ACCOUNT_BALANCE:,.0f}")
    print(f"Leverage: 1:{LEVERAGE}")
    print(f"Risk Per Trade: {RISK_PERCENT:.1%} (${risk_per_trade:.2f})")
    print(f"Position Size: ${POSITION_SIZE:,.0f} ({LOT_SIZE} lot)")
    print(f"Margin Per Trade: ${MARGIN_USED:.2f} ({MARGIN_USED/ACCOUNT_BALANCE:.1%} of account)")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%} (only trade high-confidence signals)")
    print(f"Pip Value: ${PIP_VALUE:.2f}")
    print(f"Spread: {SPREAD_PIPS} pips")
    print(f"TP: {TAKE_PROFIT_PIPS} pips, SL: {STOP_LOSS_PIPS} pips")
    print(f"Max Hold: {HOLD_BARS} bars\n")

    trades = []
    i = 0

    while i < len(df) - HOLD_BARS:
        signal = df.iloc[i]['prediction']
        confidence = df.iloc[i]['buy_probability']

        # Only trade on BUY signals (prediction == 1) with high confidence
        if signal == 1 and confidence >= CONFIDENCE_THRESHOLD:
            entry_price = df.iloc[i]['close']
            entry_time = df.iloc[i]['datetime']

            # Account for spread on entry
            entry_price_w_spread = entry_price + (SPREAD_PIPS * 0.0001)

            tp_price = entry_price_w_spread + (TAKE_PROFIT_PIPS * 0.0001)
            sl_price = entry_price_w_spread - (STOP_LOSS_PIPS * 0.0001)

            # Look forward to see what happens
            exit_reason = 'TIMEOUT'
            exit_price = None
            exit_time = None
            pnl_pips = 0

            for j in range(1, HOLD_BARS + 1):
                if i + j >= len(df):
                    break

                future_price = df.iloc[i + j]['close']

                # Check TP
                if future_price >= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price
                    exit_time = df.iloc[i + j]['datetime']
                    pnl_pips = TAKE_PROFIT_PIPS
                    break

                # Check SL
                elif future_price <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price
                    exit_time = df.iloc[i + j]['datetime']
                    pnl_pips = -STOP_LOSS_PIPS
                    break

            # Timeout
            if exit_price is None:
                exit_price = df.iloc[i + HOLD_BARS]['close']
                exit_time = df.iloc[i + HOLD_BARS]['datetime']
                pnl_pips = (exit_price - entry_price_w_spread) / 0.0001

            pnl_dollars = pnl_pips * PIP_VALUE

            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price_w_spread,
                'exit_price': exit_price,
                'pnl_pips': pnl_pips,
                'pnl_dollars': pnl_dollars,
                'exit_reason': exit_reason,
                'confidence': confidence
            })

            # Skip forward to avoid overlapping trades
            i += HOLD_BARS
        else:
            i += 1

    # Results
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    if not trades:
        print("❌ No trades generated!")
        return

    trades_df = pd.DataFrame(trades)

    # Statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_dollars'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_dollars'] < 0])
    breakeven_trades = len(trades_df[trades_df['pnl_dollars'] == 0])

    win_rate = winning_trades / total_trades * 100

    total_pnl = trades_df['pnl_dollars'].sum()
    avg_win = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars'].mean() if losing_trades > 0 else 0

    # Profit factor
    gross_profit = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl_dollars'].cumsum()
    trades_df['running_max'] = trades_df['cumulative_pnl'].cummax()
    trades_df['drawdown'] = trades_df['running_max'] - trades_df['cumulative_pnl']
    max_drawdown = trades_df['drawdown'].max()

    # Sharpe ratio (simplified)
    returns = trades_df['pnl_dollars']
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    print(f"\nTotal Trades: {total_trades}")
    print(f"  Wins:       {winning_trades} ({win_rate:.1f}%)")
    print(f"  Losses:     {losing_trades}")
    print(f"  Breakeven:  {breakeven_trades}")

    print(f"\nP&L:")
    print(f"  Total P&L:      ${total_pnl:,.2f}")
    print(f"  Avg Win:        ${avg_win:,.2f}")
    print(f"  Avg Loss:       ${avg_loss:,.2f}")
    print(f"  Profit Factor:  {profit_factor:.2f}")

    # Account performance
    final_balance = ACCOUNT_BALANCE + total_pnl
    account_return = (total_pnl / ACCOUNT_BALANCE) * 100

    print(f"\nAccount Performance:")
    print(f"  Starting Balance: ${ACCOUNT_BALANCE:,.2f}")
    print(f"  Final Balance:    ${final_balance:,.2f}")
    print(f"  Return:           {account_return:+.2f}%")

    print(f"\nConfidence:")
    print(f"  Avg Confidence: {trades_df['confidence'].mean():.1%}")
    print(f"  Min Confidence: {trades_df['confidence'].min():.1%}")
    print(f"  Max Confidence: {trades_df['confidence'].max():.1%}")

    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:   ${max_drawdown:,.2f}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")

    print(f"\nExit Reasons:")
    exit_counts = trades_df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")

    # Performance verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if total_pnl > 0:
        print(f"✅ PROFITABLE: ${total_pnl:,.2f}")
    else:
        print(f"❌ UNPROFITABLE: ${total_pnl:,.2f}")

    if win_rate >= 50 and profit_factor >= 1.5:
        print("✅ Good win rate and profit factor")
    elif win_rate >= 40:
        print("⚠️  Acceptable win rate but needs improvement")
    else:
        print("❌ Win rate too low")

    if sharpe > 1.5:
        print("✅ Excellent risk-adjusted returns")
    elif sharpe > 0.5:
        print("⚠️  Acceptable risk-adjusted returns")
    else:
        print("❌ Poor risk-adjusted returns")

    # Save results
    trades_df.to_csv("backtest_results.csv", index=False)
    print(f"\n📊 Detailed results saved to backtest_results.csv")

    return trades_df

if __name__ == "__main__":
    backtest_trading_strategy()
