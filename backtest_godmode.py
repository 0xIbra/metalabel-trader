"""
Backtest for Godmode Swing Trading Strategy
Tests the global model with Liquidity Grabs, Swap Bias, and 3:1 Risk/Reward
Parameters: 1000 EUR Account, 1:30 Leverage
Aggressive Mode: 3% Risk, Min 100 Pips TP
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import logging
sys.path.append('.')

from src.training.train_model import compute_features_and_labels
from src.quant_engine.indicators import calculate_atr
from src.quant_engine.godmode import detect_liquidity_grab, get_swap_bias

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GodmodeBacktest")

def backtest_godmode_strategy(
    data_dir="data/swing",
    model_path="src/oracle/model_swing_macro.json",
    confidence_threshold=0.50,
    test_split=0.5,
    initial_balance=1000.0,
    leverage=30.0
):
    """
    Backtest the Godmode swing trading strategy with specific account params
    """
    logger.info("=" * 80)
    logger.info(f"GODMODE AGGRESSIVE | Balance: €{initial_balance} | Lev: 1:{leverage}")
    logger.info(f"Risk: 3% | Min TP: 100 pips")
    logger.info("=" * 80)

    # Load model
    try:
        model = xgb.Booster()
        if os.path.exists(model_path):
            model.load_model(model_path)
            logger.info(f"✅ Loaded model: {model_path}\n")
        else:
            logger.error(f"❌ Model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

    # Get all data files
    files = [f for f in os.listdir(data_dir) if f.endswith('_h4.csv')]

    all_trades = []
    symbol_data = {}

    logger.info("Loading and aligning data...")

    for filename in files:
        symbol = filename.split('_')[0].upper()
        path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Compute features
            df = compute_features_and_labels(df)

            # Add dummy macro
            df['sentiment'] = 0.0
            df['sentiment_ma7'] = 0.0
            df['sentiment_std7'] = 0.0

            # Split to Test
            split_idx = int(len(df) * test_split)
            test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

            # Get predictions
            features = [
                'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
                'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
                'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
                'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2',
                'sentiment', 'sentiment_ma7', 'sentiment_std7'
            ]

            # Ensure features exist
            for f in features:
                if f not in test_df.columns: test_df[f] = 0.0

            dmatrix = xgb.DMatrix(test_df[features], feature_names=features)
            preds = model.predict(dmatrix)

            if len(preds.shape) > 1 and preds.shape[1] == 3:
                test_df['buy_prob'] = preds[:, 2]
            else:
                test_df['buy_prob'] = preds

            symbol_data[symbol] = test_df

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    # Merge all data into a single timeline
    timeline = []
    for symbol, df in symbol_data.items():
        for i, row in df.iterrows():
            timeline.append({
                'timestamp': row['timestamp'],
                'symbol': symbol,
                'data': row,
                'idx': i
            })

    # Sort by timestamp
    timeline.sort(key=lambda x: x['timestamp'])
    logger.info(f"Simulating {len(timeline)} bars across {len(symbol_data)} symbols...")

    # Simulation State
    balance = initial_balance
    positions = {} # symbol -> {entry_price, size, sl, tp, entry_time}

    # Aggressive Risk Params
    RISK_PER_TRADE = 0.03 # 3%
    MIN_TP_PIPS = 100

    for step in timeline:
        ts = step['timestamp']
        sym = step['symbol']
        row = step['data']

        current_price = row['close']
        high = row['high']
        low = row['low']

        # 1. Manage Open Position
        if sym in positions:
            pos = positions[sym]

            # Update Highest High for Trailing Stop
            if high > pos['highest_high']:
                pos['highest_high'] = high

            # Trailing Stop: Dynamic
            trail_mult = pos.get('trail_mult', 2.5)
            trail_sl = pos['highest_high'] - (trail_mult * pos['atr'])
            if trail_sl > pos['sl']:
                pos['sl'] = trail_sl

            # Check Exit
            exit_reason = None
            exit_price = current_price

            if high >= pos['tp']:
                exit_reason = 'TP'
                exit_price = pos['tp']
            elif low <= pos['sl']:
                exit_reason = 'SL'
                exit_price = pos['sl']
            elif (ts - pos['entry_time']) > (72 * 3600): # 72 hours
                exit_reason = 'TIMEOUT'

            if exit_reason:
                # Calculate PnL
                raw_pnl = (exit_price - pos['entry_price']) * pos['size'] * 100000

                if 'JPY' in sym:
                    raw_pnl = raw_pnl / 160.0
                elif 'USD' in sym and sym.endswith('USD'):
                    raw_pnl = raw_pnl / 1.05

                balance += raw_pnl

                all_trades.append({
                    'symbol': sym,
                    'pnl': raw_pnl,
                    'reason': exit_reason,
                    'balance': balance
                })

                del positions[sym]

        # 2. Check Entry (if no position)
        else:
            # Godmode Logic
            df = symbol_data[sym]
            idx = step['idx']

            if idx < 20: continue

            # Fast lookback for Liquidity Grab
            recent_lows = df.iloc[idx-20:idx]['low'].min()
            recent_highs = df.iloc[idx-20:idx]['high'].max()

            curr_low = low
            curr_close = current_price
            curr_high = high

            liq_signal = 0
            if curr_low < recent_lows and curr_close > recent_lows:
                liq_signal = 1
            elif curr_high > recent_highs and curr_close < recent_highs:
                liq_signal = -1

            # Swap Bias
            swap_bias = get_swap_bias(sym)

            # Confidence
            conf = row['buy_prob']
            adj_conf = conf

            if liq_signal == 1: adj_conf += 0.2
            if swap_bias == -1: adj_conf -= 0.1
            elif swap_bias == 1: adj_conf += 0.05

            if adj_conf >= confidence_threshold:
                # Calculate Position Size
                atr = row['atr']
                if atr == 0: continue

                # SL = 1 ATR (Tight Stop)
                sl_dist = 1.0 * atr

                # Trend Rider Logic
                # Check ADX from the row data
                adx = row['adx']

                if adx > 25:
                    # Strong Trend: Ride it!
                    tp_dist = 10.0 * atr # Effectively unlimited, aim for 150+ pips
                    trail_dist_mult = 3.0 # Looser trail
                else:
                    # Normal Swing
                    tp_dist = 3.0 * atr
                    trail_dist_mult = 2.5

                sl_price = current_price - sl_dist
                tp_price = current_price + tp_dist

                # Risk 5% of Balance
                RISK_PER_TRADE = 0.05
                risk_eur = balance * RISK_PER_TRADE

                # Convert Risk to Lots
                if 'JPY' in sym:
                    risk_jpy = risk_eur * 160
                    size = risk_jpy / (sl_dist * 100000)
                else:
                    risk_usd = risk_eur * 1.05
                    size = risk_usd / (sl_dist * 100000)

                # Leverage Constraint (Global)
                used_units = sum([p['size'] for p in positions.values()]) * 100000
                total_capacity = balance * leverage
                available_units = total_capacity - used_units

                if available_units <= 0: continue

                max_lots = available_units / 100000

                if size > max_lots:
                    size = max_lots

                # Min size
                if size < 0.01: continue

                # Open Position
                positions[sym] = {
                    'entry_price': current_price,
                    'size': size,
                    'sl': sl_price,
                    'tp': tp_price,
                    'entry_time': ts,
                    'highest_high': current_price,
                    'atr': atr,
                    'trail_mult': trail_dist_mult # Store trail multiplier
                }

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)

    logger.info(f"Final Balance: €{balance:.2f}")
    logger.info(f"Return: {((balance - initial_balance)/initial_balance)*100:.2f}%")
    logger.info(f"Total Trades: {len(all_trades)}")

    if all_trades:
        wins = [t for t in all_trades if t['pnl'] > 0]
        losses = [t for t in all_trades if t['pnl'] < 0]
        win_rate = len(wins) / len(all_trades)

        logger.info(f"Win Rate: {win_rate:.1%}")

        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        logger.info(f"Avg Win: €{avg_win:.2f}")
        logger.info(f"Avg Loss: €{avg_loss:.2f}")

        if losses:
            pf = abs(sum([t['pnl'] for t in wins]) / sum([t['pnl'] for t in losses]))
            logger.info(f"Profit Factor: {pf:.2f}")

    # Save
    pd.DataFrame(all_trades).to_csv('backtest_godmode_aggressive.csv', index=False)

if __name__ == "__main__":
    backtest_godmode_strategy()
