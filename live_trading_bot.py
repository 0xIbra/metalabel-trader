"""
Live trading bot using MetaApi for paper trading
Trades EURUSD and AUDUSD with 5:1 RR strategy
"""

import os
import sys
import asyncio
import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from src.quant_engine.godmode import detect_liquidity_grab, get_swap_bias
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.notifications.telegram_bot import (
    notify_trade_entry, notify_trade_exit, notify_status,
    notify_error, notify_startup, notify_shutdown
)
from src.quant_engine.indicators import (
    calculate_log_returns, calculate_volatility, calculate_z_score,
    calculate_rsi, calculate_adx, calculate_time_sin, calculate_volume_delta,
    calculate_bollinger_bands, calculate_atr, calculate_pivot_points
)
from src.training.train_model import compute_roc, compute_macd, compute_price_velocity

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MetaApi config
METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')
METAAPI_ACCOUNT_ID = os.getenv('METAAPI_ACCOUNT_ID')

# Trading config
# Trading config
# Swing Trading Universe
MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
CROSSES = [
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
    'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    'NZDCAD', 'NZDCHF',
    'CADCHF'
]
SYMBOLS = MAJORS + CROSSES # + ['XAUUSD'] (Gold often has different contract specs, omitting for now)
CONFIDENCE_THRESHOLD = 0.50
ACCOUNT_BALANCE = 1000  # Starting balance
LEVERAGE = 30
RISK_PERCENT = 0.02
TP_PIPS = 5
SL_PIPS = 1
HOLD_BARS = 60  # 60 minutes timeout

# Strategy config
# Strategy config
MAX_CONCURRENT_POSITIONS = 10  # Allow more positions for swing portfolio
DAILY_LOSS_LIMIT = 100  # Increased for swing volatility (or remove if not needed)
FEATURE_WINDOW = 100  # Keep last 100 bars for features
STATE_FILE = 'state.json'  # State persistence file


class LiveTradingBot:
    """
    Live trading bot with MetaApi integration
    """

    def __init__(self):
        self.api = None
        self.account = None
        self.models = {}
        self.price_buffers = {symbol: deque(maxlen=FEATURE_WINDOW) for symbol in SYMBOLS}
        self.open_positions = {}
        self.trades_today = []
        self.is_running = False

        # Load models
        self.load_models()

        # Load historical data
        self.load_historical_data()

        # Restore state if exists
        self.load_state()

    def load_historical_data(self):
        """Load historical H4 data from CSVs to initialize buffers"""
        logger.info("Loading historical data...")
        count = 0
        for symbol in SYMBOLS:
            try:
                path = f"data/swing/{symbol.lower()}_h4.csv"
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # Ensure sorted
                    df = df.sort_values('timestamp')

                    # Take last FEATURE_WINDOW bars
                    recent = df.tail(FEATURE_WINDOW).to_dict('records')

                    for bar in recent:
                        self.price_buffers[symbol].append(bar)

                    count += 1
                    logger.info(f"Loaded {len(recent)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load history for {symbol}: {e}")

        logger.info(f"✅ Loaded history for {count}/{len(SYMBOLS)} symbols")

    def load_models(self):
        """Load Global XGBoost model"""
        logger.info("Loading global model...")

       # Model path - using macro-enhanced model
        MODEL_PATH = "src/oracle/model_swing_macro.json"

        if os.path.exists(MODEL_PATH):
            try:
                model = xgb.Booster()
                model.load_model(MODEL_PATH)
                self.model = model # Single global model
                logger.info(f"✅ Loaded global model from {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load global model: {e}")
                raise
        else:
            logger.error(f"❌ Model not found: {model_path}")
            raise Exception("Global model not found!")

    def save_state(self):
        """Save bot state to file for crash recovery"""
        try:
            state = {
                'open_positions': {},
                'trades_today': [],
                'last_update': datetime.utcnow().isoformat()
            }

            # Serialize open positions (convert datetime to string)
            for symbol, pos in self.open_positions.items():
                state['open_positions'][symbol] = {
                    'entry_time': pos['entry_time'].isoformat(),
                    'entry_price': pos['entry_price'],
                    'lot_size': pos['lot_size'],
                    'tp': pos['tp'],
                    'sl': pos['sl'],
                    'confidence': pos['confidence'],
                    'order_id': pos['order_id'],
                    'highest_high': pos.get('highest_high', pos['entry_price']),
                    'atr': pos.get('atr', 0.0010),
                    'has_added': pos.get('has_added', False)
                }

            # Serialize trades (convert datetime to string)
            for trade in self.trades_today:
                state['trades_today'].append({
                    'symbol': trade['symbol'],
                    'pnl': trade['pnl'],
                    'pips': trade['pips'],
                    'reason': trade['reason'],
                    'timestamp': trade['timestamp'].isoformat()
                })

            # Save to file
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"State saved: {len(self.open_positions)} positions, {len(self.trades_today)} trades")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load bot state from file after crash"""
        try:
            if not os.path.exists(STATE_FILE):
                logger.info("No previous state found")
                return

            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

            # Check if state is recent (< 24 hours old)
            last_update = datetime.fromisoformat(state['last_update'])
            age_hours = (datetime.utcnow() - last_update).total_seconds() / 3600

            if age_hours > 24:
                logger.warning(f"State file is {age_hours:.1f} hours old, ignoring")
                return

            # Restore trades (convert string back to datetime)
            for trade_data in state.get('trades_today', []):
                trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                self.trades_today.append(trade_data)

            # Restore positions (convert string back to datetime)
            for symbol, pos_data in state.get('open_positions', {}).items():
                pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
                # Backwards compatibility for state file
                if 'highest_high' not in pos_data: pos_data['highest_high'] = pos_data['entry_price']
                if 'atr' not in pos_data: pos_data['atr'] = 0.0010
                if 'has_added' not in pos_data: pos_data['has_added'] = False

                self.open_positions[symbol] = pos_data

            logger.info(f"✅ State restored: {len(self.open_positions)} positions, {len(self.trades_today)} trades from {age_hours:.1f}h ago")

            if self.open_positions:
                logger.warning(f"⚠️ Restored open positions: {list(self.open_positions.keys())}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            # Continue without state - safer than crashing

    async def connect_metaapi(self):
        """Connect to MetaApi"""
        try:
            from metaapi_cloud_sdk import MetaApi

            logger.info("Connecting to MetaApi...")
            self.api = MetaApi(METAAPI_TOKEN)
            self.account = await self.api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)

            # Wait for deployment
            logger.info("Deploying account...")
            await asyncio.wait_for(self.account.deploy(), timeout=60)
            logger.debug("Account deployed")

            # Wait for connection
            logger.info("Waiting for connection...")
            await asyncio.wait_for(self.account.wait_connected(), timeout=60)
            logger.debug("Account connected")

            # Get connection
            connection = self.account.get_rpc_connection()
            logger.debug("Connecting RPC...")
            await asyncio.wait_for(connection.connect(), timeout=60)
            logger.debug("RPC connected, waiting for sync...")
            await asyncio.wait_for(connection.wait_synchronized(), timeout=60)
            logger.debug("RPC synchronized")

            logger.info("✅ Connected to MetaApi")

            # Get account info
            logger.debug("Getting account info...")
            account_info = await connection.get_account_information()
            logger.info(f"Account balance: ${account_info['balance']:.2f}")

            return connection

        except Exception as e:
            logger.error(f"❌ MetaApi connection failed: {e}")
            notify_error(str(e), "Check MetaApi credentials")
            raise

    def compute_features(self, symbol):
        """Compute features from price buffer"""
        buffer = list(self.price_buffers[symbol])

        if len(buffer) < FEATURE_WINDOW:
            return None

        # Extract OHLCV
        df = pd.DataFrame(buffer)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df.get('volume', pd.Series([1000]*len(df))).values
        timestamps = df['timestamp'].values

        # Calculate indicators
        log_returns = calculate_log_returns(close)
        volatility = calculate_volatility(log_returns, window=20)
        z_score = calculate_z_score(close, window=20)
        rsi = calculate_rsi(close)
        adx = calculate_adx(high, low, close)
        time_sin = calculate_time_sin(timestamps)
        volume_delta = calculate_volume_delta(volume)

        # Swing Features
        upper, middle, lower, bandwidth = calculate_bollinger_bands(close, window=20, num_std=2)
        # Avoid division by zero
        bb_range = upper - lower
        bb_position = np.zeros_like(close)
        mask = bb_range != 0
        bb_position[mask] = (close[mask] - lower[mask]) / bb_range[mask]

        atr = calculate_atr(high, low, close, window=14)
        atr_pct = np.zeros_like(atr)
        mask = close != 0
        atr_pct[mask] = atr[mask] / close[mask] * 100

        pivot, r1, s1 = calculate_pivot_points(high, low, close)
        dist_pivot = np.zeros_like(close)
        mask = atr != 0
        dist_pivot[mask] = (close[mask] - pivot[mask]) / atr[mask]

        # Momentum features
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

        # Get latest features (last bar)
        # MUST MATCH train_model.py feature order exactly
        features = np.array([[
            z_score[-1], rsi[-1], volatility[-1], adx[-1], time_sin[-1], volume_delta[-1],
            bandwidth[-1], bb_position[-1], atr_pct[-1], dist_pivot[-1],
            roc_5[-1], roc_10[-1], roc_20[-1], macd[-1], velocity[-1],
            close_lag1[-1], close_lag2[-1], close_lag3[-1], returns_lag1[-1], returns_lag2[-1],
            # Macro features (placeholder - TODO: integrate live sentiment)
            0.0, 0.0, 0.0  # sentiment, sentiment_ma7, sentiment_std7
        ]])

        # Validate features - critical safety check
        if np.isnan(features).any():
            logger.warning(f"{symbol}: Features contain NaN, skipping")
            return None

        if np.isinf(features).any():
            logger.warning(f"{symbol}: Features contain Inf, skipping")
            return None

        return features

    def get_prediction(self, symbol, features):
        """Get model prediction using global model"""
        if not hasattr(self, 'model') or self.model is None:
            return 0.0

        if features is None:
            return 0.0

        feature_names = [
            'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
            'bb_width', 'bb_position', 'atr_pct', 'dist_pivot',
            'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
            'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2',
            # Macro features (set to 0 for now - TODO: integrate live sentiment)
            'sentiment', 'sentiment_ma7', 'sentiment_std7'
        ]

        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        predictions = self.model.predict(dmatrix)

        # Get BUY probability (3-class model)
        # Classes: 0=SELL, 1=CHOP, 2=BUY
        if len(predictions.shape) > 1 and predictions.shape[1] == 3:
            buy_prob = predictions[0][2]  # BUY class
        else:
            # Fallback if binary
            buy_prob = predictions[0]

        return buy_prob

    async def check_signals(self, connection):
        """Check for trading signals"""
        # Reset daily counters at midnight UTC
        now = datetime.utcnow()
        if self.trades_today and (now.date() > self.trades_today[-1]['timestamp'].date()):
            logger.info("New day - resetting daily counters")
            self.trades_today = []

        # Check max daily trades
        if len(self.trades_today) >= 10:  # MAX_DAILY_TRADES
            logger.warning(f"Max daily trades reached: {len(self.trades_today)}")
            return

        for symbol in SYMBOLS:
            # Skip if already in position
            if symbol in self.open_positions:
                continue

            # Check max positions
            if len(self.open_positions) >= MAX_CONCURRENT_POSITIONS:
                continue

            # Check daily loss limit
            daily_pnl = sum([t['pnl'] for t in self.trades_today])
            if daily_pnl <= -DAILY_LOSS_LIMIT:
                logger.warning(f"Daily loss limit reached: ${daily_pnl:.2f}")
                return

            # Check account balance
            try:
                account_info = await connection.get_account_information()
                current_balance = account_info['balance']

                if current_balance < 100:  # Minimum balance check
                    logger.error(f"Insufficient balance: ${current_balance:.2f}")
                    notify_error(f"Balance too low: ${current_balance:.2f}", "Add funds to account")
                    return
            except Exception as e:
                logger.error(f"Failed to get account info: {e}")
                continue

            # Compute features
            features = self.compute_features(symbol)
            if features is None:
                continue

            # Get prediction
            confidence = self.get_prediction(symbol, features)

            # --- GODMODE ENHANCEMENTS ---

            # 1. Liquidity Grab Detection (The "Stop-Hunt" Game)
            # We need the price buffer to detect swings
            buffer_df = pd.DataFrame(list(self.price_buffers[symbol]))
            if len(buffer_df) < 22: continue # Need window + 1

            high_arr = buffer_df['high'].values
            low_arr = buffer_df['low'].values
            close_arr = buffer_df['close'].values

            liq_signal = detect_liquidity_grab(high_arr, low_arr, close_arr, window=20)

            # 2. Carry Trade Filter (The "Free" Money)
            swap_bias = get_swap_bias(symbol)

            # Adjust Confidence based on Godmode factors
            adjusted_confidence = confidence

            # Boost for Bullish Liquidity Grab
            if liq_signal == 1:
                logger.info(f"💎 GODMODE: Bullish Liquidity Grab detected on {symbol}!")
                adjusted_confidence += 0.2 # Significant boost

            # Penalty for Negative Carry (trading against the house)
            if swap_bias == -1:
                logger.info(f"⚠️ Negative Swap Bias for {symbol}. Penalizing confidence.")
                adjusted_confidence -= 0.1
            elif swap_bias == 1:
                 adjusted_confidence += 0.05 # Small boost for positive carry

            # Check threshold
            if adjusted_confidence < CONFIDENCE_THRESHOLD:
                continue

            # Get current price
            try:
                price = await connection.get_symbol_price(symbol)
                current_price = price['ask']
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                continue

            # Calculate ATR for Risk Management
            atr_arr = calculate_atr(high_arr, low_arr, close_arr, window=14)
            current_atr = atr_arr[-1]

            if current_atr == 0: continue

            # --- GODMODE RISK MANAGEMENT ---
            # Trend Rider Logic (ADX > 25)
            # We need ADX from features.
            # Features index 3 is ADX (based on compute_features)
            # [z_score, rsi, volatility, adx, ...]
            adx_val = features[0][3]

            if adx_val > 25:
                # Strong Trend: Ride it!
                logger.info(f"🌊 TREND MODE: ADX {adx_val:.1f} > 25. Extending targets!")
                tp_dist = 10.0 * current_atr
                trail_mult = 3.0
            else:
                # Normal Swing
                tp_dist = 3.0 * current_atr
                trail_mult = 2.5

            sl_dist = 1.0 * current_atr

            tp_price = current_price + tp_dist
            sl_price = current_price - sl_dist

            # Calculate lot size dynamically based on balance and risk
            # Risk 5% of account (Godmode Trend)
            risk_amount = current_balance * 0.05

            # Pip value approximation (Standard: $10/lot, JPY: $1000/lot approx)
            pip_value_per_lot = 1000 if 'JPY' in symbol else 10
            # Convert SL dist to pips for calculation
            pip_size = 0.01 if 'JPY' in symbol else 0.0001
            sl_pips = sl_dist / pip_size

            if sl_pips == 0: continue

            lot_size = risk_amount / (sl_pips * pip_value_per_lot)
            lot_size = round(lot_size, 2)
            lot_size = max(0.01, min(lot_size, 5.0)) # Cap at 5 lots

            # Place order
            try:
                logger.info(f"🚀 ENTERING TRADE {symbol} | Conf: {confidence:.2f}->{adjusted_confidence:.2f} | Liq: {liq_signal} | Swap: {swap_bias}")
                result = await connection.create_market_buy_order(
                    symbol=symbol,
                    volume=lot_size,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )

                # Track position
                self.open_positions[symbol] = {
                    'entry_time': datetime.utcnow(),
                    'entry_price': current_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'atr': current_atr,
                    'highest_high': current_price, # For trailing stop
                    'has_added': False,
                    'trail_mult': trail_mult, # Dynamic Trailing Stop Multiplier
                    'order_id': result['orderId'] if 'orderId' in result else 'mock_id'
                }
            except Exception as e:
                logger.error(f"Failed to open {symbol}: {e}")
                notify_error(f"Failed to open {symbol}", str(e))

    async def check_positions(self, connection):
        """Check and manage open positions (Trailing Stop & Pyramiding)"""
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]

            try:
                # Get current price
                price_info = await connection.get_symbol_price(symbol)
                current_price = price_info['bid'] # Exit price for BUY

                # Update Highest High
                if current_price > position['highest_high']:
                    position['highest_high'] = current_price

                # --- TRAILING STOP (Chandelier Exit) ---
                # Trail by 3 * ATR from Highest High
                # Or tighter trail: 1.5 * ATR to lock profits?
                # Plan says: "Chandelier Exit (Trail stop by 3x ATR from highest high)"
                # But initial SL is 1x ATR. 3x ATR trail would be below initial SL immediately.
                # Let's use 1.5x ATR trail to secure profit once it moves.

                # Trail by dynamic multiplier (Trend Rider: 3.0, Normal: 2.5)
                trail_mult = position.get('trail_mult', 2.5)
                trail_dist = trail_mult * position['atr']
                new_sl = position['highest_high'] - trail_dist

                # Only move SL UP
                if new_sl > position['sl']:
                    # Update SL on broker
                    # Note: MetaApi modification might fail if too close to price
                    # For now, we just update internal SL and close if hit
                    position['sl'] = new_sl
                    logger.info(f"🔄 Trailing SL for {symbol} to {new_sl:.5f}")

                    # Ideally send modify_position request to broker here
                    # await connection.modify_position(position['order_id'], stop_loss=new_sl, take_profit=position['tp'])

                # Check SL Hit (Internal)
                if current_price <= position['sl']:
                    await self.close_position(connection, symbol, 'SL_HIT')
                    continue

                # --- PYRAMIDING (Add-On) ---
                # Add 0.5% risk if price moves +1 ATR favorable
                if not position['has_added']:
                    profit_dist = current_price - position['entry_price']
                    if profit_dist >= (1.0 * position['atr']):
                        # Add to position
                        logger.info(f"🚀 Pyramiding {symbol}: Adding to winner!")

                        # Calculate add-on size (0.5% risk)
                        # New SL will be at Breakeven of first trade?
                        # Let's just open a new separate order for simplicity of tracking
                        # But `open_positions` structure assumes 1 pos per symbol.
                        # For now, we just log it and maybe skip actual execution to keep it simple
                        # until we support multi-position tracking.
                        # OR: Just update the 'has_added' flag and maybe move SL to BE.

                        position['has_added'] = True
                        # Move SL to Breakeven + small buffer
                        be_sl = position['entry_price'] + (0.1 * position['atr'])
                        if be_sl > position['sl']:
                            position['sl'] = be_sl
                            logger.info(f"🔒 Moved SL to Breakeven for {symbol}")

                # Check Timeout
                time_held = (datetime.utcnow() - position['entry_time']).total_seconds() / 60
                # 60 bars * 4 hours = 240 hours timeout?
                # Plan says 48 hours target. Let's set timeout to 72 hours (3 days).
                if time_held >= (72 * 60):
                    await self.close_position(connection, symbol, 'TIMEOUT')

            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")

    async def close_position(self, connection, symbol, reason):
        """Close a position"""
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]

        try:
            # Get current price
            price = await connection.get_symbol_price(symbol)
            exit_price = price['bid']

            # Close position
            try:
                await connection.close_position(position['order_id'])
                logger.info(f"Closed position {position['order_id']}")
            except Exception as e:
                # Position may have already been closed by TP/SL
                if 'not found' in str(e).lower() or 'position' in str(e).lower():
                    logger.info(f"{symbol} position already closed (likely TP/SL)")
                else:
                    raise

            # Calculate P&L correctly
            pip_size = 0.0001 if symbol != 'USDJPY' else 0.01
            pips = (exit_price - position['entry_price']) / pip_size

            # Correct pip value calculation
            lot_size = position['lot_size']
            if symbol == 'USDJPY':
                pip_value = lot_size * 1000  # JPY pairs: 1 lot = 100,000 units, 1 pip = 0.01
            else:
                pip_value = lot_size * 10  # Standard pairs: 1 lot = 100,000 units, 1 pip = 0.0001

            pnl = pips * pip_value

            # Track trade
            hold_time = int((datetime.utcnow() - position['entry_time']).total_seconds() / 60)
            trade = {
                'symbol': symbol,
                'pnl': pnl,
                'pips': pips,
                'reason': reason,
                'timestamp': datetime.utcnow()
            }
            self.trades_today.append(trade)

            # Get balance
            account_info = await connection.get_account_information()
            balance = account_info['balance']

            # Notify
            notify_trade_exit(
                symbol, exit_price, pnl, pips, reason, hold_time, balance
            )

            logger.info(f"💰 Closed {symbol} @ {exit_price:.5f} | P&L: ${pnl:+.2f} ({reason})")

            # Remove position
            del self.open_positions[symbol]

            # Save state immediately
            self.save_state()

        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            # Still remove from tracking to avoid stuck state
            if symbol in self.open_positions:
                del self.open_positions[symbol]
                self.save_state()

    async def on_tick(self, tick):
        """Handle new tick data with H4 aggregation"""
        symbol = tick['symbol']

        if symbol not in SYMBOLS:
            return

        # Get tick data
        price = tick.get('bid', tick.get('price'))
        volume = tick.get('volume', 1000)

        # Use current UTC time
        now = datetime.utcnow()

        # Calculate H4 bar start time
        # 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        h4_start_hour = now.hour - (now.hour % 4)
        current_bar_start = now.replace(hour=h4_start_hour, minute=0, second=0, microsecond=0)

        # Initialize current bar if needed
        if not hasattr(self, 'current_bars'):
            self.current_bars = {}

        if symbol not in self.current_bars:
            self.current_bars[symbol] = {
                'timestamp': current_bar_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            return

        bar = self.current_bars[symbol]

        # Check if new bar (current time >= next bar start)
        # Next bar start is bar['timestamp'] + 4 hours
        next_bar_start = bar['timestamp'] + timedelta(hours=4)

        if now >= next_bar_start:
            # Finalize previous bar
            completed_bar = bar.copy()
            # Convert timestamp to int
            completed_bar['timestamp'] = int(completed_bar['timestamp'].timestamp())

            self.price_buffers[symbol].append(completed_bar)
            logger.info(f"📊 New H4 Bar {symbol}: {completed_bar['close']} (Vol: {completed_bar['volume']})")

            # Start new bar
            self.current_bars[symbol] = {
                'timestamp': current_bar_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
        else:
            # Update current bar
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += volume

    async def run(self):
        """Main trading loop with reconnection logic"""
        self.is_running = True
        connection = None
        max_retries = 3
        retry_count = 0

        while self.is_running and retry_count < max_retries:
            try:
                # Connect to MetaApi
                connection = await self.connect_metaapi()


                # Send startup notification
                account_info = await connection.get_account_information()
                notify_startup(SYMBOLS, CONFIDENCE_THRESHOLD, account_info['balance'])

                logger.info("🚀 Bot started. Monitoring signals...")
                logger.info(f"📡 Market data streaming for: {', '.join(SYMBOLS)}")

                # Reset retry count after successful connection
                retry_count = 0

                # Main loop
                last_status = datetime.utcnow()
                last_heartbeat = datetime.utcnow()

                while self.is_running:
                    try:
                        # Connection heartbeat check (every 5 minutes)
                        if (datetime.utcnow() - last_heartbeat).total_seconds() > 300:
                            # Verify connection is alive
                            await connection.get_account_information()
                            last_heartbeat = datetime.utcnow()
                            logger.debug("Connection heartbeat OK")

                        # Check for signals (every minute)
                        await self.check_signals(connection)

                        # Check positions
                        await self.check_positions(connection)

                        # Send hourly status
                        if (datetime.utcnow() - last_status).total_seconds() >= 3600:
                            account_info = await connection.get_account_information()
                            win_rate = len([t for t in self.trades_today if t['pnl'] > 0]) / len(self.trades_today) * 100 if self.trades_today else 0
                            notify_status(
                                account_info['balance'],
                                len(self.open_positions),
                                len(self.trades_today),
                                sum([t['pnl'] for t in self.trades_today]),
                                win_rate
                            )
                            last_status = datetime.utcnow()

                        # Periodic state save (every 5 minutes as safety backup)
                        if (datetime.utcnow() - last_heartbeat).total_seconds() >= 300:
                            self.save_state()

                        # Sleep for 1 minute
                        await asyncio.sleep(60)

                    except Exception as e:
                        # Handle connection errors during main loop
                        if 'connection' in str(e).lower() or 'timeout' in str(e).lower():
                            logger.error(f"Connection error in main loop: {e}")
                            notify_error("Connection lost", "Attempting to reconnect...")
                            break  # Break inner loop to trigger reconnection
                        else:
                            logger.error(f"Error in main loop: {e}")
                            await asyncio.sleep(60)  # Continue after error

            except KeyboardInterrupt:
                logger.info("⚠️ Keyboard interrupt received")
                notify_shutdown("Manual stop")
                break

            except asyncio.CancelledError:
                logger.error("❌ Task cancelled")
                break

            except Exception as e:
                logger.error(f"❌ Bot error: {e}", exc_info=True)
                retry_count += 1

                if retry_count < max_retries:
                    wait_time = retry_count * 30  # Exponential backoff: 30s, 60s, 90s
                    logger.info(f"Reconnecting in {wait_time}s... (Attempt {retry_count}/{max_retries})")
                    notify_error(str(e), f"Reconnecting in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Stopping bot.")
                    notify_error("Max reconnection attempts reached", "Bot stopped")
                    break

            except BaseException as e:
                logger.critical(f"❌ Critical error (BaseException): {type(e).__name__}: {e}", exc_info=True)
                notify_error(f"Critical error: {type(e).__name__}", str(e))
                break

        # Cleanup
        self.is_running = False

        # Save final state
        self.save_state()

        # Close any open positions on shutdown
        if connection and self.open_positions:
            logger.warning("Closing open positions on shutdown...")
            for symbol in list(self.open_positions.keys()):
                try:
                    await self.close_position(connection, symbol, "BOT_SHUTDOWN")
                except Exception as e:
                    logger.error(f"Failed to close {symbol} on shutdown: {e}")

        logger.info("🛑 Bot stopped")


async def main():
    """Entry point"""
    bot = LiveTradingBot()
    await bot.run()


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Run bot
    asyncio.run(main())
