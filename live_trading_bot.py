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

# Custom Logging Formatter
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "[%(asctime)s] %(levelname)-8s %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

# Setup logging
logger = logging.getLogger("GodmodeBot")
logger.setLevel(logging.INFO)

# Console Handler with Custom Formatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# File Handler (Plain text)
fh = logging.FileHandler('logs/live_trading.log')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# MetaApi config
METAAPI_TOKEN = os.getenv('METAAPI_TOKEN')
METAAPI_ACCOUNT_ID = os.getenv('METAAPI_ACCOUNT_ID')

# Trading config
# Trading config
# Elite Symbol Universe (High WR or High Volume - Removed poor performers)
# Based on OOS backtest analysis: +48,742% vs +25,088% with full 28-pair universe
MAJORS = ['EURUSD', 'USDJPY', 'AUDUSD']
ELITE_CROSSES = [
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CHFJPY',  # JPY Crosses
    'EURAUD', 'EURNZD', 'EURCHF', 'GBPNZD', 'AUDNZD', 'GBPCAD', 'NZDCAD', 'NZDCHF', 'USDCAD'  # Other Crosses
]
SYMBOLS = MAJORS + ELITE_CROSSES
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

    def log_trade_entry(self, symbol, side, price, size, tp, sl, conf, liq, swap):
        """Log trade entry in a structured format"""
        sl_pips = abs(price - sl) * (100 if 'JPY' in symbol else 10000)
        tp_pips = abs(tp - price) * (100 if 'JPY' in symbol else 10000)

        msg = (
            f"\n🚀 ENTRY: {symbol} ({side})\n"
            f"--------------------------------\n"
            f"Price: {price:.5f} | Size: {size}\n"
            f"TP:    {tp:.5f} (+{tp_pips:.1f} pips)\n"
            f"SL:    {sl:.5f} (-{sl_pips:.1f} pips)\n"
            f"Conf:  {conf:.2f} (Liq: {liq}, Swap: {swap})\n"
            f"--------------------------------"
        )
        logger.info(msg)
        notify_trade_entry(symbol, side, price, sl, tp)

    def log_trade_exit(self, symbol, side, entry, exit_price, pnl, reason, time_held):
        """Log trade exit in a structured format"""
        pips = (exit_price - entry) * (100 if 'JPY' in symbol else 10000)
        if side == 'SELL': pips = -pips

        icon = "💰" if pnl > 0 else "🛑"

        msg = (
            f"\n{icon} EXIT: {symbol} ({reason})\n"
            f"--------------------------------\n"
            f"PnL:   €{pnl:.2f} ({pips:+.1f} pips)\n"
            f"Time:  {int(time_held)}m\n"
            f"--------------------------------"
        )
        logger.info(msg)
        notify_trade_exit(symbol, pnl, reason)

    def log_dashboard(self, balance):
        """Log a concise dashboard summary"""
        open_count = len(self.open_positions)
        pnl_today = sum([t['pnl'] for t in self.trades_today])

        # Only print if there's activity or every 5 mins
        if open_count > 0 or pnl_today != 0:
            logger.info(f"📊 STATUS | Bal: €{balance:.2f} | Open: {open_count} | Today: €{pnl_today:+.2f}")
        else:
            # Minimal heartbeat
            pass

    def is_market_open(self):
        """Check if Forex market is open (Sunday 22:00 UTC to Friday 22:00 UTC)"""
        now = datetime.utcnow()
        weekday = now.weekday() # Mon=0, Sun=6
        hour = now.hour

        # Saturday (5) - Closed
        if weekday == 5:
            return False

        # Friday (4) - Close at 22:00 UTC
        if weekday == 4 and hour >= 22:
            return False

        # Sunday (6) - Open at 22:00 UTC
        if weekday == 6 and hour < 22:
            return False

        return True

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

       # Model path - using strict OOS validated model
        MODEL_PATH = "src/oracle/model_strict_2024.json"

        if os.path.exists(MODEL_PATH):
            try:
                model = xgb.Booster()
                model.load_model(MODEL_PATH)
                self.model = model # Single global model
                logger.info(f"✅ Loaded global model from {MODEL_PATH}")
            except Exception as e:
                logger.error(f"❌ Failed to load global model: {e}")
                raise
        else:
            logger.error(f"❌ Model not found: {MODEL_PATH}")
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

            logger.error(f"❌ MetaApi connection failed: {e}")
            notify_error(str(e), "Check MetaApi credentials")
            raise

    def calculate_live_currency_strength(self):
        """
        Calculate live currency strength from price buffers.
        Returns dict: {'USD': 0.5, 'EUR': -0.2, ...}
        """
        currency_roc = {} # {'USD': [roc1, roc2], ...}

        for symbol, buffer in self.price_buffers.items():
            if len(buffer) < 6: continue # Need at least 6 bars for ROC-5

            # Get latest close and close 5 bars ago
            current_close = buffer[-1]['close']
            prev_close = buffer[-6]['close']

            if prev_close == 0: continue

            # ROC-5
            roc = ((current_close - prev_close) / prev_close) * 100

            base = symbol[:3]
            quote = symbol[3:]

            if base not in currency_roc: currency_roc[base] = []
            if quote not in currency_roc: currency_roc[quote] = []

            currency_roc[base].append(roc)
            currency_roc[quote].append(-roc) # Inverse for quote

        # Average
        strength_map = {}
        for currency, rocs in currency_roc.items():
            strength_map[currency] = sum(rocs) / len(rocs)

        return strength_map

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

        # Currency Strength Feature
        strength_map = self.calculate_live_currency_strength()
        base = symbol[:3]
        quote = symbol[3:]

        base_s = strength_map.get(base, 0.0)
        quote_s = strength_map.get(quote, 0.0)
        strength_diff = base_s - quote_s

        # Get latest features (last bar)
        # MUST MATCH train_model.py feature order exactly
        features = np.array([[
            z_score[-1], rsi[-1], volatility[-1], adx[-1], time_sin[-1], volume_delta[-1],
            bandwidth[-1], bb_position[-1], atr_pct[-1], dist_pivot[-1],
            roc_5[-1], roc_10[-1], roc_20[-1], macd[-1], velocity[-1],
            close_lag1[-1], close_lag2[-1], close_lag3[-1], returns_lag1[-1], returns_lag2[-1],
            strength_diff # New Feature
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
            'strength_diff' # New Feature
        ]

        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        predictions = self.model.predict(dmatrix)

        # Get probabilities (3-class model: 0=SELL, 1=CHOP, 2=BUY)
        if len(predictions.shape) > 1 and predictions.shape[1] == 3:
            sell_prob = float(predictions[0][0])
            buy_prob = float(predictions[0][2])
        else:
            # Fallback if binary
            buy_prob = float(predictions[0])
            sell_prob = 0.0

        return {'buy': buy_prob, 'sell': sell_prob}

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

        # Check account balance ONCE before looping symbols
        try:
            account_info = await connection.get_account_information()
            current_balance = account_info['balance']

            if current_balance < 100:  # Minimum balance check
                logger.error(f"Insufficient balance: ${current_balance:.2f}")
                notify_error(f"Balance too low: ${current_balance:.2f}", "Add funds to account")
                return
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
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

            # Compute features
            features = self.compute_features(symbol)
            if features is None:
                continue

            # Get prediction
            probs = self.get_prediction(symbol, features)
            buy_conf = probs['buy']
            sell_conf = probs['sell']

            # --- GODMODE ENHANCEMENTS ---

            # 1. Liquidity Grab Detection (The "Stop-Hunt" Game)
            buffer_df = pd.DataFrame(list(self.price_buffers[symbol]))
            if len(buffer_df) < 22: continue

            high_arr = buffer_df['high'].values
            low_arr = buffer_df['low'].values
            close_arr = buffer_df['close'].values

            liq_signal = detect_liquidity_grab(high_arr, low_arr, close_arr, window=20)

            # 2. Carry Trade Filter (The "Free" Money)
            swap_bias = get_swap_bias(symbol)

            # Adjust Confidence based on Godmode factors

            # --- BUY CONFIDENCE ---
            adj_buy_conf = buy_conf
            if liq_signal == 1: adj_buy_conf += 0.2 # Bullish Grab
            if swap_bias == -1: adj_buy_conf -= 0.1 # Penalty for Short Bias
            elif swap_bias == 1: adj_buy_conf += 0.05 # Boost for Long Bias

            # --- SELL CONFIDENCE ---
            adj_sell_conf = sell_conf
            if liq_signal == -1: adj_sell_conf += 0.2 # Bearish Grab
            if swap_bias == -1: adj_sell_conf += 0.05 # Boost for Short Bias
            elif swap_bias == 1: adj_sell_conf -= 0.1 # Penalty for Long Bias

            # Decision
            entry_signal = None
            final_conf = 0.0

            if adj_buy_conf >= CONFIDENCE_THRESHOLD:
                entry_signal = 'BUY'
                final_conf = adj_buy_conf
            elif adj_sell_conf >= CONFIDENCE_THRESHOLD:
                entry_signal = 'SELL'
                final_conf = adj_sell_conf

            # If both (rare), pick higher confidence
            if adj_buy_conf >= CONFIDENCE_THRESHOLD and adj_sell_conf >= CONFIDENCE_THRESHOLD:
                if adj_buy_conf > adj_sell_conf:
                    entry_signal = 'BUY'
                    final_conf = adj_buy_conf
                else:
                    entry_signal = 'SELL'
                    final_conf = adj_sell_conf

            if not entry_signal:
                continue

            # Get current price
            try:
                price = await connection.get_symbol_price(symbol)
                # Ask for BUY, Bid for SELL
                current_price = price['ask'] if entry_signal == 'BUY' else price['bid']
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                continue

            # Calculate ATR for Risk Management
            atr_arr = calculate_atr(high_arr, low_arr, close_arr, window=14)
            current_atr = atr_arr[-1]

            if current_atr == 0: continue

            # --- GODMODE RISK MANAGEMENT ---
            # Trend Rider Logic (ADX > 25)
            adx_val = features[0][3]

            if adx_val > 25:
                # Strong Trend: Ride it!
                logger.info(f"🌊 TREND MODE: ADX {adx_val:.1f} > 25. Extending targets!")
                tp_mult = 10.0
                trail_mult = 3.0
            else:
                # Normal Swing
                tp_mult = 3.0
                trail_mult = 2.5

            sl_dist = 1.0 * current_atr
            tp_dist = tp_mult * current_atr

            if entry_signal == 'BUY':
                tp_price = current_price + tp_dist
                sl_price = current_price - sl_dist
            else: # SELL
                tp_price = current_price - tp_dist
                sl_price = current_price + sl_dist

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
                # Execute Order
                if entry_signal == 'BUY':
                    result = await connection.create_market_buy_order(
                        symbol=symbol,
                        volume=lot_size,
                        stop_loss=sl_price,
                        take_profit=tp_price
                    )
                else: # SELL
                    result = await connection.create_market_sell_order(
                        symbol=symbol,
                        volume=lot_size,
                        stop_loss=sl_price,
                        take_profit=tp_price
                    )

                # Log & Notify
                self.log_trade_entry(
                    symbol, entry_signal, current_price, lot_size,
                    tp_price, sl_price, final_conf, liq_signal, swap_bias
                )

                # Track position
                self.open_positions[symbol] = {
                    'side': entry_signal,
                    'entry_time': datetime.utcnow(),
                    'entry_price': current_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'atr': current_atr,
                    'highest_high': current_price if entry_signal == 'BUY' else 0,
                    'lowest_low': current_price if entry_signal == 'SELL' else 999999,
                    'has_added': False,
                    'trail_mult': trail_mult, # Dynamic Trailing Stop Multiplier
                    'order_id': result['orderId'] if 'orderId' in result else 'mock_id',
                    'lot_size': lot_size
                }
            except Exception as e:
                logger.error(f"Failed to open {symbol}: {e}")
                notify_error(f"Failed to open {symbol}", str(e))

            # Yield control to event loop to allow WebSocket heartbeats
            await asyncio.sleep(0)

    async def check_positions(self, connection):
        """Check and manage open positions (Trailing Stop & Pyramiding)"""
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]
            side = position.get('side', 'BUY')

            try:
                # Get current price
                price_info = await connection.get_symbol_price(symbol)
                # Exit price: Bid for BUY, Ask for SELL
                current_price = price_info['bid'] if side == 'BUY' else price_info['ask']

                # --- BUY MANAGEMENT ---
                if side == 'BUY':
                    # Update Highest High
                    if current_price > position['highest_high']:
                        position['highest_high'] = current_price

                    # Trailing Stop
                    trail_mult = position.get('trail_mult', 2.5)
                    trail_dist = trail_mult * position['atr']
                    new_sl = position['highest_high'] - trail_dist

                    # Only move SL UP
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                        logger.info(f"🔄 TRAIL: {symbol} (BUY) SL -> {new_sl:.5f}")

                    # Check SL Hit
                    if current_price <= position['sl']:
                        await self.close_position(connection, symbol, 'SL_HIT')
                        continue

                # --- SELL MANAGEMENT ---
                else: # SELL
                    # Update Lowest Low
                    if current_price < position['lowest_low']:
                        position['lowest_low'] = current_price

                    # Trailing Stop (Above price)
                    trail_mult = position.get('trail_mult', 2.5)
                    trail_dist = trail_mult * position['atr']
                    new_sl = position['lowest_low'] + trail_dist

                    # Only move SL DOWN
                    if new_sl < position['sl']:
                        position['sl'] = new_sl
                        logger.info(f"🔄 TRAIL: {symbol} (SELL) SL -> {new_sl:.5f}")

                    # Check SL Hit
                    if current_price >= position['sl']:
                        await self.close_position(connection, symbol, 'SL_HIT')
                        continue

                # --- PYRAMIDING (Add-On) ---
                if not position['has_added']:
                    if side == 'BUY':
                        profit_dist = current_price - position['entry_price']
                    else:
                        profit_dist = position['entry_price'] - current_price

                    if profit_dist >= (1.0 * position['atr']):
                        # Add to position (Simulated for now)
                        logger.info(f"🚀 PYRAMID: {symbol} (+1 ATR Profit) - Adding size!")
                        position['has_added'] = True

                        # Move SL to Breakeven
                        if side == 'BUY':
                            be_sl = position['entry_price'] + (0.1 * position['atr'])
                            if be_sl > position['sl']:
                                position['sl'] = be_sl
                                logger.info(f"🔒 BREAKEVEN: {symbol} SL -> {be_sl:.5f}")
                        else:
                            be_sl = position['entry_price'] - (0.1 * position['atr'])
                            if be_sl < position['sl']:
                                position['sl'] = be_sl
                                logger.info(f"🔒 BREAKEVEN: {symbol} SL -> {be_sl:.5f}")

                # Check Timeout
                time_held = (datetime.utcnow() - position['entry_time']).total_seconds() / 60
                # 60 bars * 4 hours = 240 hours timeout?
                # Plan says 48 hours target. Let's set timeout to 72 hours (3 days).
                if time_held >= (72 * 60):
                    await self.close_position(connection, symbol, 'TIMEOUT')

            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")

            # Yield control to event loop
            await asyncio.sleep(0)

    async def close_position(self, connection, symbol, reason):
        """Close a position"""
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        side = position.get('side', 'BUY')

        try:
            # Get current price
            price = await connection.get_symbol_price(symbol)
            # Exit price: Bid for BUY (Sell to close), Ask for SELL (Buy to close)
            exit_price = price['bid'] if side == 'BUY' else price['ask']

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

            if side == 'BUY':
                pips = (exit_price - position['entry_price']) / pip_size
            else: # SELL
                pips = (position['entry_price'] - exit_price) / pip_size

            # Correct pip value calculation
            lot_size = position['lot_size']
            if symbol == 'USDJPY':
                pip_value = lot_size * 1000  # JPY pairs: 1 lot = 100,000 units, 1 pip = 0.01
            else:
                pip_value = lot_size * 10  # Standard pairs: 1 lot = 100,000 units, 1 pip = 0.0001

            pnl = pips * pip_value

            # Calculate time held
            time_held = (datetime.utcnow() - position['entry_time']).total_seconds() / 60

            # Log & Notify
            self.log_trade_exit(
                symbol, side, position['entry_price'], exit_price,
                pnl, reason, time_held
            )

            # Update balance (Simulated)
            # In live mode, balance updates automatically from broker, but for tracking:
            self.trades_today.append({
                'symbol': symbol,
                'pnl': pnl,
                'time': datetime.utcnow()
            })

            del self.open_positions[symbol]
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

                        # Check Market Hours (Forex Closed Wknds)
                        if not self.is_market_open():
                            logger.info("💤 Market Closed (Weekend). Sleeping for 1 hour...")
                            await asyncio.sleep(3600)
                            continue

                        # Check for signals (every minute)
                        await self.check_signals(connection)

                        # Check positions
                        await self.check_positions(connection)

                        # Dashboard update
                        account_info = await connection.get_account_information()
                        self.log_dashboard(account_info['balance'])

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

        # Log open positions (don't try to close on shutdown to avoid async issues)
        if self.open_positions:
            logger.warning(f"⚠️ {len(self.open_positions)} position(s) open at shutdown: {list(self.open_positions.keys())}")
            logger.info("Positions saved to state.json and will be restored on restart")

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
