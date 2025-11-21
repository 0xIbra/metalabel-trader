"""
Live trading bot using MetaApi for paper trading
Trades EURUSD and AUDUSD with 5:1 RR strategy
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
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
    calculate_rsi, calculate_adx, calculate_time_sin, calculate_volume_delta
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
SYMBOLS = ['EURUSD', 'AUDUSD']
CONFIDENCE_THRESHOLD = 0.50
ACCOUNT_BALANCE = 1000  # Starting balance
LEVERAGE = 30
RISK_PERCENT = 0.02
TP_PIPS = 5
SL_PIPS = 1
HOLD_BARS = 60  # 60 minutes timeout

# Strategy config
MAX_CONCURRENT_POSITIONS = 2  # 1 per symbol
DAILY_LOSS_LIMIT = 20  # $20 max loss per day
FEATURE_WINDOW = 100  # Keep last 100 bars for features


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

    def load_models(self):
        """Load XGBoost models for each symbol"""
        logger.info("Loading models...")

        for symbol in SYMBOLS:
            model_path = f"src/oracle/model_{symbol.lower()}.json" if symbol != 'EURUSD' else "src/oracle/model.json"

            if os.path.exists(model_path):
                try:
                    model = xgb.Booster()
                    model.load_model(model_path)
                    self.models[symbol] = model
                    logger.info(f"✅ Loaded model for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {symbol} model: {e}")
            else:
                logger.error(f"❌ Model not found: {model_path}")

        if not self.models:
            raise Exception("No models loaded!")

    async def connect_metaapi(self):
        """Connect to MetaApi"""
        try:
            from metaapi_cloud_sdk import MetaApi

            logger.info("Connecting to MetaApi...")
            self.api = MetaApi(METAAPI_TOKEN)
            self.account = await self.api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)

            # Wait for deployment
            logger.info("Deploying account...")
            await self.account.deploy()

            # Wait for connection
            logger.info("Waiting for connection...")
            await self.account.wait_connected()

            # Get connection
            connection = self.account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            logger.info("✅ Connected to MetaApi")

            # Get account info
            account_info = await connection.get_account_information()
            logger.info(f"Account balance: ${account_info['balance']:.2f}")

            return connection

        except Exception as e:
            logger.error(f"❌ MetaApi connection failed: {e}")
            await notify_error(str(e), "Check MetaApi credentials")
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
        features = np.array([[
            z_score[-1], rsi[-1], volatility[-1], adx[-1], time_sin[-1], volume_delta[-1],
            roc_5[-1], roc_10[-1], roc_20[-1], macd[-1], velocity[-1],
            close_lag1[-1], close_lag2[-1], close_lag3[-1], returns_lag1[-1], returns_lag2[-1]
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
        """Get model prediction"""
        if symbol not in self.models or features is None:
            return 0.0

        feature_names = [
            'z_score', 'rsi', 'volatility', 'adx', 'time_sin', 'volume_delta',
            'roc_5', 'roc_10', 'roc_20', 'macd', 'velocity',
            'close_lag1', 'close_lag2', 'close_lag3', 'returns_lag1', 'returns_lag2'
        ]

        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        predictions = self.models[symbol].predict(dmatrix)

        # Get BUY probability (3-class model)
        if len(predictions.shape) > 1 and predictions.shape[1] == 3:
            buy_prob = predictions[0][2]  # BUY class
        else:
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
                continue

            # Check account balance
            try:
                account_info = await connection.get_account_information()
                current_balance = account_info['balance']

                if current_balance < 100:  # Minimum balance check
                    logger.error(f"Insufficient balance: ${current_balance:.2f}")
                    await notify_error(f"Balance too low: ${current_balance:.2f}", "Add funds to account")
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

            # Check threshold
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Get current price
            try:
                price = await connection.get_symbol_price(symbol)
                current_price = price['ask']
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                continue

            # Calculate TP/SL
            pip_size = 0.0001 if symbol != 'USDJPY' else 0.01
            tp_price = current_price + (TP_PIPS * pip_size)
            sl_price = current_price - (SL_PIPS * pip_size)

            # Calculate lot size dynamically based on balance and risk
            risk_amount = current_balance * RISK_PERCENT
            pip_value_per_lot = 10 if symbol != 'USDJPY' else 1000
            lot_size = risk_amount / (SL_PIPS * pip_value_per_lot)
            lot_size = round(lot_size, 2)  # Round to 2 decimals
            lot_size = max(0.01, min(lot_size, 0.10))  # Clamp between 0.01 and 0.10

            # Place order
            try:
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
                    'lot_size': lot_size,
                    'tp': tp_price,
                    'sl': sl_price,
                    'confidence': confidence,
                    'order_id': result['orderId']
                }

                # Notify
                await notify_trade_entry(
                    symbol, 'BUY', current_price, tp_price, sl_price,
                    lot_size, risk_amount, confidence * 100
                )

                logger.info(f"🟢 Opened {symbol} @ {current_price:.5f} (Confidence: {confidence:.1%}, Lot: {lot_size})")

            except Exception as e:
                logger.error(f"Failed to open {symbol}: {e}")
                await notify_error(f"Failed to open {symbol}", str(e))

    async def check_positions(self, connection):
        """Check and manage open positions"""
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]

            # Check timeout
            time_held = (datetime.utcnow() - position['entry_time']).total_seconds() / 60
            if time_held >= HOLD_BARS:
                await self.close_position(connection, symbol, 'TIMEOUT')

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
            await notify_trade_exit(
                symbol, exit_price, pnl, pips, reason, hold_time, balance
            )

            logger.info(f"💰 Closed {symbol} @ {exit_price:.5f} | P&L: ${pnl:+.2f} ({reason})")

            # Remove position
            del self.open_positions[symbol]

        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            # Still remove from tracking to avoid stuck state
            if symbol in self.open_positions:
                del self.open_positions[symbol]

    async def on_tick(self, tick):
        """Handle new tick data"""
        symbol = tick['symbol']

        if symbol not in SYMBOLS:
            return

        # Add to buffer
        self.price_buffers[symbol].append({
            'timestamp': int(datetime.utcnow().timestamp()),
            'open': tick.get('bid', tick.get('price')),
            'high': tick.get('bid', tick.get('price')),
            'low': tick.get('bid', tick.get('price')),
            'close': tick.get('bid', tick.get('price')),
            'volume': tick.get('volume', 1000)
        })

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
                await notify_startup(SYMBOLS, CONFIDENCE_THRESHOLD, account_info['balance'])

                logger.info("🚀 Bot started. Monitoring signals...")

                # Subscribe to market data with verification
                for symbol in SYMBOLS:
                    try:
                        await connection.subscribe_to_market_data(symbol)
                        logger.info(f"✅ Subscribed to {symbol} market data")
                    except Exception as e:
                        logger.error(f"❌ Failed to subscribe to {symbol}: {e}")
                        await notify_error(f"Subscription failed for {symbol}", str(e))
                        raise

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
                            await notify_status(
                                account_info['balance'],
                                len(self.open_positions),
                                len(self.trades_today),
                                sum([t['pnl'] for t in self.trades_today]),
                                win_rate
                            )
                            last_status = datetime.utcnow()

                        # Sleep for 1 minute
                        await asyncio.sleep(60)

                    except Exception as e:
                        # Handle connection errors during main loop
                        if 'connection' in str(e).lower() or 'timeout' in str(e).lower():
                            logger.error(f"Connection error in main loop: {e}")
                            await notify_error("Connection lost", "Attempting to reconnect...")
                            break  # Break inner loop to trigger reconnection
                        else:
                            logger.error(f"Error in main loop: {e}")
                            await asyncio.sleep(60)  # Continue after error

            except KeyboardInterrupt:
                logger.info("⚠️ Keyboard interrupt received")
                await notify_shutdown("Manual stop")
                break

            except Exception as e:
                logger.error(f"❌ Bot error: {e}")
                retry_count += 1

                if retry_count < max_retries:
                    wait_time = retry_count * 30  # Exponential backoff: 30s, 60s, 90s
                    logger.info(f"Reconnecting in {wait_time}s... (Attempt {retry_count}/{max_retries})")
                    await notify_error(str(e), f"Reconnecting in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Stopping bot.")
                    await notify_error("Max reconnection attempts reached", "Bot stopped")
                    break

        # Cleanup
        self.is_running = False

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
