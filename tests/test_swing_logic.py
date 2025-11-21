import unittest
import asyncio
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from live_trading_bot import LiveTradingBot

class TestSwingLogic(unittest.TestCase):
    def setUp(self):
        # Patch SYMBOLS to just EURUSD to avoid multi-symbol noise
        self.symbols_patcher = patch('live_trading_bot.SYMBOLS', ['EURUSD'])
        self.mock_symbols = self.symbols_patcher.start()

        # Patch load_historical_data to avoid reading CSVs during test
        self.load_hist_patcher = patch.object(LiveTradingBot, 'load_historical_data')
        self.mock_load_hist = self.load_hist_patcher.start()

        self.bot = LiveTradingBot()
        # Mock model to avoid loading actual files
        self.bot.model = MagicMock()
        self.bot.model.predict.return_value = np.array([[0.1, 0.1, 0.8]]) # High BUY prob

        # Mock connection
        self.connection = AsyncMock()
        self.connection.get_account_information.return_value = {'balance': 10000}
        self.connection.get_symbol_price.return_value = {'bid': 1.1000, 'ask': 1.1002}
        self.connection.create_market_buy_order.return_value = {'orderId': '12345'}

    def tearDown(self):
        self.symbols_patcher.stop()
        self.load_hist_patcher.stop()

    def test_feature_computation(self):
        """Test that features are computed correctly from H4 data"""
        symbol = 'EURUSD'

        # Generate dummy H4 data (100 bars)
        data = []
        base_price = 1.1000
        for i in range(100):
            price = base_price + (i * 0.0005) # Uptrend
            data.append({
                'timestamp': int(datetime.utcnow().timestamp()) + i*14400,
                'open': price,
                'high': price + 0.0020,
                'low': price - 0.0020,
                'close': price,
                'volume': 1000
            })

        self.bot.price_buffers[symbol].extend(data)

        features = self.bot.compute_features(symbol)

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 20)) # 20 features

        # Check ATR (index 8) is not zero
        # Features: z_score, rsi, volatility, adx, time_sin, volume_delta,
        # bandwidth, bb_position, atr_pct, dist_pivot...
        # ATR pct should be > 0
        self.assertGreater(features[0][8], 0)

    def test_signal_generation_atr_risk(self):
        """Test that signals use ATR for TP/SL"""
        symbol = 'EURUSD'

        # Setup buffer with known ATR
        # High-Low = 0.0020 constant
        # So ATR should be approx 0.0020
        data = []
        base_price = 1.1000
        for i in range(100):
            # Add small noise to avoid zero volatility/std which causes NaN Z-score
            noise = (i % 2) * 0.0001
            price = base_price + noise
            data.append({
                'timestamp': int(datetime.utcnow().timestamp()) + i*14400,
                'open': price,
                'high': price + 0.0020,
                'low': price - 0.0020,
                'close': price,
                'volume': 1000
            })
        self.bot.price_buffers[symbol].extend(data)

        # Run check_signals
        asyncio.run(self.bot.check_signals(self.connection))

        # Verify order placement
        self.connection.create_market_buy_order.assert_called_once()
        call_args = self.connection.create_market_buy_order.call_args[1]

        # Expected ATR ~ 0.0020 (since TR is always 0.0040? No, High-Low=0.0040. Wait.
        # High = 1.1020, Low = 1.0980. Diff = 0.0040.
        # So ATR should be 0.0040.

        # TP = 2 * ATR = 0.0080
        # SL = 1 * ATR = 0.0040
        # Entry = 1.1002 (Ask)

        expected_tp = 1.1002 + 0.0080
        expected_sl = 1.1002 - 0.0040

        self.assertAlmostEqual(call_args['take_profit'], expected_tp, places=4)
        self.assertAlmostEqual(call_args['stop_loss'], expected_sl, places=4)

    def test_trailing_stop(self):
        """Test Chandelier Exit logic"""
        symbol = 'EURUSD'
        atr = 0.0020
        entry_price = 1.1000
        sl = 1.0980 # Initial SL (1 ATR)

        # Setup open position
        self.bot.open_positions[symbol] = {
            'entry_time': datetime.utcnow(),
            'entry_price': entry_price,
            'lot_size': 0.1,
            'tp': 1.1100,
            'sl': sl,
            'confidence': 0.9,
            'order_id': '123',
            'highest_high': entry_price,
            'atr': atr,
            'has_added': False
        }

        # Mock price moving UP to 1.1050
        # New High = 1.1050
        # Trail = 1.5 * ATR = 0.0030
        # New SL = 1.1050 - 0.0030 = 1.1020
        self.connection.get_symbol_price.return_value = {'bid': 1.1050, 'ask': 1.1052}

        asyncio.run(self.bot.check_positions(self.connection))

        # Verify SL updated
        pos = self.bot.open_positions[symbol]
        self.assertEqual(pos['highest_high'], 1.1050)
        self.assertAlmostEqual(pos['sl'], 1.1020, places=4)

if __name__ == '__main__':
    unittest.main()
