import unittest
import numpy as np
import pandas as pd
import os
from src.quant_engine.main import QuantEngine
from src.shared.datatypes import TickData
from src.quant_engine.indicators import calculate_log_returns, calculate_volatility, calculate_z_score

class TestQuantEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load real market data once for all tests"""
        data_path = "data/raw/eurusd_m1.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df.columns = [c.lower() for c in df.columns]
            # Take first 50 rows as test sample
            cls.real_data = df.head(50)
        else:
            cls.real_data = None

    def test_indicators_with_real_data(self):
        """Test Numba indicators with real market data"""
        if self.real_data is None:
            self.skipTest("Real market data not available")

        prices = self.real_data['close'].values[:30]

        # Log returns
        log_ret = calculate_log_returns(prices)
        self.assertEqual(len(log_ret), len(prices))
        self.assertGreater(np.abs(log_ret[1:]).sum(), 0)  # Should have movement

        # Volatility
        vol = calculate_volatility(log_ret, window=20)
        self.assertGreater(vol[-1], 0)

        # Z-Score
        z = calculate_z_score(prices, window=20)
        self.assertIsNotNone(z[-1])

    def test_engine_flow_with_real_data(self):
        """Test QuantEngine with real market ticks"""
        if self.real_data is None:
            self.skipTest("Real market data not available")

        engine = QuantEngine(window_size=20)

        # Feed real ticks
        for idx, row in self.real_data.iterrows():
            tick = TickData(
                symbol="EURUSD",
                bid=row['close'],  # Using close as bid
                ask=row['close'] + 0.0001,  # Synthetic spread
                timestamp=float(row['timestamp']),
                volume=float(row['volume'])
            )
            features = engine.on_tick(tick)

            if idx >= 20:  # After enough data
                self.assertIsNotNone(features)

                # Check all expected keys
                expected_keys = ["z_score", "rsi", "volatility", "adx", "time_sin", "volume_delta"]
                for key in expected_keys:
                    self.assertIn(key, features)

                # Sanity checks
                self.assertGreaterEqual(features["rsi"], 0)
                self.assertLessEqual(features["rsi"], 100)
                self.assertGreaterEqual(features["time_sin"], -1)
                self.assertLessEqual(features["time_sin"], 1)

                # Break after first valid feature check
                break


if __name__ == '__main__':
    unittest.main()
