import unittest
import numpy as np
from src.quant_engine.main import QuantEngine
from src.shared.datatypes import TickData
from src.quant_engine.indicators import calculate_log_returns, calculate_volatility, calculate_z_score

class TestQuantEngine(unittest.TestCase):
    def test_indicators(self):
        # Test Numba functions directly
        prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0])

        # Log returns
        returns = calculate_log_returns(prices)
        self.assertEqual(len(returns), 5)
        self.assertAlmostEqual(returns[1], np.log(101.0/100.0))

        # Volatility
        vol = calculate_volatility(returns, window=5)
        self.assertGreater(vol, 0)

        # Z-Score
        z = calculate_z_score(100.0, prices)
        self.assertNotEqual(z, 0) # Should be non-zero given the variance

    def test_engine_flow(self):
        engine = QuantEngine(window_size=5)

        # Feed initial ticks
        for i in range(5):
            tick = TickData(symbol="EURUSD", bid=1.0 + i*0.01, ask=1.0 + i*0.01 + 0.0001, timestamp=1000+i)
            features = engine.on_tick(tick)

        # After 5 ticks, we should have features
        self.assertIsNotNone(features)
        self.assertIn("z_score", features)
        self.assertIn("rsi", features)
        self.assertIn("volatility", features)

if __name__ == '__main__':
    unittest.main()
