import unittest
from datetime import datetime, timedelta
from collections import deque

class TestCandleAggregation(unittest.TestCase):
    def setUp(self):
        self.price_buffers = {'EURUSD': deque(maxlen=100)}
        self.current_bar = {}  # Simulating the bot's state
        self.completed_bars = []

    def on_tick(self, tick):
        """Simulated on_tick logic from the planned fix"""
        symbol = tick['symbol']
        price = tick['bid']
        volume = tick.get('volume', 1000)
        timestamp = tick['time'] # Using datetime object for test simplicity

        # Round down to nearest minute
        current_minute = timestamp.replace(second=0, microsecond=0)

        # Initialize current bar if needed
        if symbol not in self.current_bar:
            self.current_bar[symbol] = {
                'timestamp': current_minute,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
            }
            return

        bar = self.current_bar[symbol]

        # Check if new minute
        if current_minute > bar['timestamp']:
            # Finalize previous bar
            self.price_buffers[symbol].append(bar.copy())
            self.completed_bars.append(bar.copy())

            # Start new bar
            self.current_bar[symbol] = {
                'timestamp': current_minute,
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

    def test_aggregation(self):
        # Minute 1: 10:00
        t1 = datetime(2023, 1, 1, 10, 0, 5)
        self.on_tick({'symbol': 'EURUSD', 'bid': 1.0500, 'volume': 100, 'time': t1})

        t2 = datetime(2023, 1, 1, 10, 0, 30)
        self.on_tick({'symbol': 'EURUSD', 'bid': 1.0510, 'volume': 100, 'time': t2})

        t3 = datetime(2023, 1, 1, 10, 0, 55)
        self.on_tick({'symbol': 'EURUSD', 'bid': 1.0490, 'volume': 100, 'time': t3})

        # Verify current bar state (still open)
        bar = self.current_bar['EURUSD']
        self.assertEqual(bar['open'], 1.0500)
        self.assertEqual(bar['high'], 1.0510)
        self.assertEqual(bar['low'], 1.0490)
        self.assertEqual(bar['close'], 1.0490)
        self.assertEqual(bar['volume'], 300)
        self.assertEqual(len(self.price_buffers['EURUSD']), 0) # No completed bars yet

        # Minute 2: 10:01 (Triggers finalization of Minute 1)
        t4 = datetime(2023, 1, 1, 10, 1, 5)
        self.on_tick({'symbol': 'EURUSD', 'bid': 1.0520, 'volume': 100, 'time': t4})

        # Verify Minute 1 is saved
        self.assertEqual(len(self.price_buffers['EURUSD']), 1)
        saved_bar = self.price_buffers['EURUSD'][0]
        self.assertEqual(saved_bar['timestamp'], datetime(2023, 1, 1, 10, 0, 0))
        self.assertEqual(saved_bar['close'], 1.0490)

        # Verify Minute 2 is active
        current = self.current_bar['EURUSD']
        self.assertEqual(current['timestamp'], datetime(2023, 1, 1, 10, 1, 0))
        self.assertEqual(current['open'], 1.0520)

if __name__ == '__main__':
    unittest.main()
