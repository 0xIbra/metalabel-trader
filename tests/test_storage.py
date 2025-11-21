import unittest
from unittest.mock import MagicMock, patch
from src.shared.storage import InfluxWriter
from src.shared.datatypes import TickData
import time

class TestStorage(unittest.TestCase):
    def test_write_tick(self):
        # Mock the client to avoid actual connection in unit test
        with patch('src.shared.storage.InfluxDBClient') as MockClient:
            mock_write_api = MagicMock()
            MockClient.return_value.write_api.return_value = mock_write_api

            writer = InfluxWriter(token="test-token")
            tick = TickData(symbol="EURUSD", bid=1.05, ask=1.06, timestamp=time.time())

            writer.write_tick(tick)

            mock_write_api.write.assert_called_once()

if __name__ == '__main__':
    unittest.main()
