import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import asyncio
from src.shared.datatypes import TickData

# Mock the config BEFORE importing FeedHandler if possible,
# but since FeedHandler imports config at module level, we might need to patch where it's used
# or patch the module in sys.modules.
# However, the error happens at import time of main.py because it imports config.py which raises error.

# We need to mock os.getenv or the config module before import.
from unittest.mock import patch
import os

# Patch os.getenv to return a dummy key so config import doesn't fail
with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
    from src.feed_handler.main import FeedHandler


class TestFeedHandler(unittest.IsolatedAsyncioTestCase):
    async def test_process_message_valid(self):
        handler = FeedHandler(api_key="test_key")
        handler.handle_tick = MagicMock()

        # Valid message
        message = json.dumps({"s": "EURUSD", "a": 1.0543, "b": 1.0541, "t": 1678888888000})
        await handler.process_message(message)

        handler.handle_tick.assert_called_once()
        args = handler.handle_tick.call_args[0][0]
        self.assertIsInstance(args, TickData)
        self.assertEqual(args.symbol, "EURUSD")
        self.assertEqual(args.bid, 1.0541)
        self.assertEqual(args.ask, 1.0543)
        self.assertEqual(args.timestamp, 1678888888.0)

    async def test_process_message_invalid(self):
        handler = FeedHandler(api_key="test_key")
        handler.handle_tick = MagicMock()

        # Invalid message (missing fields)
        message = json.dumps({"s": "EURUSD", "t": 1678888888000})
        await handler.process_message(message)

        handler.handle_tick.assert_not_called()

if __name__ == '__main__':
    unittest.main()
