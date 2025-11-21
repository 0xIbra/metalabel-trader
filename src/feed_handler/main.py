import asyncio
import json
import websockets
import logging
from datetime import datetime
from src.config import EODHD_API_KEY
from src.shared.datatypes import TickData
from src.shared.storage import InfluxWriter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeedHandler")

class FeedHandler:
    def __init__(self, api_key: str, symbol: str = "EURUSD"):
        self.api_key = api_key
        self.symbol = symbol
        self.uri = f"wss://ws.eodhistoricaldata.com/ws/forex?api_token={self.api_key}"
        self.running = False

        # Initialize InfluxWriter with fixed token for now
        self.writer = InfluxWriter(token="my-super-secret-auth-token")


    async def connect(self):
        self.running = True
        while self.running:
            try:
                logger.info(f"Connecting to {self.uri}...")
                async with websockets.connect(self.uri) as websocket:
                    logger.info("Connected!")

                    # Subscribe to the symbol
                    subscribe_msg = {
                        "action": "subscribe",
                        "symbols": self.symbol
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {self.symbol}")

                    async for message in websocket:
                        await self.process_message(message)
            except Exception as e:
                logger.error(f"Connection error: {e}")
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def process_message(self, message: str):
        try:
            data = json.loads(message)
            # EODHD Forex WebSocket format example:
            # {"s": "EURUSD", "a": 1.0543, "b": 1.0541, "t": 1678888888000}
            # keys: s=symbol, a=ask, b=bid, t=timestamp (ms)

            if "s" in data and "a" in data and "b" in data:
                tick = TickData(
                    symbol=data["s"],
                    bid=float(data["b"]),
                    ask=float(data["a"]),
                    timestamp=float(data["t"]) / 1000.0  # Convert ms to seconds
                )
                self.handle_tick(tick)
            else:
                # Heartbeats or other messages
                pass
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def handle_tick(self, tick: TickData):
        # Write to InfluxDB
        self.writer.write_tick(tick)
        logger.info(f"Tick received and stored: {tick}")


async def main():
    feed = FeedHandler(api_key=EODHD_API_KEY)
    await feed.connect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping Feed Handler...")
