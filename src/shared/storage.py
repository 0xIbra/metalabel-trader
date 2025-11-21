import logging
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from src.shared.datatypes import TickData

logger = logging.getLogger("InfluxWriter")

class InfluxWriter:
    def __init__(self, url="http://localhost:8086", token="token-secrect", org="metalabel", bucket="market_data"):
        """
        Initialize InfluxDB writer.
        Note: In a real setup, token should be passed securely or retrieved after setup.
        For the docker-compose auto-setup, we might need to generate a token or use the one created.
        However, the auto-setup in docker-compose creates a user/password, but generating a token
        programmatically or using a fixed one requires more config.

        For this 'Week 1' MVP, we will assume the user will provide the token after initial setup
        or we use the CLI to retrieve it.

        BUT, `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN` can be set in docker-compose to fix the token.
        Let's update docker-compose to set a fixed token for development ease.
        """
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org

    def write_tick(self, tick: TickData):
        try:
            point = (
                Point("tick_data")
                .tag("symbol", tick.symbol)
                .field("bid", tick.bid)
                .field("ask", tick.ask)
                .time(datetime.utcfromtimestamp(tick.timestamp))
            )

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            # logger.debug(f"Wrote tick to InfluxDB: {tick}")
        except Exception as e:
            logger.error(f"Failed to write to InfluxDB: {e}")

    def close(self):
        self.client.close()
