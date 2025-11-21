import time
from src.shared.storage import InfluxWriter
from src.shared.datatypes import TickData
from influxdb_client import InfluxDBClient

import logging

# Configure logging to see InfluxWriter errors
logging.basicConfig(level=logging.DEBUG)

def test_live_connection():

    token = "my-super-secret-auth-token"
    org = "metalabel"
    bucket = "market_data"

    print("Connecting to InfluxDB...")
    client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

    # Check connection by listing buckets
    try:
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        print(f"Available buckets: {[b.name for b in buckets]}")
    except Exception as e:
        print(f"Failed to list buckets: {e}")
        return

    writer = InfluxWriter(token=token, org=org, bucket=bucket)


    # Write a test tick
    tick = TickData(symbol="TESTUSD", bid=1.0, ask=1.1, timestamp=time.time())
    print(f"Writing tick: {tick}")
    writer.write_tick(tick)

    # Give it a moment to persist
    time.sleep(1)

    # Query back
    client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
    query_api = client.query_api()

    # Query back with a wider range to catch timezone issues
    query = f'from(bucket: "{bucket}") |> range(start: -1h) |> filter(fn: (r) => r["_measurement"] == "tick_data") |> filter(fn: (r) => r["symbol"] == "TESTUSD")'


    print("Querying data...")
    tables = query_api.query(query)

    found = False
    for table in tables:
        for record in table.records:
            print(f"Found record: {record.get_field()} = {record.get_value()}")
            found = True

    if found:
        print("SUCCESS: Data written and retrieved!")
    else:
        print("FAILURE: No data found.")

if __name__ == "__main__":
    try:
        test_live_connection()
    except Exception as e:
        print(f"ERROR: {e}")
