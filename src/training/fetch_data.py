import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from src.config import EODHD_API_KEY

def fetch_historical_data(symbol="EURUSD.FOREX", days=30):
    """
    Fetch historical M1 data from EODHD.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # EODHD Intraday API format:
    # https://eodhistoricaldata.com/api/intraday/{SYMBOL}?api_token={TOKEN}&interval=1m&from={FROM}&to={TO}
    # Timestamps are UNIX

    from_ts = int(start_date.timestamp())
    to_ts = int(end_date.timestamp())

    url = f"https://eodhistoricaldata.com/api/intraday/{symbol}?api_token={EODHD_API_KEY}&interval=1m&from={from_ts}&to={to_ts}&fmt=json"

    print(f"Fetching data from {start_date} to {end_date}...")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if not data:
            print("No data returned.")
            return

        df = pd.DataFrame(data)
        # EODHD returns: timestamp, open, high, low, close, volume
        # We need to map/rename if necessary.
        # Usually keys are: 'timestamp', 'open', 'high', 'low', 'close', 'volume'

        # Save to CSV
        output_path = "data/raw/eurusd_m1.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        return df
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Fetch 4 weeks (approx 30 days)
    fetch_historical_data(days=30)
