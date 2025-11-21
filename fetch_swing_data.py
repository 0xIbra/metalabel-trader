import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv('EODHD_API_KEY')
if not API_KEY:
    raise ValueError("EODHD_API_KEY not found in .env file")

# Swing Trading Universe
MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
CROSSES = [
    'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY',
    'EURAUD', 'EURNZD', 'EURCAD', 'EURCHF',
    'GBPAUD', 'GBPNZD', 'GBPCAD', 'GBPCHF',
    'AUDNZD', 'AUDCAD', 'AUDCHF',
    'NZDCAD', 'NZDCHF',
    'CADCHF'
]
COMMODITIES = ['XAUUSD']

UNIVERSE = MAJORS + CROSSES + COMMODITIES

def fetch_hourly_data(symbol, days=720):
    """
    Fetch 1-hour data from EODHD and resample to 4-hour.
    Fetching ~2 years of data for swing analysis.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # EODHD URL for Forex/Commodities
    # Note: Gold is usually 'XAUUSD.FOREX' or 'XAUUSD.CC' depending on provider,
    # but EODHD uses 'XAUUSD.FOREX' usually.
    suffix = ".FOREX"

    print(f"Fetching {symbol} (1H -> 4H)...")

    url = f"https://eodhd.com/api/intraday/{symbol}{suffix}"

    params = {
        'api_token': API_KEY,
        'interval': '1h',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp()),
        'fmt': 'json'
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"❌ Error fetching {symbol}: {response.status_code}")
            return None

        data = response.json()

        if not data:
            print(f"⚠️ No data for {symbol}")
            return None

        df = pd.DataFrame(data)

        # Process Timestamp
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        df.set_index('timestamp', inplace=True)

        # Ensure numeric
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Resample to 4H
        # Logic:
        # Open = First Open
        # High = Max High
        # Low = Min Low
        # Close = Last Close
        # Volume = Sum Volume

        ohlcv_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        df_4h = df.resample('4H').agg(ohlcv_dict).dropna()

        # Reset index to make timestamp a column again
        df_4h.reset_index(inplace=True)

        # Convert timestamp back to unix int for consistency
        df_4h['timestamp'] = df_4h['timestamp'].astype(int) // 10**9

        return df_4h

    except Exception as e:
        print(f"❌ Exception for {symbol}: {e}")
        return None

def main():
    output_dir = 'data/swing'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Swing Data Fetch for {len(UNIVERSE)} pairs...")
    print(f"Output Directory: {output_dir}")

    success_count = 0

    for symbol in UNIVERSE:
        df = fetch_hourly_data(symbol)

        if df is not None:
            output_path = f"{output_dir}/{symbol.lower()}_h4.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {symbol}: {len(df)} bars")
            success_count += 1

        # Rate limit protection
        time.sleep(1)

    print(f"\nCompleted! Successfully fetched {success_count}/{len(UNIVERSE)} pairs.")

if __name__ == "__main__":
    main()
