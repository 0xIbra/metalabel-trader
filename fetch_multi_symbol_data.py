import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv('EODHD_API_KEY')
if not API_KEY:
    raise ValueError("EODHD_API_KEY not found in .env file")

def fetch_symbol_data(symbol, days=90):
    """
    Fetch M1 data for a specific forex symbol
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Format for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"Fetching {symbol} M1 data")
    print(f"Period: {start_str} to {end_str} ({days} days)")
    print('='*80)

    # EODHD URL for M1 forex data
    url = f"https://eodhd.com/api/intraday/{symbol}.FOREX"

    params = {
        'api_token': API_KEY,
        'interval': '1m',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp()),
        'fmt': 'json'
    }

    print(f"Requesting data from EODHD API...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    data = response.json()

    if not data:
        print(f"❌ No data returned for {symbol}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Check if 'datetime' key exists, otherwise look for alternatives
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime']).astype(int) // 10**9
    elif 'timestamp' in df.columns:
        # Already has timestamp column
        pass
    else:
        print(f"❌ No timestamp column found. Columns: {df.columns.tolist()}")
        return None

    # Ensure we have OHLCV columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            return None

    # Select relevant columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    print(f"✅ Downloaded {len(df):,} bars")
    print(f"   From: {datetime.fromtimestamp(df['timestamp'].min())}")
    print(f"   To:   {datetime.fromtimestamp(df['timestamp'].max())}")

    return df

def main():
    """
    Fetch data for multiple forex symbols
    """
    symbols = ['GBPUSD', 'USDJPY', 'AUDUSD']  # EURUSD already fetched

    os.makedirs('data/raw', exist_ok=True)

    for symbol in symbols:
        df = fetch_symbol_data(symbol, days=90)

        if df is not None:
            output_file = f'data/raw/{symbol.lower()}_m1_extended.csv'
            df.to_csv(output_file, index=False)
            print(f"💾 Saved to {output_file}")
        else:
            print(f"⚠️  Skipping {symbol} - no data")

        print()

if __name__ == "__main__":
    main()
