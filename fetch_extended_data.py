import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from src.config import EODHD_API_KEY

def fetch_historical_data(symbol="EURUSD.FOREX", days=90):
    """
    Fetch historical M1 data from EODHD.
    EODHD intraday API has limits - typically 30-120 days for M1 data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    from_ts = int(start_date.timestamp())
    to_ts = int(end_date.timestamp())

    url = f"https://eodhistoricaldata.com/api/intraday/{symbol}?api_token={EODHD_API_KEY}&interval=1m&from={from_ts}&to={to_ts}&fmt=json"

    print(f"Fetching {days} days of M1 data...")
    print(f"From: {start_date}")
    print(f"To:   {end_date}")
    print(f"\nAPI URL (truncated): {url[:80]}...")

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if not data:
            print("❌ No data returned from API")
            print("This could mean:")
            print("  1. EODHD doesn't have data for this range")
            print("  2. API key issue")
            print("  3. Symbol format incorrect")
            return None

        df = pd.DataFrame(data)

        # Print data info
        print(f"\n✅ Successfully fetched {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {pd.to_datetime(df['timestamp'].min(), unit='s')} to {pd.to_datetime(df['timestamp'].max(), unit='s')}")
        print(f"Actual days: {(pd.to_datetime(df['timestamp'].max(), unit='s') - pd.to_datetime(df['timestamp'].min(), unit='s')).days}")

        # Save to CSV
        output_path = "data/raw/eurusd_m1_extended.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to {output_path}")

        return df
    else:
        print(f"❌ Error fetching data: {response.status_code}")
        print(f"Response: {response.text[:200]}")
        return None

if __name__ == "__main__":
    # Try to fetch 90 days
    print("=" * 70)
    print("FETCHING EXTENDED HISTORICAL DATA FOR MODEL IMPROVEMENT")
    print("=" * 70)
    print("\nAttempting to fetch 90 days...")

    df = fetch_historical_data(days=90)

    if df is not None:
        print("\n" + "=" * 70)
        print("SUCCESS - Ready to retrain model with more data!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Run: python3.10 src/training/train_model.py")
        print(f"2. Update data_path to 'data/raw/eurusd_m1_extended.csv'")
        print(f"3. Evaluate improvement with walk-forward validation")
    else:
        print("\n" + "=" * 70)
        print("FALLBACK - Trying 60 days...")
        print("=" * 70)
        df = fetch_historical_data(days=60)
