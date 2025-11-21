"""
Fetch Macro Features for Swing Trading Enhancement
- Economic Calendar (EODHD)
- News Sentiment (EODHD)
- Market Regime Indicators (Yahoo Finance)
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()
API_KEY = os.getenv('EODHD_API_KEY')

def fetch_economic_calendar(symbols=['USD', 'EUR', 'GBP', 'JPY'], days=30):
    """
    Fetch economic calendar events from EODHD
    Focus on high-impact events (NFP, FOMC, CPI, GDP)
    """
    print("Fetching Economic Calendar...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    url = "https://eodhd.com/api/economic-events"
    params = {
        'api_token': API_KEY,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'fmt': 'json'
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            events = response.json()
            df = pd.DataFrame(events)

            # Filter for high-impact events
            if 'importance' in df.columns:
                df = df[df['importance'].isin(['High', 'Medium'])]

            # Save
            df.to_csv('data/macro/economic_calendar.csv', index=False)
            print(f"✅ Saved {len(df)} economic events")
            return df
        else:
            print(f"❌ Failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def fetch_news_sentiment(symbols=['EURUSD', 'GBPUSD', 'USDJPY'], days=90):
    """
    Fetch news sentiment from EODHD for currency pairs
    """
    print("\nFetching News Sentiment...")

    all_sentiment = {}

    for symbol in symbols:
        print(f"  Processing {symbol}...")

        url = f"https://eodhd.com/api/sentiments"
        params = {
            'api_token': API_KEY,
            's': f"{symbol}.FOREX",
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    all_sentiment[symbol] = df
                    print(f"    ✅ {len(df)} sentiment records")
            else:
                print(f"    ⚠️  No data (status: {response.status_code})")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Save combined sentiment
    if all_sentiment:
        combined = pd.concat([df.assign(symbol=sym) for sym, df in all_sentiment.items()])
        combined.to_csv('data/macro/news_sentiment.csv', index=False)
        print(f"\n✅ Saved sentiment data for {len(all_sentiment)} symbols")

    return all_sentiment

def fetch_market_regime_data(days=365):
    """
    Fetch cross-asset data for market regime detection
    - BTC (risk-on proxy)
    - SPX (equity market)
    - VIX (volatility index)
    """
    print("\nSkipping Market Regime Indicators (yfinance complexity)")
    print("Focus on EODHD sentiment and economic calendar for now")
    return {}

def compute_cot_proxy_features():
    """
    Since CFTC COT data is weekly and complex to parse,
    we'll use EODHD's fundamental data as a proxy for institutional sentiment
    """
    print("\nNote: Using price action and volume as COT proxy")
    print("CFTC data requires separate weekly scraping - implement if needed")

def main():
    # Create output directory
    os.makedirs('data/macro', exist_ok=True)

    print("="*80)
    print("MACRO FEATURE DATA FETCHING")
    print("="*80)

    # 1. Economic Calendar
    calendar = fetch_economic_calendar()

    # 2. News Sentiment (try EODHD first)
    sentiment = fetch_news_sentiment()

    # 3. Market Regime Data
    regime_data = fetch_market_regime_data()

    print("\n" + "="*80)
    print("FETCH COMPLETE")
    print("="*80)
    print(f"  Economic Events: {'✅' if calendar is not None else '❌'}")
    print(f"  News Sentiment:  {'✅' if sentiment else '❌'}")
    print(f"  Regime Data:     {'✅' if regime_data else '❌'}")

if __name__ == "__main__":
    main()
