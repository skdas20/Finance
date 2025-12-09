"""
Download historical stock data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Define a diverse basket of stocks for Generalization Research
# Sector diversity ensures the model learns market dynamics, not just one stock's personality.
tickers = {
    'Tech': ['AAPL', 'MSFT', 'NVDA'],
    'Finance': ['JPM', 'BAC', 'GS'],
    'Healthcare': ['JNJ', 'PFE', 'UNH'],
    'Consumer': ['PG', 'KO', 'WMT'],
    'Energy': ['XOM', 'CVX']
}

print("Downloading Research-Grade Dataset (10 Years)...")
end_date = datetime.now()
start_date = end_date - timedelta(days=10*365) # 10 Years of data

all_tickers = [t for sector in tickers.values() for t in sector]

# Download S&P 500 Benchmark
print(f"Downloading Benchmark (S&P 500)...")
sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
sp500.to_csv('data/benchmark_sp500.csv')

for sector, stocks in tickers.items():
    print(f"\nProcessing Sector: {sector}")
    for stock in stocks:
        print(f"  Downloading {stock}...", end=' ')
        try:
            data = yf.download(stock, start=start_date, end=end_date, progress=False)
            # Ensure we have enough data
            if len(data) > 200:
                data.to_csv(f'data/{stock}.csv')
                print(f"✓ ({len(data)} trading days)")
            else:
                print(f"⚠ (Skipped: Insufficient data)")
        except Exception as e:
            print(f"❌ Error: {e}")

print("\n✓ Research Dataset Preparation Complete.")

