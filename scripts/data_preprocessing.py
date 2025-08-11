# Filename: task1_data_preprocessing.py

import yfinance as yf
import pandas as pd
import numpy as np

# Settings
pd.set_option("display.max_columns", None)

# Define tickers and date range
TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'
END_DATE = '2025-07-31'

# Download data
data = {}
for ticker in TICKERS:
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df['Ticker'] = ticker
    data[ticker] = df

# Merge into one DataFrame
df_all = pd.concat(data.values(), keys=data.keys(), names=['Ticker', 'Date']).reset_index()

# Save raw data
df_all.to_csv("raw_financial_data.csv", index=False)

# Pivot data for analysis
pivot_close = df_all.pivot(index='Date', columns='Ticker', values='Close')
pivot_volume = df_all.pivot(index='Date', columns='Ticker', values='Volume')

# Handle missing values
pivot_close = pivot_close.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
pivot_volume = pivot_volume.fillna(0)

# Daily returns
daily_returns = pivot_close.pct_change().dropna()

# Rolling volatility (30-day)
rolling_volatility = daily_returns.rolling(window=30).std()

# Save processed datasets
pivot_close.to_csv("pivot_close.csv")
pivot_volume.to_csv("pivot_volume.csv")
daily_returns.to_csv("daily_returns.csv")
rolling_volatility.to_csv("rolling_volatility.csv")

print("Data preprocessing completed and files saved.")
