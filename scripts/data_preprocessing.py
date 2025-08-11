
import pandas as pd
import yfinance as yf

def preprocess_data(tickers=None, start_date=None, end_date=None):
    if tickers is None:
        tickers = ['TSLA', 'BND', 'SPY']
    if start_date is None:
        start_date = '2015-07-01'
    if end_date is None:
        end_date = '2025-07-31'

    # Download data
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df['Ticker'] = ticker
        data[ticker] = df

    # Merge into one DataFrame
    df_all = pd.concat(data.values(), keys=data.keys(), names=['Ticker', 'Date']).reset_index()

    # Pivot data
    pivot_close = df_all.pivot(index='Date', columns='Ticker', values='Close')
    pivot_volume = df_all.pivot(index='Date', columns='Ticker', values='Volume')

    # Handle missing values
    pivot_close = pivot_close.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    pivot_volume = pivot_volume.fillna(0)

    # Daily returns
    daily_returns = pivot_close.pct_change().dropna()

    # Rolling volatility (30-day)
    rolling_volatility = daily_returns.rolling(window=30).std()

    return {
        "pivot_close": pivot_close,
        "pivot_volume": pivot_volume,
        "daily_returns": daily_returns,
        "rolling_volatility": rolling_volatility
    }
