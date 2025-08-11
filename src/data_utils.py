import pandas as pd
import numpy as np
from pathlib import Path

def load_price_data(filename="sample.csv"):
    """
    Load a CSV file from the data folder and return a DataFrame with a 'price' column.
    Expects a 'Date' column for the index.
    """
    data_path = Path(__file__).resolve().parents[1] / "data" / filename
    df_raw = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

    # Rename 'Close' to 'price' if exists
    if 'Close' in df_raw.columns:
        df = df_raw[['Close']].rename(columns={'Close': 'price'})
    elif 'close' in df_raw.columns:
        df = df_raw[['close']].rename(columns={'close': 'price'})
    else:
        # Use first numeric column
        num_cols = df_raw.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            df = df_raw[[num_cols[0]]].rename(columns={num_cols[0]: 'price'})
        else:
            raise ValueError("No numeric column found for price data.")

    df = df.sort_index()
    return df

def add_log_returns(df):
    """
    Add log returns column to a price DataFrame.
    """
    df['log_ret'] = np.log(df['price']).diff()
    return df