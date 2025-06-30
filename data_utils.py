import pandas as pd
import os

def load_stock_data(data_dir):
    """
    Loads all stock data from CSV files in the specified directory,
    filters it for the required date range, adds a ticker symbol,
    and excludes the 'QQQ' benchmark ticker.

    Args:
        data_dir (str): The directory containing the stock data CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and filtered stock data.
    """
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            ticker = filename.split('_')[0]
            if ticker.upper() == 'QQQ':
                continue  # Skip the benchmark ticker
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            df['ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2025)]
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_benchmark_data(data_dir, benchmark_ticker='QQQ'):
    """
    Loads the benchmark data (e.g., QQQ) from a CSV file.

    Args:
        data_dir (str): The directory containing the stock data CSV files.
        benchmark_ticker (str): The ticker symbol for the benchmark.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark data.
    """
    for filename in os.listdir(data_dir):
        if filename.startswith(benchmark_ticker):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            return df
    return None
