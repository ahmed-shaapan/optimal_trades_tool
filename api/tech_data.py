from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from stock_indicators import indicators
from stock_indicators.indicators.common import Quote
import os
import time
from typing import List, Dict, Any
import warnings
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.dialects.postgresql import insert as pg_insert

warnings.filterwarnings('ignore')

# --- Database Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define the table structure
stock_data_table = Table('stock_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String, nullable=False),
    Column('Date', DateTime, nullable=False),
    Column('Open', Float),
    Column('High', Float),
    Column('Low', Float),
    Column('Close', Float),
    Column('Volume', Float),
    Column('Dividends', Float),
    Column('Stock Splits', Float),
    Column('EMA_20', Float),
    Column('bb_upper', Float),
    Column('bb_middle', Float),
    Column('bb_lower', Float),
    Column('bb_percent_b', Float),
    Column('bb_z_score', Float),
    Column('bb_width', Float),
    Column('rsi', Float),
    Column('macd_line', Float),
    Column('macd_signal', Float),
    Column('macd_histogram', Float),
    Column('stoch_k', Float),
    Column('stoch_d', Float),
    Column('ichimoku_tenkan', Float),
    Column('ichimoku_kijun', Float),
    Column('ichimoku_senkou_a', Float),
    Column('ichimoku_senkou_b', Float),
    Column('ichimoku_chikou', Float),
    Column('fib_0', Float),
    Column('fib_236', Float),
    Column('fib_382', Float),
    Column('fib_500', Float),
    Column('fib_618', Float),
    Column('fib_786', Float),
    Column('fib_100', Float),
    Column('adx', Float),
    Column('pdi', Float),
    Column('mdi', Float),
    Column('psar', Float),
    Column('psar_is_reversal', Boolean),
    Column('donchian_upper', Float),
    Column('donchian_lower', Float),
    Column('donchian_center', Float),
    Column('donchian_width', Float),
    Column('roc', Float),
    Column('elliott_wave_oscillator', Float),
    Column('doji', Boolean),
    Column('hammer', Boolean),
    Column('shooting_star', Boolean),
    Column('engulfing_bullish', Boolean),
    Column('engulfing_bearish', Boolean),
)


def create_table_if_not_exists():
    """Create the stock_data table if it doesn't exist."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    inspector = inspect(engine)
    if not inspector.has_table('stock_data'):
        print("Creating 'stock_data' table...")
        metadata.create_all(engine)
        print("Table created.")
    else:
        print("'stock_data' table already exists.")

# --- End Database Setup ---

nasdaq100__symbols = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "AVGO", "GOOGL", "GOOG", "TSLA", "ADBE",
    "COST", "PEP", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "CMCSA", "TXN", "AMGN",
    "HON", "INTU", "QCOM", "SBUX", "BKNG", "MDLZ", "ISRG", "ADP", "GILD", "ADI",
    "REGN", "VRTX", "AMAT", "LRCX", "MU", "CSX", "PANW", "MAR", "ASML", "SNPS",
    "MNST", "CDNS", "ORLY", "AEP", "FTNT", "MELI", "KDP", "CHTR", "EXC", "KLAC",
    "CTAS", "PAYX", "MCHP", "LULU", "WDAY", "DXCM", "ROST", "BIIB", "IDXX", "EA",
    "KHC", "PCAR", "CRWD", "AZN", "FAST", "XEL", "BKR", "TEAM", "CPRT", "MRVL",
    "WBD", "FANG", "ADSK", "DDOG", "ILMN", "WBA", "ODFL", "CEG", "VRSK", "SIRI",
    "ABNB", "CTSH", "ALGN", "PDD", "PYPL", "ANSS", "EBAY", "JD", "ZS", "MRNA",
    "TTD", "GFS", "CDW", "DLTR", "NXPI", "ON", "GEHC", "ARM", "DASH", "ROP", "LI", "QQQ"
]

class StockDataDownloader:

    def __init__(self):
        self.stocks = nasdaq100__symbols
        self.end_date = datetime.now().strftime('%Y-%m-%d')

    def get_latest_date_from_db(self, symbol: str) -> str:
        """Get the latest date for a symbol from the database."""
        with engine.connect() as connection:
            query = text('SELECT MAX("Date") FROM stock_data WHERE symbol = :symbol')
            result = connection.execute(query, {'symbol': symbol}).scalar()
            if result:
                return (result + timedelta(days=1)).strftime('%Y-%m-%d')
            return "2019-01-01"

    def download_stock_data(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Download stock data for a given symbol"""
        try:
            print(f"Downloading data for {symbol} from {start_date} to {self.end_date}...")
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=self.end_date)

            if data.empty:
                print(f"No new data found for {symbol}")
                return None

            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            return data

        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
            return None

    def get_full_history_for_indicators(self, symbol: str) -> List[Quote]:
        """Fetches the full history for a symbol to ensure accurate indicator calculation."""
        print(f"Fetching full history for {symbol} to calculate indicators accurately...")
        with engine.connect() as connection:
            query = text('SELECT "Date", "Open", "High", "Low", "Close", "Volume" FROM stock_data WHERE symbol = :symbol ORDER BY "Date" ASC')
            result = connection.execute(query, {'symbol': symbol}).fetchall()

        quotes = [
            Quote(date=row[0], open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5])
            for row in result
        ]
        return quotes

    def convert_to_quotes(self, df: pd.DataFrame) -> List[Quote]:
        """Convert pandas DataFrame to Quote objects for stock-indicators library"""
        quotes = []
        for _, row in df.iterrows():
            quote = Quote(
                date=row['Date'],
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            )
            quotes.append(quote)
        return quotes

    def calculate_all_indicators(self, quotes: List[Quote]) -> Dict:
        """Calculate all technical indicators"""
        print("Calculating technical indicators...")
        indicators_data = {}
        
        # A helper to run and catch errors for each indicator function
        def run_indicator(func, *args, **kwargs):
            try:
                return func(quotes, *args, **kwargs)
            except Exception as e:
                print(f"Error calculating {func.__name__}: {e}")
                return {}

        indicators_data.update(run_indicator(indicators.get_ema, 20))
        indicators_data.update(run_indicator(indicators.get_bollinger_bands, 20, 2.0))
        # ... Add other indicator calculations here in the same way ...

        return indicators_data

    def save_data_to_db(self, symbol: str, df: pd.DataFrame, indicators_data: Dict):
        """Save stock data and indicators to the database."""
        try:
            print(f"Saving new data for {symbol} to the database...")
            combined_df = df.copy()

            for indicator_name, values in indicators_data.items():
                if len(values) > 0:
                    combined_df[indicator_name] = values[-len(df):]
                else:
                    combined_df[indicator_name] = None

            for col in stock_data_table.columns.keys():
                if col not in combined_df.columns and col not in ['id', 'symbol']:
                    combined_df[col] = None
            
            combined_df['symbol'] = symbol

            records_to_insert = combined_df.to_dict(orient='records')

            if not records_to_insert:
                print("No new records to save.")
                return

            with engine.connect() as connection:
                with connection.begin():
                    for record in records_to_insert:
                        record_for_insert = {key: record.get(key) for key in stock_data_table.columns.keys() if key != 'id'}
                        stmt = pg_insert(stock_data_table).values(**record_for_insert)
                        update_dict = {c.name: c for c in stmt.excluded if c.name not in ['id', 'symbol', 'Date']}
                        stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'Date'], set_=update_dict)
                        connection.execute(stmt)

            print(f"Successfully saved {len(records_to_insert)} new records for {symbol}.")

        except Exception as e:
            print(f"Error saving data to DB for {symbol}: {e}")

    def process_all_stocks(self):
        """Process all stocks and calculate indicators"""
        print(f"Starting to process {len(self.stocks)} stocks...")

        for i, symbol in enumerate(self.stocks, 1):
            print(f"--- Processing {symbol} ({i}/{len(self.stocks)}) ---")
            try:
                start_date = self.get_latest_date_from_db(symbol)
                new_df = self.download_stock_data(symbol, start_date)

                if new_df is None or new_df.empty:
                    print(f"No new data for {symbol}. Skipping.")
                    continue

                historical_quotes = self.get_full_history_for_indicators(symbol)
                new_quotes = self.convert_to_quotes(new_df)
                all_quotes = historical_quotes + new_quotes

                if not all_quotes:
                    continue

                indicators_data = self.calculate_all_indicators(all_quotes)
                self.save_data_to_db(symbol, new_df, indicators_data)
                time.sleep(1)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        print("\n=== Processing Complete ===")
        print(f"Database is now up-to-date.")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            print("Starting database update process...")
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable is not set.")
            
            create_table_if_not_exists()
            downloader = StockDataDownloader()
            downloader.process_all_stocks()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'message': 'Database update completed successfully.'}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"An error occurred: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'error', 'message': str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))
