from dotenv import load_dotenv
load_dotenv() # This line loads variables from the .env file
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
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, UniqueConstraint

from sqlalchemy.dialects.postgresql import insert as pg_insert

warnings.filterwarnings('ignore')

# --- Database Setup ---
DATABASE_URL = os.environ.get('DATABASE_URL')

# Fix for deprecated postgres:// URL format
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

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
    # Add unique constraint on symbol and Date combination
    UniqueConstraint('symbol', 'Date', name='uq_symbol_date')
)

def drop_table_if_exists():
    """Drop the stock_data table if it exists."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    try:
        with engine.connect() as connection:
            with connection.begin():
                # Drop the table
                connection.execute(text("DROP TABLE IF EXISTS stock_data"))
                print("Table 'stock_data' dropped successfully.")
    except Exception as e:
        print(f"Error dropping table: {e}")

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
                result = func(quotes, *args, **kwargs)
                if hasattr(result, '__iter__') and not isinstance(result, str):
                    # Convert result to dictionary format expected by the rest of the code
                    if hasattr(result[0], '__dict__'):
                        # Handle objects with attributes
                        result_dict = {}
                        for attr in dir(result[0]):
                            if not attr.startswith('_'):
                                try:
                                    result_dict[attr] = [getattr(item, attr) for item in result]
                                except:
                                    continue
                        return result_dict
                    else:
                        return {'value': result}
                return result
            except Exception as e:
                print(f"Error calculating {func.__name__}: {e}")
                return {}

        # Calculate indicators - you'll need to implement the actual calculations
        try:
            # EMA
            ema_20 = indicators.get_ema(quotes, 20)
            if ema_20:
                indicators_data['EMA_20'] = [item.ema for item in ema_20]

            # Bollinger Bands
            bb = indicators.get_bollinger_bands(quotes, 20, 2.0)
            if bb:
                indicators_data['bb_upper'] = [item.upper_band for item in bb]
                indicators_data['bb_middle'] = [item.sma for item in bb]
                indicators_data['bb_lower'] = [item.lower_band for item in bb]
                indicators_data['bb_percent_b'] = [item.percent_b for item in bb]
                indicators_data['bb_z_score'] = [item.z_score for item in bb]
                indicators_data['bb_width'] = [item.width for item in bb]

            # RSI
            rsi = indicators.get_rsi(quotes, 14)
            if rsi:
                indicators_data['rsi'] = [item.rsi for item in rsi]

            # MACD
            macd = indicators.get_macd(quotes, 12, 26, 9)
            if macd:
                indicators_data['macd_line'] = [item.macd for item in macd]
                indicators_data['macd_signal'] = [item.signal for item in macd]
                indicators_data['macd_histogram'] = [item.histogram for item in macd]

            # Stochastic
            stoch = indicators.get_stoch(quotes, 14, 3, 3)
            if stoch:
                indicators_data['stoch_k'] = [item.k for item in stoch]
                indicators_data['stoch_d'] = [item.d for item in stoch]

            # Ichimoku
            ichimoku = indicators.get_ichimoku(quotes, 9, 26, 52)
            if ichimoku:
                indicators_data['ichimoku_tenkan'] = [item.tenkan_sen for item in ichimoku]
                indicators_data['ichimoku_kijun'] = [item.kijun_sen for item in ichimoku]
                indicators_data['ichimoku_senkou_a'] = [item.senkou_span_a for item in ichimoku]
                indicators_data['ichimoku_senkou_b'] = [item.senkou_span_b for item in ichimoku]
                indicators_data['ichimoku_chikou'] = [item.chikou_span for item in ichimoku]

            # ADX
            adx = indicators.get_adx(quotes, 14)
            if adx:
                indicators_data['adx'] = [item.adx for item in adx]
                indicators_data['pdi'] = [item.pdi for item in adx]
                indicators_data['mdi'] = [item.mdi for item in adx]

            # Parabolic SAR
            psar = indicators.get_parabolic_sar(quotes, 0.02, 0.2)
            if psar:
                indicators_data['psar'] = [item.sar for item in psar]
                indicators_data['psar_is_reversal'] = [item.is_reversal for item in psar]

            # Donchian Channels
            donchian = indicators.get_donchian(quotes, 20)
            if donchian:
                indicators_data['donchian_upper'] = [item.upper_band for item in donchian]
                indicators_data['donchian_lower'] = [item.lower_band for item in donchian]
                indicators_data['donchian_center'] = [item.center_line for item in donchian]
                indicators_data['donchian_width'] = [item.width for item in donchian]

            # Rate of Change
            roc = indicators.get_roc(quotes, 20)
            if roc:
                indicators_data['roc'] = [item.roc for item in roc]

            # Elliott Wave Oscillator (using EMA difference as approximation)
            ema_5 = indicators.get_ema(quotes, 5)
            ema_35 = indicators.get_ema(quotes, 35)
            if ema_5 and ema_35:
                ewo_values = []
                for i in range(len(quotes)):
                    if i < len(ema_5) and i < len(ema_35) and ema_5[i].ema and ema_35[i].ema:
                        ewo_values.append(ema_5[i].ema - ema_35[i].ema)
                    else:
                        ewo_values.append(None)
                indicators_data['elliott_wave_oscillator'] = ewo_values

        except Exception as e:
            print(f"Error in indicator calculations: {e}")

        return indicators_data

    def save_data_to_db(self, symbol: str, df: pd.DataFrame, indicators_data: Dict):
        """Save stock data and indicators to the database."""
        try:
            print(f"Saving new data for {symbol} to the database...")
            combined_df = df.copy()

            # Add indicator data to dataframe
            for indicator_name, values in indicators_data.items():
                if values and len(values) > 0:
                    # Pad with None values if indicator data is shorter than df
                    if len(values) < len(df):
                        padded_values = [None] * (len(df) - len(values)) + values
                        combined_df[indicator_name] = padded_values
                    else:
                        # Take the last len(df) values if indicator data is longer
                        combined_df[indicator_name] = values[-len(df):]
                else:
                    combined_df[indicator_name] = None

            # Ensure all required columns exist
            for col in stock_data_table.columns.keys():
                if col not in combined_df.columns and col not in ['id', 'symbol']:
                    combined_df[col] = None
            
            combined_df['symbol'] = symbol

            # Rename columns to match database schema
            if 'Stock Splits' in combined_df.columns:
                combined_df = combined_df.rename(columns={'Stock Splits': 'Stock_Splits'})

            records_to_insert = combined_df.to_dict(orient='records')

            if not records_to_insert:
                print("No new records to save.")
                return

            with engine.connect() as connection:
                with connection.begin():
                    for record in records_to_insert:
                        # Prepare record for insertion
                        record_for_insert = {}
                        for key in stock_data_table.columns.keys():
                            if key != 'id':
                                if key == 'Stock_Splits' and 'Stock Splits' in record:
                                    record_for_insert[key] = record['Stock Splits']
                                else:
                                    record_for_insert[key] = record.get(key)
                        
                        stmt = pg_insert(stock_data_table).values(**record_for_insert)
                        update_dict = {c.name: c for c in stmt.excluded if c.name not in ['id', 'symbol', 'Date']}
                        stmt = stmt.on_conflict_do_update(index_elements=['symbol', 'Date'], set_=update_dict)
                        connection.execute(stmt)

            print(f"Successfully saved {len(records_to_insert)} new records for {symbol}.")

        except Exception as e:
            print(f"Error saving data to DB for {symbol}: {e}")
            import traceback
            traceback.print_exc()

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
                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n=== Processing Complete ===")
        print(f"Database is now up-to-date.")

if __name__ == "__main__":
    try:
        print("Starting database update process...")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set. Make sure you have a .env file.")
        
        create_table_if_not_exists()
        downloader = StockDataDownloader()
        downloader.process_all_stocks()
        # drop_table_if_exists()
    except Exception as e:
        print(f"An error occurred during the script execution: {e}")
        import traceback
        traceback.print_exc()