import pandas as pd
import yfinance as yf
from ta.momentum import rsi
from ta.trend import MACD

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data, handles MultiIndex columns, and adds indicators.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker} from {start_date} to {end_date}.")
            return pd.DataFrame()
        
        # --- THE FIX: Handle MultiIndex Columns ---
        # If the columns are a MultiIndex, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1) # Drop the 'Ticker' level

        # Now, force all column names to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Calculate indicators using the standardized 'close' column
        data['rsi'] = rsi(data['close'], window=14)
        macd_indicator = MACD(data['close'])
        data['macd'] = macd_indicator.macd()
        data['macd_signal'] = macd_indicator.macd_signal()
        data['macd_hist'] = macd_indicator.macd_diff()
        
        data.dropna(inplace=True)
        
        print(f"Successfully downloaded and prepared data for {ticker}.")
        return data
    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    spy_data = fetch_data(ticker='SPY', start_date='2020-01-01', end_date='2023-12-31')
    if not spy_data.empty:
        print("\nSPY Data Head:")
        print(spy_data.head())