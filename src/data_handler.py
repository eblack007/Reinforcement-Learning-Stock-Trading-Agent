import pandas as pd
import yfinance as yf
import logging

# Configure basic logging to show INFO level messages and a timestamp.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'SPY').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data, or an empty
                      DataFrame if the download fails.
    """
    try:
        # Disable the yfinance progress bar for cleaner log output
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logging.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
            return pd.DataFrame()
        logging.info(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        logging.error(f"An error occurred while downloading data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example of how to use the function
    spy_data = fetch_data(ticker='SPY', start_date='2020-01-01', end_date='2023-12-31')
    if not spy_data.empty:
        print("\nSPY Data Head:")
        print(spy_data.head())