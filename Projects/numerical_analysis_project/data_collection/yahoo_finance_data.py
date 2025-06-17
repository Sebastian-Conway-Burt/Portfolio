import yfinance as yf
import pandas as pd
import os

def collect_option_data(ticker_symbol, data_dir="data/raw"):
    """
    Collects call option data from Yahoo Finance for all available expiration dates
    for a given ticker symbol and saves it to CSV files.

    Args:
        ticker_symbol (str): The ticker symbol of the underlying asset (e.g., "AAPL").
        data_dir (str): The directory to save the CSV files. Defaults to "data/raw".
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    asset = yf.Ticker(ticker_symbol)
    expirations = asset.options

    if not expirations:
        print(f"No option data found for ticker {ticker_symbol}.")
        return

    for expiration in expirations:
        try:
            option_chain = asset.option_chain(expiration)
            call_options = option_chain.calls

            if not call_options.empty:
                expiry_date_str = pd.to_datetime(expiration).strftime('%Y-%m-%d')
                filename = os.path.join(data_dir, f"{ticker_symbol}_call_options_{expiry_date_str}.csv")
                call_options.to_csv(filename, index=False)
            else:
                print(f"No call option data found for {ticker_symbol} with expiry {expiration}.")

        except Exception as e:
            print(f"An error occurred while processing {ticker_symbol} for {expiration}: {e}")
    print(f"Option data collection for {ticker_symbol} completed.")         

if __name__ == "__main__":
    # Example usage:
    TICKERS = [
    # Your Originals:
    'AAPL', 'MSFT', 'GOOG', 'SPY', 'V','TSLA', 'NVDA', 'AMD', 'T', 'JPM',
    'MA', 'NFLX', 'COST', 'UNH', 'JNJ', 'CSCO', 'CRM',
    # Major Index ETFs:
    'QQQ', 'IWM', 'DIA',
    # S&P 100 / Large Cap Additions (various sectors):
    'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'XOM', 'WMT', 'HD',
    'PG', 'ABBV', 'CVX', 'MRK', 'KO', 'PEP', 'ADBE', 'BAC', 'MCD', 'CSCO', # CSCO repeated - OK
    'ACN', 'WFC', 'LIN', 'DIS', 'INTC', 'VZ', 'ABT', 'IBM', 'ORCL', 'NEE',
    'PM', 'CMCSA', 'NKE', 'HON', 'UPS', 'TXN', 'PFE', 'BMY', 'AMGN', 'RTX', # Raytheon (formerly UTX)
    'SBUX', 'CAT', 'GS', 'BLK', 'LOW', 'TMO', 'AXP', 'PYPL', # PayPal added
    'BA', 'ELV', 'LMT', 'COP', 'DE', 'GE', 'MMM', 'ADP', 'CVS', 'MDLZ',
    'SO', 'GILD', 'TJX', 'MO', 'CL', 'DUK', 'SCHW', 'USB', 'MS', 'CI',
    'ANTM', 'CME', 'ETN', 'FISV', 'NOW', # ServiceNow added
    # Other Large/Relevant Tickers:
    'PYPL', 'INTU', 'ISRG', 'BKNG', 'CAT', # Some repeated from S&P100 section - OK
    'UBER', 'ZM', 'SNOW', 'PLTR', 'SQ', 'SHOP', 'ETSY', 'PINS', 'RBLX',
    'FDX', 'GM', 'F', 'DAL', 'UAL', 'AAL', 'BA', # Boeing repeated - OK
    'PNC', 'GS', 'COF', 'USB', # Some banks repeated - OK
    'PFE', 'MRK', 'BMY', 'LLY', 'JNJ', 'ABBV', 'GILD', 'AMGN', # Some Pharma repeated - OK
    'OXY', 'SLB', 'HAL', 'EOG', 'DVN', # Energy
    'FCX', 'NEM', # Materials
    'AMT', 'PLD', 'EQIX', # REITs
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ' # Communication Services repeated - OK
]
    for ticker in TICKERS:
        collect_option_data(ticker)
    print("Data collection complete for all available expirations.")