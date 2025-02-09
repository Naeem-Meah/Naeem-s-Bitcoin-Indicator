import yfinance as yf
import pandas as pd
from Constants import Constants

class BitcoinOHLCVData:
    def __init__(self):
        self.ticker = Constants.TICKER
        self.startDate = Constants.START_DATE
        self.interval = Constants.INTERVAL
        self.fetchedData = None

    def fetch_data(self):
        btc = yf.Ticker(self.ticker)
        data = btc.history(start=self.startDate, interval=self.interval)
        self.fetchedData = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.fetchedData.index = self.fetchedData.index.strftime('%Y-%m-%d')
        self.fetchedData.index.name = 'Date'
        self.fetchedData.index = pd.to_datetime(self.fetchedData.index)  # Ensure the index is of datetime type
        return self.fetchedData