import pandas as pd
import ta
from Constants import Constants

class BTCIndicatorCalculate:
    def __init__(self):
        self.raw_csv_filename = Constants.RAW_CSV_FILENAME
        self.indicator_csv_filename = Constants.INDICATOR_CSV_FILENAME
        self.data = pd.read_csv(self.raw_csv_filename, index_col='Date', parse_dates=True)

    def calculate_indicators(self):
        self.data['SMA5'] = ta.trend.sma_indicator(self.data['Close'], window=5)
        self.data['SMA20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        self.data['RSI14'] = ta.momentum.rsi(self.data['Close'], window=14)
        
        self.data['MACD'] = ta.trend.macd(self.data['Close'])
        self.data['MACD_Signal'] = ta.trend.macd_signal(self.data['Close'])
        self.data['MACD_Histogram'] = ta.trend.macd_diff(self.data['Close'])
        
        self.data['ATR14'] = ta.volatility.average_true_range(self.data['High'], self.data['Low'], self.data['Close'], window=14)
        
        self.data['BB_Middle'] = ta.volatility.bollinger_mavg(self.data['Close'], window=20)
        self.data['BB_Upper'] = ta.volatility.bollinger_hband(self.data['Close'], window=20)
        self.data['BB_Lower'] = ta.volatility.bollinger_lband(self.data['Close'], window=20)

        self.data = self.data.iloc[35:]

        self.save_to_csv()

    def save_to_csv(self):
        self.data.to_csv(self.indicator_csv_filename)
