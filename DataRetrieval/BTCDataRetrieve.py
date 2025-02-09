from DataRetrieval.BitcoinOHLCVData import BitcoinOHLCVData
from DataRetrieval.FearGreedIndexData import FearGreedIndexData
from Utilities.Constants import Constants

class BTCDataRetrieve:
    def __init__(self):
        self.csv_filename = Constants.RAW_CSV_FILENAME

    def fetch_and_save_data(self):
        # Fetch Bitcoin OHLCV data
        bitcoin_ohlcv_data = BitcoinOHLCVData()
        ohlcv_data = bitcoin_ohlcv_data.fetch_data()

        # Fetch Fear and Greed Index data
        fear_greed_data = FearGreedIndexData()
        fear_greed_index = fear_greed_data.fetch_data()

        # Combine the data
        raw_data = self.combine_data(ohlcv_data, fear_greed_index)
        print(raw_data.head())

        # Save the raw data to a CSV file
        self.save_to_csv(raw_data)

    def combine_data(self, ohlcv_data, fear_greed_index):
        # Merge the data on the Date column
        combined_data = ohlcv_data.merge(fear_greed_index, left_index=True, right_index=True, how='left')
        return combined_data

    def save_to_csv(self, data):
        if data is not None:
            data.to_csv(self.csv_filename, index_label='Date')
        else:
            print("No data to save.")
