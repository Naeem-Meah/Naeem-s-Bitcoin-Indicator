from BTCDataRetrieve import BTCDataRetrieve
from BTCIndicatorCalculate import BTCIndicatorCalculate

# Fetch and save the Bitcoin OHLCV data along with Fear and Greed index into a CSV file
btc_data_retrieve = BTCDataRetrieve()
btc_data_retrieve.fetch_and_save_data()

# Calculate technical indicators and save the processed data to a new CSV file
btc_indicator_calculate = BTCIndicatorCalculate()
btc_indicator_calculate.calculate_indicators()