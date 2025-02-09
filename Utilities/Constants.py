class Constants:
    # Yahoo Finance API
    TICKER = "BTC-USD"
    START_DATE = "2018-02-01"
    INTERVAL = "1d"
    
    # File paths
    RAW_CSV_FILENAME = "csv/raw_bitcoin_data.csv"
    INDICATOR_CSV_FILENAME = "csv/indicator_bitcoin_data.csv"
    
    # URLs
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    FEAR_GREED_LIMIT = 10000
    
    # Model hyperparameters
    WINDOW_SIZE = 30
    BATCH_SIZE = 16
    SPLIT_INDEX = 0.95
    INPUT_DIM = 17
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    OUTPUT_DIM = 1
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.0001