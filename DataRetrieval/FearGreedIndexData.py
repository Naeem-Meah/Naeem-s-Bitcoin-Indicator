import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from Utilities.Constants import Constants

class FearGreedIndexData:
    def __init__(self):
        self.url = f"{Constants.FEAR_GREED_URL}?limit={Constants.FEAR_GREED_LIMIT}"
        self.data = None

    def fetch_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            self.data = response.json()['data']
            formatted_data = [
                {
                    'date': datetime.fromtimestamp(int(item['timestamp']), timezone.utc).strftime('%Y-%m-%d'),
                    'FearGreedIndex': int(item['value'])
                }
                for item in self.data
            ]
            fear_greed_df = pd.DataFrame(formatted_data)
            fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
            fear_greed_df.set_index('date', inplace=True)
            print("Fear and Greed Index Data:")
            print(fear_greed_df.head())
            return fear_greed_df
        else:
            print("Failed to fetch Fear and Greed Index data.")
            return None