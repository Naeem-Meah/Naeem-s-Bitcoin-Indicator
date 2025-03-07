# Bitcoin Price Direction Prediction Using LSTM and Technical Indicators

## 🚀 Overview
My repository contains a predictive machine learning model utilizing **Long Short-Term Memory (LSTM)** neural networks built with **PyTorch**. The model forecasts Bitcoin's (BTC-USD) next-day price direction ("long" or "short") by analysing historical OHLCV data (Open, High, Low, Close, Volume) combined with technical indicators and market sentiment metrics.

This model is simply a passion project and shouldn't be used for any real-time trading. Accuracy of the model varies.

An LSTM is a type of recurrent neural network (RNN) specifically designed to capture dependencies in sequential data by remembering information over extended time periods.

## 📈 Data Sources
- **Historical OHLCV Data**: Fetched from Yahoo Finance API.
- **Crypto Market Sentiment**: Fear and Greed Index retrieved via the [Alternative.me API](https://alternative.me/crypto/fear-and-greed-index/).

## 🛠️ Features Engineered
The model uses technical indicators and sentiment data, including:

- Simple Moving Averages (`SMA5`, `SMA20`)
- Relative Strength Index (`RSI14`)
- Moving Average Convergence Divergence (`MACD`, `MACD Signal`, `MACD Histogram`)
- Average True Range (`ATR14`)
- Bollinger Bands (`Upper`, `Middle`, `Lower`)
- Markov State Change
- Daily percentage price change
- Fear and Greed Index

## 🎯 Model Architecture
- **LSTM Neural Network**: Designed to capture temporal dependencies in Bitcoin price data.
- **Layers**: 4 stacked LSTM layers with 128 hidden units each, followed by a fully connected output layer with sigmoid activation.
- **Training Configuration**:
  - **Epochs**: 100
  - **Batch size**: 16
  - **Learning rate**: 0.0001
  - **Loss function**: Binary Cross Entropy (BCE)
  - **Optimizer**: Adam
 
Please note that these hyperparameters are experimental. I am currently working on tuning these for improved results.

## 🧪 Performance
The model achieves varied accuracy in predicting next-day price movements, demonstrating the potential of combining deep learning with traditional technical analysis. 

However, due to the randomness of current machine learning algorithms, this can vary greatly.

Current accuracy varies between 45% - 52%. Not much better than a coin flip.

## ⚙️ Usage

### 1️⃣ **Data Collection**
Automatically fetch historical data and sentiment indices. Generate technical indicators from raw data.
```bash
run the GetBTCData.py file
```

### 2️⃣ **Model Training & Evaluation**
Train the LSTM model and evaluate on a hold-out test set. Then generate prediction and accuracy.
```bash
run the LSTM.py file
```

## 📋 Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- ta
- scikit-learn
- requests

## 👎 Cons
- Doesn't work with live trading platforms for automated execution.
- Only takes into account technical analysis data and basic market sentiment,should include broader market conditions and macroeconomic data.
- Machine Learning hyperparameters are eyeballed and general practise in the Constants.py, and have no real optimised values.
- Innacurate model which is only slightly more advantageous than a coin toss.
- Lacks high-quality, cleaned and high-resolution data. Simply uses daily data and stores in a CSV file rather than KDB.

## 👍 Pros
- A cool demonstration on how Machine Learning models can give a slight edge by looking at past data.
- Includes market sentiment (Fear and Greed Index) as an additional dimension beyond pure price metrics.
- Easily adaptable for additional features and more complex data inputs in future iterations.
- Could be used in conjunction with other tools to indicate positions.

## 👨‍💻 Author
Developed by Naeem Meah, as a passion project in Algorithmic Trading.
