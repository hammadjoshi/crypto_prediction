from django.http    import HttpResponse
from django.shortcuts import render
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.pyplot.switch_backend('Agg')
import matplotlib.pyplot as plt
import requests
import json


from django.http    import HttpResponse
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import *

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as web
import datetime as dt




def index(request):
    return HttpResponse("Hello")
def getAllCryptoNames(request):
    return HttpResponse("All crypto names")

def currentValues(request):
    def get_cryptocurrency_data(symbol, interval='15m', limit=10):
        url = f'https://api.binance.com/api/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
        response = requests.get(url)
        if response.status_code == 200:
            data = json.loads(response.text)
           # return data
            print('dat----',data)
            return HttpResponse("current values of BTC, ETH, ADA, DOT",response,response.status_code)

        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return HttpResponse("current values of BTC, ETH, ADA, DOT",'in else')

    # Example usage
    symbol = 'BTCUSDT'  # Example: Bitcoin/USDT pair
    data = get_cryptocurrency_data(symbol)
    if data:
        print("One-minute interval cryptocurrency data:")
        return HttpResponse("current values of BTC, ETH, ADA, DOT   ")

    #   print(data)









   # BTC_Ticker = yf.Ticker("BTC-AUD")
    #BTC_Data = BTC_Ticker.history(period="max")
    #feature = []
    #for col in BTC_Data.columns.values:
     #   feature.append(col)
    #print("Features----->>>",feature)
    #print("BTC Close---->", BTC_Data['Close'])
    #print(BTC_Data)


   # Define the cryptocurrency symbol (e.g., Bitcoin - BTC-USD)
  # symbol = 'BTC-AUD'

   # Fetch cryptocurrency data using yfinance
   #crypto_data = yf.download(symbol, start='2022-01-01', end='2023-01-01')
   #print(crypto_data)
   # Plot the 'Close' price of the cryptocurrency
   #plt.figure(figsize=(10, 6))
   #plt.plot(crypto_data['Close'], label=f'{symbol} Close Price')

   #plt.xlabel('Date')
   #plt.ylabel('Price')
   #plt.title('Coin Price')
   #plt.legend()
   #plt.grid(True)
   #plt.show()
   #print("baga is here at my lest side")


 #  x = [1, 5, 1.5, 4]
  # y = [9, 1.8, 8, 11]
   #plt.scatter(x, y)
   #plt.show()


def getBTCValue(request):
    get_btc_yfinance()
    return HttpResponse("BTC HERE")


# def get_historical_data(symbol, interval='1d', limit=720):
#     base_url = 'https://api.binance.com/api/v1/klines'
#     end_time = dt.now()
#     start_time = end_time - timedelta(days=365)
#     start_timestamp = int(start_time.timestamp()) * 1000
#     end_timestamp = int(end_time.timestamp()) * 1000
#     url = f'{base_url}?symbol={symbol}&interval={interval}&startTime={start_timestamp}&endTime={end_timestamp}&limit={limit}'
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = json.loads(response.text)
#         return data
#     else:
#         print(f"Failed to fetch data. Status code: {response.status_code}")
#         return None

# symbol = 'BTCUSDT'
# data = get_historical_data(symbol)
# if data:
#     print("Historical data for BTC/USDT:")
#     print(data)


def get_btc_yfinance():
    print('here3')

    BTC_Ticker = yf.Ticker("BTC-USD")
    BTC_Data = BTC_Ticker.history(period="max")
    feature = []
    for col in BTC_Data.columns.values:
        feature.append(col)

    print("BTC Close---->", BTC_Data['Close'])
    print(feature)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(BTC_Data['Close'].values.reshape(-1, 1))

    prediction_days = 2
    x_train, y_train = [], []

    print(scaled_data)

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    print("End")
    # model.add(Dropout(0.2))









