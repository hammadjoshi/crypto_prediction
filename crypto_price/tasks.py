from celery import shared_task
from .models import DailyCryptoData, MinuteCryptoData
import requests
from datetime import datetime,timedelta
# def get_historical_data(symbol, interval='1d', limit=720):
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import json
from keras.initializers import GlorotUniform
import os
@shared_task(bind=True)
def train_data(self):
    closing_data = MinuteCryptoData.objects.values_list('closing', flat=True)

    # Convert the queryset to a list and then to a NumPy array
    closing_data = list(closing_data)
    closing_data = np.array(closing_data)

    # Normalize the closing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_closing_data = scaler.fit_transform(closing_data.reshape(-1, 1))

    # Define the number of time steps to consider (e.g., past 60 minutes)
    time_steps = 60

    # Prepare the training data
    X_train, y_train = [], []
    for i in range(time_steps, len(scaled_closing_data)):
        X_train.append(scaled_closing_data[i - time_steps:i, 0])
        y_train.append(scaled_closing_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for LSTM input [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_initializer=GlorotUniform()))
    model.add(LSTM(units=50, kernel_initializer=GlorotUniform()))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    if os.path.exists("static/cryptocurrency_lstm_model_new.keras"):
        os.remove("static/cryptocurrency_lstm_model_new.keras")
    # Save the trained model
    history=model.save('static/cryptocurrency_lstm_model_new.keras')
@shared_task(bind=True)
def fetch_1_minute_data(self):
    interval="1m"
    limit=1
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    url = f'https://api.binance.us/api/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'\
    
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        for item in data:
            start_interval_timestamp = datetime.utcfromtimestamp(item[0] / 1000)
            opening = float(item[1])
            highest = float(item[2])
            lowest = float(item[3])
            closing = float(item[4])
            volume = float(item[5])
            end_interval_timestamp = datetime.utcfromtimestamp(item[6] / 1000)

            if interval == '1m':
                MinuteCryptoData.objects.create(
                    start_interval_timestamp=start_interval_timestamp,
                    opening=opening,
                    highest=highest,
                    lowest=lowest,
                    closing=closing,
                    volume=volume,
                    end_interval_timestamp=end_interval_timestamp
                )

            

        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
   

@shared_task(bind=True)
def fetch_1_day_data(self):
    interval="1d"
    limit=1
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    start_timestamp = int(start_time.timestamp()) * 1000
    end_timestamp = int(end_time.timestamp()) * 1000
    url = f'{base_url}?symbol={symbol}&interval={interval}&startTime={start_timestamp}&endTime={end_timestamp}&limit={limit}'
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        for item in data:
            start_interval_timestamp = datetime.utcfromtimestamp(item[0] / 1000)
            opening = float(item[1])
            highest = float(item[2])
            lowest = float(item[3])
            closing = float(item[4])
            volume = float(item[5])
            end_interval_timestamp = datetime.utcfromtimestamp(item[6] / 1000)

            if interval == '1d':
                DailyCryptoData.objects.create(
                    start_interval_timestamp=start_interval_timestamp,
                    opening=opening,
                    highest=highest,
                    lowest=lowest,
                    closing=closing,
                    volume=volume,
                    end_interval_timestamp=end_interval_timestamp
                )
            
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
    