from celery import shared_task
from .models import DailyCryptoData, MinuteCryptoData,prediction_data
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
from django.utils import timezone
from .serializers import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.initializers import Orthogonal
import pytz
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def load_and_predict(model_path, data, time_steps, minutes):
    # Load the model
    print("checkpoint 1")
    custom_objects = {'Orthogonal': Orthogonal}

    # Load the model with the custom_objects parameter
    model = load_model(model_path)
    # model = load_model(model_path)
    print("checkpoint 2")
    # Scale the data
    print(data.shape,data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Predict next days
    predicted_prices = predict_next_minutes(model, scaled_data, scaler, time_steps, minutes)

    return predicted_prices

def predict_next_minutes(model, data, scaler, time_steps, minutes):
    # Copy the original data
    input_data = data.copy()

    # Initialize list to store predicted prices
    predicted_prices = []

    # Iterate over the number of days
    for _ in range(minutes):
        # Prepare the input data for prediction
        input_sequence = input_data[-time_steps:]
        input_sequence_scaled = scaler.transform(input_sequence.reshape(-1, 1))
        X_input = np.reshape(input_sequence_scaled, (1, time_steps, 1))

        # Predict the next value
        predicted_price = model.predict(X_input)

        # Inverse transform the predicted price
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # Append the predicted price to the list
        predicted_prices.append(predicted_price)

        # Append the predicted price to the input data and remove the oldest value
        input_data = np.append(input_data, predicted_price)
        input_data = input_data[-time_steps:]

    return predicted_prices

def generate_date_sequence(start_date, num_minutes):
    # Generate a sequence of timestamps starting from the given start_date
    return [(start_date + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(num_minutes)]

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
    predicted_prices=model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, predicted_prices))
    print(f'RMSE: {rmse}')

    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_train, label='Actual Values')
    plt.plot(predicted_prices, label='Predicted Values', color='red')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f"static/{datetime.strftime(datetime.now(),'%H_%M_%S')}.png")
    return rmse
   



@shared_task(bind=True)
def fetch_1_minute_data(self):
    interval="1m"
    limit=1
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    url = f'https://api.binance.us/api/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
    
    response = requests.get(url)
    # if response.status_code == 200:
        # data = json.loads(response.text)
        # for item in data:
        #    # item=clean_data(item)
        #     if len(item)==0:
        #         continue
        #     else:  
        #         if interval == '1m':
        #             MinuteCryptoData.objects.create(
        #                 start_interval_timestamp=item[0],
        #                 opening=item[1],
        #                 highest=item[2],
        #                 lowest=item[3],
        #                 closing=item[4],
        #                 volume=item[5],
        #                 end_interval_timestamp=item[6]
        #             )
                
        #         # try:
        #         #     print("start_interval_timestamp:",start_interval_timestamp,type(start_interval_timestamp))
        #         #     obj=prediction_data.objects.get(timestamp=start_interval_timestamp + timedelta(hours=5))
        #         #     obj.orignal_value=closing
        #         #     obj.save()
        #         # except Exception as e:
        #         #     print(f"Exception is {e}")
                

            

        # return data
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

def clean_data(item):
    # First Check Any NULL Value
    if any(value is None for value in item):
        return []

    # Check if the item has exactly 7 elements
    if len(item) != 7:
        return []

    try:
        # Conversion of data into the required format
        start_interval_timestamp = datetime.utcfromtimestamp(item[0] / 1000)
        opening = float(item[1])
        highest = float(item[2])
        lowest = float(item[3])
        closing = float(item[4])
        volume = float(item[5])
        end_interval_timestamp = datetime.utcfromtimestamp(item[6] / 1000)
    except (ValueError, TypeError) as e:
        # If conversion fails, return an empty list
        return []

    return [start_interval_timestamp, opening, highest, lowest, closing, volume, end_interval_timestamp]
@shared_task(bind=True)
def fetch_1_day_data(self):
    interval="1d"
    limit=10
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=10)
    start_timestamp = int(start_time.timestamp()) * 1000
    end_timestamp = int(end_time.timestamp()) * 1000
    print("starttime:",start_time,"endtime:",end_time)
    url = f'{base_url}?symbol={symbol}&interval={interval}&startTime={start_timestamp}&endTime={end_timestamp}&limit={limit}'
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = json.loads(response.text)
        for item in data:
            item=clean_data(item)
            if len(item)==0:
                continue
            else:  
                if interval == '1d':
                    DailyCryptoData.objects.create(
                        start_interval_timestamp=item[0],
                        opening=item[1],
                        highest=item[2],
                        lowest=item[3],
                        closing=item[4],
                        volume=item[5],
                        end_interval_timestamp=item[6]
                    )
            
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
    
@shared_task(bind=True)
def minute_prediction(self):
    minutes=1
    end_date = timezone.now()
    start_date = end_date - timedelta(minutes=90)  # Previous 90 minutes
    print(start_date,end_date)
    # daily_data = DailyCryptoData.objects.filter(start_interval_timestamp__range=(start_date, end_date)).order_by('start_interval_timestamp')
    daily_data=MinuteCryptoData.objects.all().order_by('-start_interval_timestamp')[:90]
    print(daily_data)
    # Serialize the data
    serializer = DailyCryptoDataSerializer(daily_data, many=True)

    # Extract the relevant feature from the serialized data (e.g., closing prices)
    data = [item['closing'] for item in serializer.data]
    data=np.array(data)
    # Define the path to your model
    model_path = 'static/cryptocurrency_lstm_model_new.keras'  # Replace with your actual model path

    # Define the time steps (assuming you know this value)
    time_steps = 10  # Replace with your actual time steps

    # Call the load_and_predict function to generate predictions
    predicted_prices = load_and_predict(model_path, data, time_steps, minutes)
    print(minutes)
    end_time = datetime.now()
    # Generate time sequence for the previous 90 minutes
    previous_dates = generate_date_sequence(end_time - timedelta(minutes=len(data)), len(data))
    next_dates = generate_date_sequence(end_time , minutes)
    print(predicted_prices[0])
    if minutes==1:
        # Get the previous instance and update it
        try:
            previous_predict = prediction_data.objects.order_by("-timestamp").first()
            previous_predict.orignal_value = MinuteCryptoData.objects.all().order_by('-start_interval_timestamp').first().closing
            previous_predict.save()
            print("Previous updated")
        except Exception as e:
            print(f"Exception is: {e}")
        # Create a new instance and save it
        current_date = datetime.combine(datetime.now().date(), datetime.strptime(next_dates[0], '%H:%M:%S').time()).replace(second=0)
        new_predict = prediction_data(
            timestamp=current_date,
            predicted_value=float(predicted_prices[0])
        )
        new_predict.save()
        print("New created",new_predict.timestamp)
    # print("next_dates: ", next_dates)
    # # Combine dates with predicted prices
    # previous_data = list(zip(previous_dates, data))
    # next_data = list(zip(next_dates, predicted_prices))

    # # Combine previous and next data lists
    # combined_data = previous_data + next_data