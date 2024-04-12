
from .serializers import  MinuteCryptoDataSerializer, DailyCryptoDataSerializer
from datetime import datetime, timedelta,date

from .models import DailyCryptoData, MinuteCryptoData
import json
import requests
from .tasks import *

#fetch_1_minute_data()

from datetime import datetime, timedelta,date
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from django.utils import timezone
from keras.initializers import Orthogonal


import numpy as np



from django.http    import HttpResponse


def home(request):
    return HttpResponse("I'm prediction app'")

def celeryTest(request):
    fetch_1_minute_data()
    return HttpResponse("I'm celery Test'")

def ajass(request):
    return HttpResponse("I'm prediction app ajajajak'")



def minuteDataView(APIView):
        # Query MinuteCryptoData for the most recent 1-minute data
        minute_data = MinuteCryptoData.objects.all().order_by('-start_interval_timestamp')

        # Serialize the data
        serializer = MinuteCryptoDataSerializer(minute_data, many=True)
        print(len(serializer.data))
        print("___++++++++ m",serializer.data)
        print("----->>> minute",len(serializer.data))

        return HttpResponse("Im prediction app ajajajak ", len(serializer.data))
        #return Response(serializer.data)


def dailyDataView(APIView):
    # Query MinuteCryptoData for the most recent 1-minute data
    daily_data = DailyCryptoData.objects.all().order_by('-start_interval_timestamp')

    # Serialize the data
    serializer = DailyCryptoDataSerializer(daily_data, many=True)
    print(len(serializer.data))
    print("___++++++++", serializer.data)
    print("------->>> daily",len(serializer.data))

    return HttpResponse("Im prediction app ajajajak ", len(serializer.data))
    # return Response(serializer.data)


def createMinuteData(self):
    interval = "1m"
    limit = 1
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    url = f'https://api.binance.us/api/v1/klines?symbol=BTCUSDT&interval=1m&limit=1'
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
        print("Success",data)
        return HttpResponse("I'm prediction app ajajajak'",data)
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return HttpResponse("I'm prediction app ajajajak'")





def createDailyData(request):
    interval = "1M"
    limit = 1000
    symbol = 'BTCUSDT'
    base_url = 'https://api.binance.us/api/v1/klines'
    url = f'https://api.binance.us/api/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
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
        print("Day data len---->", len(data))

        print("Day data---->",data)
        return HttpResponse("I'm prediction app ajajajak'",data)
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return HttpResponse("I'm prediction app ajajajak'")








####Prediction#####



def predictionView(request):
        # Get num_days from query parameters
        num_days = 12
        print("prediction start----")


        # Load daily data from the database for the previous 3 months
        end_date = timezone.now()
        start_date = end_date - timedelta(days=90)  # Previous 3 months
        daily_data = DailyCryptoData.objects.filter(start_interval_timestamp__range=(start_date, end_date)).order_by('start_interval_timestamp')

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
        predicted_prices = load_and_predict(model_path, data, time_steps, num_days)

        start_date = date.today() - timedelta(days=len(data))  # Assuming today is the last day in the previous data
        previous_dates = generate_date_sequence(start_date, len(data))
        next_dates = generate_date_sequence(start_date + timedelta(days=len(data)), num_days)

        # Combine dates with predicted prices
        previous_data = list(zip(previous_dates, data))
        next_data = list(zip(next_dates, predicted_prices))

        # Combine previous and next data lists
        combined_data = previous_data + next_data

        # Send combined data as a response
        response_data = [{"date": date, "price": price} for date, price in combined_data]
        print('prediction',response_data)
        return HttpResponse("I'm prediction app ajajajak'")
def load_and_predict(model_path, data, time_steps, num_days):
    # Load the model
    print("checkpoint 1")
    custom_objects = {'Orthogonal': Orthogonal}

    # Load the model with the custom_objects parameter
    model = load_model(model_path)
    # model = load_model(model_path)
    print("checkpoint 2")
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Predict next days
    predicted_prices = predict_next_days(model, scaled_data, scaler, time_steps, num_days)

    return predicted_prices


def predict_next_days(model, data, scaler, time_steps, num_days):
    # Copy the original data
    input_data = data.copy()

    # Initialize list to store predicted prices
    predicted_prices = []

    # Iterate over the number of days
    for _ in range(num_days):
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


def generate_date_sequence(start_date, num_days):
    # Generate a sequence of dates starting from the given start_date
    return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
