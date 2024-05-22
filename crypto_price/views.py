from django.shortcuts import render

# Create your views here.
import csv
from .models import DailyCryptoData
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import DailyCryptoData, MinuteCryptoData
from .serializers import DailyCryptoDataSerializer, MinuteCryptoDataSerializer
from datetime import datetime, timedelta,date
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from django.utils import timezone
from keras.initializers import Orthogonal
from rest_framework import status
from .tasks import *
from .models import *
# fetch_1_minute_data()
# fetch_1_day_data()
# Define custom objects dictionary with the custom initializer
# train_data()
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
def home(request):
    return render(request,"home.html")

class DailyDataView(APIView):
    def get(self, request):
        # Calculate start and end dates for the previous 2 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Previous 2 months

        # Query DailyCryptoData for data within the specified date range
        daily_data = DailyCryptoData.objects.filter(
            start_interval_timestamp__range=(start_date, end_date)
        )

        # Serialize the data
        serializer = DailyCryptoDataSerializer(daily_data, many=True)

        return Response(serializer.data)

class MinuteDataView(APIView):
    def get(self, request):
        # Query MinuteCryptoData for the most recent 1-minute data
        try:
            minute_data = MinuteCryptoData.objects.all().order_by('-start_interval_timestamp')[:90]
            print("minute_data",minute_data)
            # Serialize the data
            serializer = MinuteCryptoDataSerializer(minute_data, many=True)

            return Response(serializer.data)
        except Exception as e:
            return Response({"error": str(e)})

def generate_date_sequence(start_date, num_minutes):
    # Generate a sequence of timestamps starting from the given start_date
    return [(start_date + timedelta(minutes=i)).strftime('%H:%M:%S') for i in range(num_minutes)]


class PredictionView(APIView):
    def get(self, request):
        # Get num_days from query parameters
        data=prediction_data.objects.all().order_by('-timestamp')
        if data.count()>90:
            data=data[:90]
        else:
            pass
        response_data=prediction_serializer(data,many=True).data
       

        
        return Response(response_data)


def save_daily_data_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse datetime string
            start_interval_timestamp_str = row['start_interval_Timestamp']
            end_interval_timestamp_str = row['end_interval_Timestamp']

            # Remove milliseconds if present
            if '.' in start_interval_timestamp_str:
                start_interval_timestamp_str = start_interval_timestamp_str.split('.')[0]
            if '.' in end_interval_timestamp_str:
                end_interval_timestamp_str = end_interval_timestamp_str.split('.')[0]

            # Parse datetime string to datetime object
            start_interval_timestamp = datetime.strptime(start_interval_timestamp_str, '%Y-%m-%d %H:%M:%S')
            end_interval_timestamp = datetime.strptime(end_interval_timestamp_str, '%Y-%m-%d %H:%M:%S')

            opening = float(row['Opening'])
            highest = float(row['Highest'])
            lowest = float(row['Lowest'])
            closing = float(row['Closing'])
            volume = float(row['Volume'])

            # Save data to model
            DailyCryptoData.objects.create(
                start_interval_timestamp=start_interval_timestamp,
                opening=opening,
                highest=highest,
                lowest=lowest,
                closing=closing,
                volume=volume,
                end_interval_timestamp=end_interval_timestamp
            )
            print("row created")
            
def save_1_day_data_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse datetime string
            start_interval_timestamp_str = row['start_interval_Timestamp']
            end_interval_timestamp_str = row['end_interval_Timestamp']

            # Remove milliseconds if present
            if '.' in start_interval_timestamp_str:
                start_interval_timestamp_str = start_interval_timestamp_str.split('.')[0]
            if '.' in end_interval_timestamp_str:
                end_interval_timestamp_str = end_interval_timestamp_str.split('.')[0]

            # Parse datetime string to datetime object
            start_interval_timestamp = datetime.strptime(start_interval_timestamp_str, '%Y-%m-%d %H:%M:%S')
            end_interval_timestamp = datetime.strptime(end_interval_timestamp_str, '%Y-%m-%d %H:%M:%S')

            opening = float(row['Opening'])
            highest = float(row['Highest'])
            lowest = float(row['Lowest'])
            closing = float(row['Closing'])
            volume = float(row['Volume'])

            # Save data to model
            MinuteCryptoData.objects.create(
                start_interval_timestamp=start_interval_timestamp,
                opening=opening,
                highest=highest,
                lowest=lowest,
                closing=closing,
                volume=volume,
                end_interval_timestamp=end_interval_timestamp
            )
            print("row created")
from .tasks import *
# minute_prediction()
# minute_prediction()           
# save_1_day_data_from_csv('static/cryptocurrency_data_11day.csv')