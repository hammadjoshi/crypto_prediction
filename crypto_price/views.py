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
from pytz import timezone as pytz_timezone
from datetime import datetime
import pytz

# Assuming you have your model and serializer already defined
# from .models import MinuteCryptoData
# from .serializers import MinuteCryptoDataSerializer

def convert_to_timezone(dt, tz_name):
    """
    Convert a UTC datetime object to a specified timezone.
    """
    tz = pytz_timezone(tz_name)
    return dt.astimezone(tz)
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
            minute_data = MinuteCryptoData.objects.all().order_by('-start_interval_timestamp')
            print("minute_data",minute_data)
            # Serialize the data
            australia_tz = 'Australia/ACT'
            for item in minute_data:
                item.start_interval_timestamp = item.start_interval_timestamp+timedelta(hours=8)

                print(item.start_interval_timestamp)
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
        data=prediction_data.objects.all().order_by('timestamp')
        # if data.count()>90:
        #     data=data[:90]
        # else:
        #     pass
        data=data[:data.count()-1]
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



class MinutePredictionView(APIView):
    def get(self, request):
        # Get num_days from query parameters
        minutes = request.query_params.get('minutes')

        if not minutes:
            return Response({"error": "minutes parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            minutes = int(minutes)
        except ValueError:
            return Response({"error": "minutes must be an integer"}, status=status.HTTP_400_BAD_REQUEST)

        # Load daily data from the database for the previous 3 months
        end_date = timezone.now()
        start_date = end_date - timedelta(minutes=90)  # Previous 3 months
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
        print("next_dates: ", next_dates)
        # Combine dates with predicted prices
        previous_data = list(zip(previous_dates, data))
        next_data = list(zip(next_dates, predicted_prices))

        # Combine previous and next data lists
        combined_data = previous_data + next_data

        # Send combined data as a response
        response_data = [{"date": date, "price": price} for date, price in combined_data]
        return Response(response_data)

from .tasks import *
# minute_prediction()
# minute_prediction()           
# save_1_day_data_from_csv('static/cryptocurrency_data_11day.csv')
# fetch_1_day_data()
# fetch_1_minute_data()