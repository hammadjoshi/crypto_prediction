from celery import shared_task

from time import sleep
import time

from .models import DailyCryptoData, MinuteCryptoData
import requests
import json
from datetime import datetime, timedelta


# def get_historical_data(symbol, interval='1d', limit=720):


@shared_task(bind=True)
def fetch_1_minute_data(self):
    print("-----------Fetching one Minute data and Store into DB-----------")
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
        print("Success")
    else:
        print("Failed")

    sleep(30)
    fetch_1_minute_data()
    return "Task Complete Minute Data!"


@shared_task
def my_task():
    print("Running my task...")
    # Your function code goes here
    time.sleep(30)  # Wait for 30 seconds before running again
    my_task.delay()  # Reschedule the task


