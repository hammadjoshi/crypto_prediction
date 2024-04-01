
from .serializers import  MinuteCryptoDataSerializer
from datetime import datetime, timedelta,date

from .models import DailyCryptoData, MinuteCryptoData
import json
import requests


from django.http    import HttpResponse


def home(request):
    return HttpResponse("I'm prediction app'")


def ajass(request):
    return HttpResponse("I'm prediction app ajajajak'")



def MinuteDataView(APIView):
        # Query MinuteCryptoData for the most recent 1-minute data
        minute_data = MinuteCryptoData.objects.all().order_by('-start_interval_timestamp')

        # Serialize the data
        serializer = MinuteCryptoDataSerializer(minute_data, many=True)
        print("___++++++++",serializer.data)
        return HttpResponse("I'm prediction app ajajajak'",serializer.data)
        #return Response(serializer.data)


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


