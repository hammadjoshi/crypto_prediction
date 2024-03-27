from django.http    import HttpResponse
from django.shortcuts import render
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.pyplot.switch_backend('Agg')
import matplotlib.pyplot as plt







def index(request):
    return HttpResponse("Hello")
def getAllCryptoNames(request):
    return HttpResponse("All crypto names")

def currentValues(request):
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


   return HttpResponse("current values of BTC, ETH, ADA, DOT   ")



