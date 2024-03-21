from django.http    import HttpResponse

def index(request):
    return HttpResponse("Hello")
def getAllCryptoNames(request):
    return HttpResponse("All crypto names")

def currentValues(request):
    return HttpResponse("current values of BTC, ETH, ADA, DOT   ")
