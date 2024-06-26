from django.db import models

class DailyCryptoData(models.Model):
    start_interval_timestamp = models.DateTimeField()
    opening = models.DecimalField(max_digits=20, decimal_places=10)
    highest = models.DecimalField(max_digits=20, decimal_places=10)
    lowest = models.DecimalField(max_digits=20, decimal_places=10)
    closing = models.DecimalField(max_digits=20, decimal_places=10)
    volume = models.DecimalField(max_digits=20, decimal_places=10)
    end_interval_timestamp = models.DateTimeField()

class MinuteCryptoData(models.Model):
    start_interval_timestamp = models.DateTimeField()
    opening = models.DecimalField(max_digits=20, decimal_places=10)
    highest = models.DecimalField(max_digits=20, decimal_places=10)
    lowest = models.DecimalField(max_digits=20, decimal_places=10)
    closing = models.DecimalField(max_digits=20, decimal_places=10)
    volume = models.DecimalField(max_digits=20, decimal_places=10)
    end_interval_timestamp = models.DateTimeField()
    
    
class prediction_data(models.Model):
    timestamp = models.DateTimeField(primary_key=True)
    orignal_value = models.DecimalField(max_digits=20, decimal_places=10,default=0)
    predicted_value = models.DecimalField(max_digits=20, decimal_places=10,default=0)
