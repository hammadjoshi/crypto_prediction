from django.contrib import admin
from . import models
admin.site.register(models.DailyCryptoData)
admin.site.register(models.MinuteCryptoData)
admin.site.register(models.prediction_data)

# Register your models here.
