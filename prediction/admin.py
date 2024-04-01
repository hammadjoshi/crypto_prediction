from django.contrib import admin
from . import models
admin.site.register(models.DailyCryptoData)
admin.site.register(models.MinuteCryptoData)


# Register your models here.
