from rest_framework import serializers
from .models import DailyCryptoData, MinuteCryptoData

class DailyCryptoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyCryptoData
        fields = ['start_interval_timestamp', 'opening', 'highest', 'lowest', 'closing', 'volume', 'end_interval_timestamp']

class MinuteCryptoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MinuteCryptoData
        fields = ['start_interval_timestamp', 'opening', 'highest', 'lowest', 'closing', 'volume', 'end_interval_timestamp']