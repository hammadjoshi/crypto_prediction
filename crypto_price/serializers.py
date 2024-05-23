from rest_framework import serializers
from .models import DailyCryptoData, MinuteCryptoData,prediction_data

class DailyCryptoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DailyCryptoData
        fields = ['start_interval_timestamp', 'opening', 'highest', 'lowest', 'closing', 'volume', 'end_interval_timestamp']

class MinuteCryptoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MinuteCryptoData
        fields = ['start_interval_timestamp', 'opening', 'highest', 'lowest', 'closing', 'volume', 'end_interval_timestamp']
        
        
from rest_framework import serializers
from django.utils.timesince import timesince
from datetime import timedelta
class prediction_serializer(serializers.ModelSerializer):
    formatted_timestamp = serializers.SerializerMethodField()

    class Meta:
        model = prediction_data
        fields = ['formatted_timestamp', 'orignal_value', 'predicted_value']

    def get_formatted_timestamp(self, obj):
        # Replace 'your_format' with the desired datetime format
        time_obj=obj.timestamp+timedelta(hours=10)
        return time_obj.strftime('%B %d, %Y %H:%M:%S')