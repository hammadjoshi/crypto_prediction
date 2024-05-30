from __future__ import absolute_import, unicode_literals
import os

from celery import Celery
from django.conf import settings
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crypto_price_prediction.settings')

app = Celery('crypto_price_prediction')
app.conf.enable_utc = False
# TIME_ZONE = ''
app.conf.update(timezone = 'Australia/Perth')

app.config_from_object(settings, namespace='CELERY')

# Celery Beat Settings @shared_task(bind=True) 

app.conf.beat_schedule = {
    'fetch_1_minute_data': {
        'task': 'crypto_price.tasks.fetch_1_minute_data',  # Replace with the actual path to your first task
        'schedule': 60,  # Fetch data every 60 seconds (1 minute)
    },
    'train_1_minute_data': {
        'task': 'crypto_price.tasks.train_data',  # Replace with the actual path to your first task
        'schedule': 60,  # Fetch data every 60 seconds (1 minute)
    },
    'fetch_1_day_data': {
        'task': 'crypto_price.tasks.fetch_1_day_data',  # Replace with the actual path to your second task
        'schedule': 86400,  # Fetch data every 86400 seconds (1 day)
    },
    'minute_prediction': {
        'task': 'crypto_price.tasks.minute_prediction',  # Replace with the actual path to your second task
        'schedule': 86400,  # Fetch data every 86400 seconds (1 day)
    },
    
}
# Celery Schedules - https://docs.celeryproject.org/en/stable/reference/celery.schedules.html

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')