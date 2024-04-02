from __future__ import absolute_import, unicode_literals
import os

from celery import Celery
from django.conf import settings
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crypto.settings')

app = Celery('crypto')
app.conf.enable_utc = False
# TIME_ZONE = ''
app.conf.update(timezone='Australia/Perth')

app.config_from_object(settings, namespace='CELERY')

# Celery Beat Settings @shared_task(bind=True)


# Celery Schedules - https://docs.celeryproject.org/en/stable/reference/celery.schedules.html

app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')