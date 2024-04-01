from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("MinuteDataView/", views.MinuteDataView, name="MinuteDataView"),
    path("MinuteDataCreate/", views.createMinuteData, name="MinuteDataCreate"),

]