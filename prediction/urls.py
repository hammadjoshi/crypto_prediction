from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("MinuteDataView/", views.MinuteDataView, name="MinuteDataView"),
    path("createMinuteData/", views.createMinuteData, name="createMinuteData"),
    path("createDayData/", views.createDayData, name="createDayData"),
    path("celeryTest/", views.celeryTest, name="celeryTest")

]