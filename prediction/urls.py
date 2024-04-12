from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("minuteDataView/", views.minuteDataView, name="MinuteDataView"),
    path("dailyDataView/", views.dailyDataView, name="MinuteDataView"),
    path("createMinuteData/", views.createMinuteData, name="createMinuteData"),
    path("createDailyData/", views.createDailyData, name="createDailyData"),
    path("celeryTest/", views.celeryTest, name="celeryTest"),
    path("predictionView/", views.predictionView, name="predictionView")




]