from django.urls import path
from . import views

urlpatterns = [
    path("",views.home,name="home"),
    path("DailyDataView/",views.DailyDataView.as_view(),name="DailyDataView"),
    path("MinuteDataView/",views.MinuteDataView.as_view(),name="MinuteDataView"),
    path("PredictionView/",views.PredictionView.as_view(),name="PredictionView"),
    
    
    # path("admin/", admin.site.urls),
]