"""
URL configuration for market_data app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    SymbolViewSet, ExchangeViewSet, ProviderViewSet,
    PeriodicTaskViewSet, CrontabScheduleViewSet, IntervalScheduleViewSet,
    TaskExecutionViewSet
)

router = DefaultRouter()
router.register(r'symbols', SymbolViewSet, basename='symbol')
router.register(r'exchanges', ExchangeViewSet, basename='exchange')
router.register(r'providers', ProviderViewSet, basename='provider')
router.register(r'scheduled-tasks', PeriodicTaskViewSet, basename='periodictask')
router.register(r'crontab-schedules', CrontabScheduleViewSet, basename='crontabschedule')
router.register(r'interval-schedules', IntervalScheduleViewSet, basename='intervalschedule')
router.register(r'tasks', TaskExecutionViewSet, basename='taskexecution')

urlpatterns = [
    path('', include(router.urls)),
]

