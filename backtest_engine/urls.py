"""
URL configuration for backtest_engine app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BacktestViewSet, TradeViewSet, BacktestStatisticsViewSet

router = DefaultRouter()
router.register(r'backtests', BacktestViewSet, basename='backtest')
router.register(r'trades', TradeViewSet, basename='trade')
router.register(r'backtest-statistics', BacktestStatisticsViewSet, basename='backtest-statistics')

urlpatterns = [
    path('', include(router.urls)),
]

