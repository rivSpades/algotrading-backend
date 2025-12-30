"""
URL configuration for Live Trading app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    BrokerViewSet,
    SymbolBrokerAssociationViewSet,
    LiveTradingDeploymentViewSet,
    LiveTradeViewSet
)

router = DefaultRouter()
router.register(r'brokers', BrokerViewSet, basename='broker')
router.register(r'symbol-broker-associations', SymbolBrokerAssociationViewSet, basename='symbol-broker-association')
router.register(r'live-trading-deployments', LiveTradingDeploymentViewSet, basename='live-trading-deployment')
router.register(r'live-trades', LiveTradeViewSet, basename='live-trade')

urlpatterns = [
    path('', include(router.urls)),
]


