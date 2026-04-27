"""URL configuration for Live Trading app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    BrokerViewSet,
    DeploymentEventViewSet,
    LiveTradeViewSet,
    MarketOpenProgressViewSet,
    StrategyDeploymentViewSet,
    SymbolBrokerAssociationViewSet,
)

router = DefaultRouter()
router.register(r'brokers', BrokerViewSet, basename='broker')
router.register(r'symbol-broker-associations', SymbolBrokerAssociationViewSet, basename='symbol-broker-association')
router.register(r'live-trades', LiveTradeViewSet, basename='live-trade')
router.register(r'strategy-deployments', StrategyDeploymentViewSet, basename='strategy-deployment')
router.register(r'deployment-events', DeploymentEventViewSet, basename='deployment-event')
router.register(r'market-open-progress', MarketOpenProgressViewSet, basename='market-open-progress')

urlpatterns = [
    path('', include(router.urls)),
]
