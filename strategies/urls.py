"""
URL configuration for strategies app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StrategyDefinitionViewSet, StrategyAssignmentViewSet

router = DefaultRouter()
router.register(r'strategies', StrategyDefinitionViewSet, basename='strategy')
router.register(r'assignments', StrategyAssignmentViewSet, basename='assignment')

urlpatterns = [
    path('', include(router.urls)),
]


