"""
URL configuration for analytical_tools app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ToolDefinitionViewSet,
    ToolAssignmentViewSet,
    IndicatorValueViewSet
)

router = DefaultRouter()
router.register(r'tools', ToolDefinitionViewSet, basename='tool')
router.register(r'assignments', ToolAssignmentViewSet, basename='assignment')
router.register(r'values', IndicatorValueViewSet, basename='indicatorvalue')

urlpatterns = [
    path('', include(router.urls)),
]

