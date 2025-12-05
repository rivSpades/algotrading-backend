"""
Django Admin configuration for Analytical Tools
"""

from django.contrib import admin
from .models import ToolDefinition, ToolAssignment, IndicatorValue


@admin.register(ToolDefinition)
class ToolDefinitionAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'created_at', 'updated_at']
    list_filter = ['category', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(ToolAssignment)
class ToolAssignmentAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'tool', 'enabled', 'created_at']
    list_filter = ['enabled', 'tool', 'created_at']
    search_fields = ['symbol__ticker', 'tool__name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(IndicatorValue)
class IndicatorValueAdmin(admin.ModelAdmin):
    list_display = ['assignment', 'timestamp', 'value', 'created_at']
    list_filter = ['assignment__tool', 'created_at']
    search_fields = ['assignment__symbol__ticker', 'assignment__tool__name']
    readonly_fields = ['created_at']
    date_hierarchy = 'timestamp'
