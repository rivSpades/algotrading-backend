"""
Admin configuration for market_data app
"""

from django.contrib import admin
from .models import Symbol, Exchange, Provider, OHLCV


@admin.register(Exchange)
class ExchangeAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'country', 'timezone']
    search_fields = ['name', 'code']
    list_filter = ['country']


@admin.register(Provider)
class ProviderAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'is_active', 'created_at']
    search_fields = ['name', 'code']
    list_filter = ['is_active']


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ['ticker', 'exchange', 'provider', 'type', 'status', 'last_updated']
    search_fields = ['ticker', 'name']
    list_filter = ['type', 'status', 'exchange', 'provider']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(OHLCV)
class OHLCVAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'timestamp', 'timeframe', 'close', 'volume']
    list_filter = ['timeframe', 'symbol']
    search_fields = ['symbol__ticker']
    readonly_fields = ['created_at']
    date_hierarchy = 'timestamp'
