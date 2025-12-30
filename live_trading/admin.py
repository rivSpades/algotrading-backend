"""
Live Trading Admin
"""

from django.contrib import admin
from .models import Broker, SymbolBrokerAssociation, LiveTradingDeployment, LiveTrade


@admin.register(Broker)
class BrokerAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'paper_trading_active', 'real_money_active', 'created_at']
    list_filter = ['paper_trading_active', 'real_money_active', 'created_at']
    search_fields = ['name', 'code']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(SymbolBrokerAssociation)
class SymbolBrokerAssociationAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'broker', 'long_active', 'short_active', 'verified_at', 'updated_at']
    list_filter = ['broker', 'long_active', 'short_active', 'verified_at']
    search_fields = ['symbol__ticker', 'broker__name']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(LiveTradingDeployment)
class LiveTradingDeploymentAdmin(admin.ModelAdmin):
    list_display = ['name', 'backtest', 'broker', 'deployment_type', 'status', 'started_at', 'created_at']
    list_filter = ['deployment_type', 'status', 'broker', 'created_at']
    search_fields = ['name', 'backtest__strategy__name', 'broker__name']
    readonly_fields = ['created_at', 'updated_at']
    filter_horizontal = ['symbols']


@admin.register(LiveTrade)
class LiveTradeAdmin(admin.ModelAdmin):
    list_display = ['symbol', 'deployment', 'trade_type', 'position_mode', 'entry_price', 'status', 'pnl', 'entry_timestamp']
    list_filter = ['status', 'trade_type', 'position_mode', 'deployment', 'entry_timestamp']
    search_fields = ['symbol__ticker', 'deployment__backtest__strategy__name', 'broker_order_id']
    readonly_fields = ['created_at', 'updated_at']
