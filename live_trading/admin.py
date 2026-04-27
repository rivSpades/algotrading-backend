"""Live Trading Admin."""

from django.contrib import admin

from .models import (
    Broker,
    SymbolBrokerAssociation,
    StrategyDeployment,
    DeploymentSymbol,
    DeploymentEvent,
    LiveTrade,
)


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


@admin.register(StrategyDeployment)
class StrategyDeploymentAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', 'strategy', 'parameter_set', 'broker',
        'deployment_type', 'position_mode', 'status', 'created_at',
    ]
    list_filter = ['deployment_type', 'status', 'position_mode', 'broker', 'strategy']
    search_fields = ['name', 'strategy__name', 'broker__name', 'parameter_set__signature']
    raw_id_fields = ['strategy', 'parameter_set', 'broker', 'parent_deployment']
    readonly_fields = [
        'created_at', 'updated_at', 'started_at', 'activated_at',
        'evaluated_at', 'last_signal_at',
    ]
    ordering = ['-created_at']


@admin.register(DeploymentSymbol)
class DeploymentSymbolAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'deployment', 'symbol', 'position_mode', 'status',
        'tier', 'color_overall', 'priority', 'updated_at',
    ]
    list_filter = ['status', 'tier', 'color_overall', 'position_mode']
    search_fields = ['symbol__ticker', 'deployment__name']
    raw_id_fields = ['deployment', 'symbol']
    readonly_fields = ['created_at', 'updated_at', 'last_signal_at', 'last_evaluated_at']
    ordering = ['deployment', 'priority']


@admin.register(DeploymentEvent)
class DeploymentEventAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'deployment', 'deployment_symbol', 'event_type',
        'level', 'actor_type', 'actor_id', 'created_at',
    ]
    list_filter = ['event_type', 'level', 'actor_type']
    search_fields = ['deployment__name', 'message', 'actor_id']
    raw_id_fields = ['deployment', 'deployment_symbol']
    readonly_fields = ['created_at']
    ordering = ['-created_at']


@admin.register(LiveTrade)
class LiveTradeAdmin(admin.ModelAdmin):
    list_display = [
        'symbol', 'deployment', 'trade_type', 'position_mode',
        'entry_price', 'status', 'pnl', 'entry_timestamp',
    ]
    list_filter = ['status', 'trade_type', 'position_mode']
    search_fields = ['symbol__ticker', 'deployment__name', 'broker_order_id']
    raw_id_fields = ['deployment', 'deployment_symbol', 'symbol']
    readonly_fields = ['created_at', 'updated_at']
