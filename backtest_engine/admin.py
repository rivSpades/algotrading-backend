"""
Admin configuration for backtest_engine app
"""

from django.contrib import admin
from .models import (
    Backtest,
    Trade,
    BacktestStatistics,
    HedgeLabSettings,
    SymbolBacktestRun,
    SymbolBacktestTrade,
    SymbolBacktestStatistics,
    SymbolBacktestParameterSet,
)


@admin.register(Backtest)
class BacktestAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'strategy', 'status', 'initial_capital', 'created_at', 'completed_at']
    list_filter = ['status', 'strategy', 'created_at', 'completed_at']
    search_fields = ['name', 'strategy__name', 'error_message']
    readonly_fields = ['created_at', 'updated_at', 'completed_at']
    filter_horizontal = ['symbols']  # Better UI for many-to-many relationships
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'strategy', 'strategy_assignment', 'symbols')
        }),
        ('Backtest Configuration', {
            'fields': (
                'start_date',
                'end_date',
                'split_ratio',
                'initial_capital',
                'bet_size_percentage',
                'strategy_parameters',
                'hedge_enabled',
                'hedge_config',
                'position_modes',
            )
        }),
        ('Status', {
            'fields': ('status', 'error_message')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
    date_hierarchy = 'created_at'


@admin.register(HedgeLabSettings)
class HedgeLabSettingsAdmin(admin.ModelAdmin):
    list_display = ['singleton_key', 'updated_at']
    readonly_fields = ['singleton_key', 'updated_at']


@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ['id', 'backtest', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'pnl', 'is_winner', 'entry_timestamp']
    list_filter = ['trade_type', 'is_winner', 'backtest', 'symbol', 'entry_timestamp']
    search_fields = ['symbol__ticker', 'backtest__name', 'backtest__strategy__name']
    readonly_fields = ['entry_timestamp', 'exit_timestamp']
    fieldsets = (
        ('Trade Information', {
            'fields': ('backtest', 'symbol', 'trade_type')
        }),
        ('Prices', {
            'fields': ('entry_price', 'exit_price', 'quantity')
        }),
        ('Timestamps', {
            'fields': ('entry_timestamp', 'exit_timestamp')
        }),
        ('Results', {
            'fields': ('pnl', 'pnl_percentage', 'is_winner', 'max_drawdown')
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
    )
    date_hierarchy = 'entry_timestamp'


@admin.register(BacktestStatistics)
class BacktestStatisticsAdmin(admin.ModelAdmin):
    list_display = ['id', 'backtest', 'symbol', 'total_trades', 'total_pnl', 'win_rate', 'cagr', 'created_at']
    list_filter = ['backtest', 'created_at']
    search_fields = ['backtest__name', 'backtest__strategy__name', 'symbol__ticker']
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        ('Basic Information', {
            'fields': ('backtest', 'symbol')
        }),
        ('Trade Metrics', {
            'fields': ('total_trades', 'winning_trades', 'losing_trades', 'win_rate')
        }),
        ('PnL Metrics', {
            'fields': ('total_pnl', 'total_pnl_percentage', 'average_pnl', 'average_winner', 'average_loser', 'profit_factor')
        }),
        ('Risk Metrics', {
            'fields': (
                'max_drawdown',
                'max_drawdown_duration',
                'avg_intra_trade_drawdown',
                'worst_intra_trade_drawdown',
                'sharpe_ratio',
            )
        }),
        ('Performance Metrics', {
            'fields': ('cagr', 'total_return')
        }),
        ('Additional Data', {
            'fields': ('equity_curve', 'additional_stats'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    date_hierarchy = 'created_at'


@admin.register(SymbolBacktestParameterSet)
class SymbolBacktestParameterSetAdmin(admin.ModelAdmin):
    list_display = ['signature', 'label', 'strategy', 'broker', 'created_at']
    list_filter = ['strategy', 'broker', 'created_at']
    search_fields = ['signature', 'label', 'strategy__name']
    readonly_fields = ['signature', 'created_at']
    fieldsets = (
        ('Identity', {'fields': ('signature', 'label')}),
        ('Scope', {'fields': ('strategy', 'broker')}),
        ('Parameters', {'fields': ('parameters',)}),
        ('Timestamps', {'fields': ('created_at',), 'classes': ('collapse',)}),
    )
    date_hierarchy = 'created_at'

    def delete_model(self, request, obj):
        # Model FK is SET_NULL; for admin UX we want "delete global test" to also delete all runs.
        SymbolBacktestRun.objects.filter(parameter_set=obj).delete()
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        # Bulk delete from changelist: delete runs for all selected parameter sets.
        SymbolBacktestRun.objects.filter(parameter_set__in=list(queryset)).delete()
        super().delete_queryset(request, queryset)


@admin.register(SymbolBacktestRun)
class SymbolBacktestRunAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'strategy', 'symbol', 'parameter_set', 'status', 'created_at', 'completed_at']
    list_filter = ['status', 'strategy', 'broker', 'created_at', 'completed_at']
    search_fields = ['name', 'strategy__name', 'symbol__ticker', 'parameter_set__signature', 'parameter_set__label']
    readonly_fields = ['created_at', 'updated_at', 'completed_at', 'error_message']
    fieldsets = (
        ('Basic', {'fields': ('name', 'strategy', 'symbol', 'broker', 'parameter_set')}),
        ('Config', {'fields': ('start_date', 'end_date', 'split_ratio', 'initial_capital', 'bet_size_percentage', 'strategy_parameters', 'position_modes')}),
        ('Hedge', {'fields': ('hedge_enabled', 'run_strategy_only_baseline', 'hedge_config')}),
        ('Status', {'fields': ('status', 'error_message')}),
        ('Timestamps', {'fields': ('created_at', 'updated_at', 'completed_at'), 'classes': ('collapse',)}),
    )
    date_hierarchy = 'created_at'


@admin.register(SymbolBacktestTrade)
class SymbolBacktestTradeAdmin(admin.ModelAdmin):
    list_display = ['id', 'run', 'symbol', 'trade_type', 'entry_price', 'exit_price', 'pnl', 'entry_timestamp']
    list_filter = ['trade_type', 'run', 'symbol', 'entry_timestamp']
    search_fields = ['symbol__ticker', 'run__name', 'run__strategy__name', 'run__parameter_set__label', 'run__parameter_set__signature']
    readonly_fields = ['entry_timestamp', 'exit_timestamp']
    date_hierarchy = 'entry_timestamp'


@admin.register(SymbolBacktestStatistics)
class SymbolBacktestStatisticsAdmin(admin.ModelAdmin):
    list_display = ['id', 'run', 'symbol', 'total_trades', 'total_pnl', 'win_rate', 'cagr', 'created_at']
    list_filter = ['run', 'created_at']
    search_fields = ['run__name', 'run__strategy__name', 'symbol__ticker', 'run__parameter_set__label', 'run__parameter_set__signature']
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        ('Basic Information', {'fields': ('run', 'symbol')}),
        ('Trade Metrics', {'fields': ('total_trades', 'winning_trades', 'losing_trades', 'win_rate')}),
        ('PnL Metrics', {'fields': ('total_pnl', 'total_pnl_percentage', 'average_pnl', 'average_winner', 'average_loser', 'profit_factor')}),
        ('Risk Metrics', {'fields': ('max_drawdown', 'max_drawdown_duration', 'avg_intra_trade_drawdown', 'worst_intra_trade_drawdown', 'sharpe_ratio')}),
        ('Performance Metrics', {'fields': ('cagr', 'total_return')}),
        ('Additional Data', {'fields': ('equity_curve', 'additional_stats'), 'classes': ('collapse',)}),
        ('Timestamps', {'fields': ('created_at', 'updated_at'), 'classes': ('collapse',)}),
    )
    date_hierarchy = 'created_at'
