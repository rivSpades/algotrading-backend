"""
Admin configuration for backtest_engine app
"""

from django.contrib import admin
from .models import Backtest, Trade, BacktestStatistics


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
            'fields': ('start_date', 'end_date', 'split_ratio', 'initial_capital', 'bet_size_percentage', 'strategy_parameters')
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
            'fields': ('max_drawdown', 'max_drawdown_duration', 'sharpe_ratio')
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
