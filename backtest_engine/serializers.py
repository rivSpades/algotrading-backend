"""
Serializers for Backtest Engine models
"""

from decimal import Decimal
from rest_framework import serializers
from .models import Backtest, Trade, BacktestStatistics
from strategies.serializers import StrategyDefinitionSerializer
from market_data.serializers import SymbolListSerializer


class TradeSerializer(serializers.ModelSerializer):
    """Serializer for Trade"""
    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    
    class Meta:
        model = Trade
        fields = [
            'id', 'backtest', 'symbol', 'symbol_info', 'trade_type',
            'entry_price', 'exit_price', 'entry_timestamp', 'exit_timestamp',
            'quantity', 'pnl', 'pnl_percentage', 'is_winner', 'max_drawdown', 'metadata'
        ]
        read_only_fields = ['id']


class BacktestStatisticsSerializer(serializers.ModelSerializer):
    """Serializer for BacktestStatistics"""
    symbol_info = SymbolListSerializer(source='symbol', read_only=True)
    
    class Meta:
        model = BacktestStatistics
        fields = [
            'id', 'backtest', 'symbol', 'symbol_info',
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'total_pnl', 'total_pnl_percentage', 'average_pnl',
            'average_winner', 'average_loser', 'profit_factor',
            'max_drawdown', 'max_drawdown_duration', 'sharpe_ratio',
            'cagr', 'total_return', 'equity_curve', 'additional_stats',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class BacktestSerializer(serializers.ModelSerializer):
    """Serializer for Backtest"""
    strategy_info = StrategyDefinitionSerializer(source='strategy', read_only=True)
    symbols_info = SymbolListSerializer(source='symbols', many=True, read_only=True)
    trades = TradeSerializer(many=True, read_only=True)
    statistics = BacktestStatisticsSerializer(many=True, read_only=True)
    
    class Meta:
        model = Backtest
        fields = [
            'id', 'name', 'strategy', 'strategy_info', 'strategy_assignment',
            'symbols', 'symbols_info', 'start_date', 'end_date', 'split_ratio',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters', 'status', 'error_message',
            'created_at', 'updated_at', 'completed_at',
            'trades', 'statistics'
        ]
        read_only_fields = ['created_at', 'updated_at', 'completed_at', 'status', 'error_message']


class BacktestCreateSerializer(serializers.Serializer):
    """Serializer for creating a new backtest"""
    name = serializers.CharField(required=False, allow_blank=True)
    strategy_id = serializers.IntegerField(required=True)
    symbol_tickers = serializers.ListField(
        child=serializers.CharField(),
        required=True,
        min_length=1
    )
    start_date = serializers.DateTimeField(required=False, allow_null=True)
    end_date = serializers.DateTimeField(required=False, allow_null=True)
    split_ratio = serializers.FloatField(required=False, default=0.7, min_value=0.0, max_value=1.0)
    strategy_parameters = serializers.JSONField(required=False, default=dict)
    initial_capital = serializers.DecimalField(required=False, default=10000.0, max_digits=20, decimal_places=2, min_value=Decimal('0.01'))
    bet_size_percentage = serializers.FloatField(required=False, default=100.0, min_value=0.1, max_value=100.0, help_text="Percentage of available capital to bet per trade")

