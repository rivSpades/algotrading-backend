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
    # Only include symbol ID, not full symbol object or ticker
    # Symbols should be searched separately via API
    symbol_ticker = serializers.SerializerMethodField()
    equity_curve_x = serializers.SerializerMethodField()
    equity_curve_y = serializers.SerializerMethodField()
    stats_by_mode = serializers.SerializerMethodField()
    
    class Meta:
        model = BacktestStatistics
        fields = [
            'id', 'backtest', 'symbol', 'symbol_ticker',
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'total_pnl', 'total_pnl_percentage', 'average_pnl',
            'average_winner', 'average_loser', 'profit_factor',
            'max_drawdown', 'max_drawdown_duration', 'sharpe_ratio',
            'cagr', 'total_return', 'equity_curve', 'equity_curve_x', 'equity_curve_y',
            'additional_stats', 'stats_by_mode',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def get_symbol_ticker(self, obj):
        """Get symbol ticker if symbol exists"""
        return obj.symbol.ticker if obj.symbol else None
    
    def get_equity_curve_x(self, obj):
        """Convert equity curve to X-axis array (timestamps)"""
        if not obj.equity_curve or not isinstance(obj.equity_curve, list):
            return []
        return [point.get('timestamp') for point in obj.equity_curve if isinstance(point, dict)]
    
    def get_equity_curve_y(self, obj):
        """Convert equity curve to Y-axis array (equity values)"""
        if not obj.equity_curve or not isinstance(obj.equity_curve, list):
            return []
        return [float(point.get('equity', 0)) for point in obj.equity_curve if isinstance(point, dict)]
    
    def get_stats_by_mode(self, obj):
        """Return statistics organized by mode (ALL, LONG, SHORT)"""
        # Helper function to extract equity curve arrays
        def extract_equity_curve_arrays(equity_curve_data):
            """Extract X and Y arrays from equity curve data"""
            if not equity_curve_data or not isinstance(equity_curve_data, list):
                return {'x': [], 'y': []}
            x = [point.get('timestamp') for point in equity_curve_data if isinstance(point, dict)]
            y = [float(point.get('equity', 0)) for point in equity_curve_data if isinstance(point, dict)]
            return {'x': x, 'y': y}
        
        # Get 'all' mode equity curve from main field
        all_equity_curve = extract_equity_curve_arrays(obj.equity_curve)
        
        result = {
            'all': {
                'total_trades': obj.total_trades,
                'winning_trades': obj.winning_trades,
                'losing_trades': obj.losing_trades,
                'win_rate': round(float(obj.win_rate), 2) if obj.win_rate else None,
                'total_pnl': round(float(obj.total_pnl), 2) if obj.total_pnl else None,
                'total_pnl_percentage': round(float(obj.total_pnl_percentage), 2) if obj.total_pnl_percentage else None,
                'average_pnl': round(float(obj.average_pnl), 2) if obj.average_pnl else None,
                'average_winner': round(float(obj.average_winner), 2) if obj.average_winner else None,
                'average_loser': round(float(obj.average_loser), 2) if obj.average_loser else None,
                'profit_factor': round(float(obj.profit_factor), 2) if obj.profit_factor else None,
                'max_drawdown': round(float(obj.max_drawdown), 2) if obj.max_drawdown else None,
                'max_drawdown_duration': obj.max_drawdown_duration,
                'sharpe_ratio': round(float(obj.sharpe_ratio), 2) if obj.sharpe_ratio else None,
                'cagr': round(float(obj.cagr), 2) if obj.cagr else None,
                'total_return': round(float(obj.total_return), 2) if obj.total_return else None,
                'equity_curve': obj.equity_curve or [],
                'equity_curve_x': all_equity_curve['x'],
                'equity_curve_y': all_equity_curve['y'],
            }
        }
        
        # Extract LONG and SHORT stats from additional_stats
        if obj.additional_stats and isinstance(obj.additional_stats, dict):
            long_stats = obj.additional_stats.get('long', {})
            short_stats = obj.additional_stats.get('short', {})
            
            if long_stats:
                long_equity_curve = extract_equity_curve_arrays(long_stats.get('equity_curve'))
                result['long'] = {
                    'total_trades': long_stats.get('total_trades', 0),
                    'winning_trades': long_stats.get('winning_trades', 0),
                    'losing_trades': long_stats.get('losing_trades', 0),
                    'win_rate': round(float(long_stats.get('win_rate', 0)), 2) if long_stats.get('win_rate') else None,
                    'total_pnl': round(float(long_stats.get('total_pnl', 0)), 2) if long_stats.get('total_pnl') is not None else None,
                    'total_pnl_percentage': round(float(long_stats.get('total_pnl_percentage', 0)), 2) if long_stats.get('total_pnl_percentage') is not None else None,
                    'average_pnl': round(float(long_stats.get('average_pnl', 0)), 2) if long_stats.get('average_pnl') is not None else None,
                    'average_winner': round(float(long_stats.get('average_winner', 0)), 2) if long_stats.get('average_winner') is not None else None,
                    'average_loser': round(float(long_stats.get('average_loser', 0)), 2) if long_stats.get('average_loser') is not None else None,
                    'profit_factor': round(float(long_stats.get('profit_factor', 0)), 2) if long_stats.get('profit_factor') is not None else None,
                    'max_drawdown': round(float(long_stats.get('max_drawdown', 0)), 2) if long_stats.get('max_drawdown') is not None else None,
                    'max_drawdown_duration': long_stats.get('max_drawdown_duration'),
                    'sharpe_ratio': round(float(long_stats.get('sharpe_ratio', 0)), 2) if long_stats.get('sharpe_ratio') is not None else None,
                    'cagr': round(float(long_stats.get('cagr', 0)), 2) if long_stats.get('cagr') is not None else None,
                    'total_return': round(float(long_stats.get('total_return', 0)), 2) if long_stats.get('total_return') is not None else None,
                    'equity_curve': long_stats.get('equity_curve', []),
                    'equity_curve_x': long_equity_curve['x'],
                    'equity_curve_y': long_equity_curve['y'],
                }
            
            if short_stats:
                short_equity_curve = extract_equity_curve_arrays(short_stats.get('equity_curve'))
                result['short'] = {
                    'total_trades': short_stats.get('total_trades', 0),
                    'winning_trades': short_stats.get('winning_trades', 0),
                    'losing_trades': short_stats.get('losing_trades', 0),
                    'win_rate': round(float(short_stats.get('win_rate', 0)), 2) if short_stats.get('win_rate') else None,
                    'total_pnl': round(float(short_stats.get('total_pnl', 0)), 2) if short_stats.get('total_pnl') is not None else None,
                    'total_pnl_percentage': round(float(short_stats.get('total_pnl_percentage', 0)), 2) if short_stats.get('total_pnl_percentage') is not None else None,
                    'average_pnl': round(float(short_stats.get('average_pnl', 0)), 2) if short_stats.get('average_pnl') is not None else None,
                    'average_winner': round(float(short_stats.get('average_winner', 0)), 2) if short_stats.get('average_winner') is not None else None,
                    'average_loser': round(float(short_stats.get('average_loser', 0)), 2) if short_stats.get('average_loser') is not None else None,
                    'profit_factor': round(float(short_stats.get('profit_factor', 0)), 2) if short_stats.get('profit_factor') is not None else None,
                    'max_drawdown': round(float(short_stats.get('max_drawdown', 0)), 2) if short_stats.get('max_drawdown') is not None else None,
                    'max_drawdown_duration': short_stats.get('max_drawdown_duration'),
                    'sharpe_ratio': round(float(short_stats.get('sharpe_ratio', 0)), 2) if short_stats.get('sharpe_ratio') is not None else None,
                    'cagr': round(float(short_stats.get('cagr', 0)), 2) if short_stats.get('cagr') is not None else None,
                    'total_return': round(float(short_stats.get('total_return', 0)), 2) if short_stats.get('total_return') is not None else None,
                    'equity_curve': short_stats.get('equity_curve', []),
                    'equity_curve_x': short_equity_curve['x'],
                    'equity_curve_y': short_equity_curve['y'],
                }
        
        return result


class BacktestListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for Backtest list views - excludes nested objects"""
    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    symbols_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Backtest
        fields = [
            'id', 'name', 'strategy', 'strategy_name', 'strategy_assignment',
            'start_date', 'end_date', 'split_ratio',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters', 'status', 'error_message',
            'created_at', 'updated_at', 'completed_at',
            'symbols_count'
        ]
        read_only_fields = ['created_at', 'updated_at', 'completed_at', 'status', 'error_message']
    
    def get_symbols_count(self, obj):
        """Get count of symbols without loading all symbol details"""
        if hasattr(obj, 'symbols'):
            return obj.symbols.count()
        return 0


class BacktestDetailSerializer(serializers.ModelSerializer):
    """Lightweight serializer for Backtest detail views - excludes trades and statistics
    
    This serializer only returns basic metadata since:
    - Trades are fetched separately via /backtests/{id}/trades/ endpoint
    - Statistics are fetched separately via /backtests/{id}/statistics/optimized/ endpoint
    - Symbols are fetched separately via /backtests/{id}/symbols/ endpoint
    
    This dramatically improves performance for backtests with thousands of symbols.
    """
    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    
    class Meta:
        model = Backtest
        fields = [
            'id', 'name', 'strategy', 'strategy_name', 'strategy_assignment',
            'start_date', 'end_date', 'split_ratio',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters', 'status', 'error_message',
            'created_at', 'updated_at', 'completed_at',
        ]
        read_only_fields = ['created_at', 'updated_at', 'completed_at', 'status', 'error_message']


class BacktestSerializer(serializers.ModelSerializer):
    """Full serializer for Backtest - includes nested objects (legacy, not recommended for detail views)
    
    WARNING: This serializer includes ALL trades and statistics, which can be very slow
    for backtests with thousands of symbols. Use BacktestDetailSerializer instead for detail views.
    """
    strategy_info = StrategyDefinitionSerializer(source='strategy', read_only=True)
    trades = TradeSerializer(many=True, read_only=True)
    statistics = BacktestStatisticsSerializer(many=True, read_only=True)
    
    class Meta:
        model = Backtest
        fields = [
            'id', 'name', 'strategy', 'strategy_info', 'strategy_assignment',
            'start_date', 'end_date', 'split_ratio',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters', 'status', 'error_message',
            'created_at', 'updated_at', 'completed_at',
            'trades', 'statistics'
        ]
        read_only_fields = ['created_at', 'updated_at', 'completed_at', 'status', 'error_message']
        # Note: 'symbols' field is NOT included in fields list, so it will not be serialized
        # Symbols should be searched separately via API, not included in backtest response


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

