"""
Serializers for Backtest Engine models
"""

from decimal import Decimal
from rest_framework import serializers
from .models import Backtest, Trade, BacktestStatistics
from .position_modes import normalize_position_modes
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
    benchmark_equity_curve_x = serializers.SerializerMethodField()
    benchmark_equity_curve_y = serializers.SerializerMethodField()
    benchmark_ticker = serializers.SerializerMethodField()
    benchmark_error = serializers.SerializerMethodField()
    hedge_equity_curve_x = serializers.SerializerMethodField()
    hedge_equity_curve_y = serializers.SerializerMethodField()
    hedge_metrics = serializers.SerializerMethodField()
    hedge_error = serializers.SerializerMethodField()
    stats_by_mode = serializers.SerializerMethodField()
    
    class Meta:
        model = BacktestStatistics
        fields = [
            'id', 'backtest', 'symbol', 'symbol_ticker',
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'total_pnl', 'total_pnl_percentage', 'average_pnl',
            'average_winner', 'average_loser', 'profit_factor',
            'max_drawdown', 'max_drawdown_duration',
            'avg_intra_trade_drawdown', 'worst_intra_trade_drawdown',
            'sharpe_ratio',
            'cagr', 'total_return', 'equity_curve', 'equity_curve_x', 'equity_curve_y',
            'benchmark_equity_curve_x', 'benchmark_equity_curve_y', 'benchmark_ticker', 'benchmark_error',
            'hedge_equity_curve_x', 'hedge_equity_curve_y', 'hedge_metrics', 'hedge_error',
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
    
    def _portfolio_benchmark_curve(self, obj) -> list:
        """Buy-hold ^GSPC curve; portfolio rows only (symbol is null)."""
        if obj.symbol is not None:
            return []
        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}
        block = extra.get('benchmark') or {}
        if not isinstance(block, dict):
            return []
        curve = block.get('equity_curve') or []
        return curve if isinstance(curve, list) else []
    
    def get_benchmark_equity_curve_x(self, obj):
        curve = self._portfolio_benchmark_curve(obj)
        return [p.get('timestamp') for p in curve if isinstance(p, dict)]
    
    def get_benchmark_equity_curve_y(self, obj):
        curve = self._portfolio_benchmark_curve(obj)
        return [float(p.get('equity', 0)) for p in curve if isinstance(p, dict)]
    
    def get_benchmark_ticker(self, obj):
        if obj.symbol is not None:
            return None
        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}
        block = extra.get('benchmark') or {}
        if isinstance(block, dict) and block.get('ticker'):
            return block['ticker']
        return '^GSPC'
    
    def get_benchmark_error(self, obj):
        if obj.symbol is not None:
            return None
        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}
        block = extra.get('benchmark') or {}
        if isinstance(block, dict) and block.get('error'):
            return str(block['error'])
        return None
    
    def _portfolio_hedge_block(self, obj) -> dict:
        if obj.symbol is not None:
            return {}
        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}
        block = extra.get('hedge') or {}
        return block if isinstance(block, dict) else {}
    
    def get_hedge_equity_curve_x(self, obj):
        curve = self._portfolio_hedge_block(obj).get('equity_curve') or []
        if not isinstance(curve, list):
            return []
        return [p.get('timestamp') for p in curve if isinstance(p, dict)]
    
    def get_hedge_equity_curve_y(self, obj):
        curve = self._portfolio_hedge_block(obj).get('equity_curve') or []
        if not isinstance(curve, list):
            return []
        return [float(p.get('equity', 0)) for p in curve if isinstance(p, dict)]
    
    def get_hedge_metrics(self, obj):
        block = self._portfolio_hedge_block(obj)
        m = block.get('metrics')
        return m if isinstance(m, dict) else {}
    
    def get_hedge_error(self, obj):
        block = self._portfolio_hedge_block(obj)
        if block.get('error'):
            return str(block['error'])
        return None

    def _strategy_only_bundle(self, obj, mode_key: str) -> dict:
        """Baseline (no hedge split) for portfolio or per-symbol rows when dual execution ran."""
        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}
        so_root = extra.get('strategy_only') or {}
        if not isinstance(so_root, dict):
            return {'x': [], 'y': [], 'metrics': {}}
        block = so_root.get(mode_key) or {}
        if not isinstance(block, dict):
            return {'x': [], 'y': [], 'metrics': {}}
        curve = block.get('equity_curve') or []
        if not isinstance(curve, list):
            curve = []
        x = [p.get('timestamp') for p in curve if isinstance(p, dict)]
        y = [float(p.get('equity', 0)) for p in curve if isinstance(p, dict)]
        metric_keys = (
            'total_trades',
            'winning_trades',
            'losing_trades',
            'win_rate',
            'total_pnl',
            'total_pnl_percentage',
            'average_pnl',
            'average_winner',
            'average_loser',
            'profit_factor',
            'max_drawdown',
            'max_drawdown_duration',
            'sharpe_ratio',
            'cagr',
            'total_return',
            'skipped_trades_count',
        )
        metrics = {}
        for k in metric_keys:
            v = block.get(k)
            if v is not None:
                if k in ('win_rate', 'total_pnl_percentage', 'max_drawdown', 'sharpe_ratio', 'cagr', 'total_return'):
                    try:
                        metrics[k] = round(float(v), 2)
                    except (TypeError, ValueError):
                        metrics[k] = v
                elif k in ('total_trades', 'winning_trades', 'losing_trades', 'skipped_trades_count', 'max_drawdown_duration'):
                    metrics[k] = int(v) if v is not None else 0
                else:
                    try:
                        metrics[k] = round(float(v), 2)
                    except (TypeError, ValueError):
                        metrics[k] = v
        return {'x': x, 'y': y, 'metrics': metrics}
    
    def get_stats_by_mode(self, obj):
        """Return statistics by long/short. Main DB row holds primary mode (long when both ran; short when short-only)."""
        # Helper function to extract equity curve arrays
        def extract_equity_curve_arrays(equity_curve_data):
            """Extract X and Y arrays from equity curve data"""
            if not equity_curve_data or not isinstance(equity_curve_data, list):
                return {'x': [], 'y': []}
            x = [point.get('timestamp') for point in equity_curve_data if isinstance(point, dict)]
            y = [float(point.get('equity', 0)) for point in equity_curve_data if isinstance(point, dict)]
            return {'x': x, 'y': y}

        def block_from_model(so_bundle_key):
            """Build one mode block from top-level BacktestStatistics fields (primary row)."""
            eq = extract_equity_curve_arrays(obj.equity_curve)
            so = self._strategy_only_bundle(obj, so_bundle_key)
            return {
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
                'max_drawdown': round(float(obj.max_drawdown), 2) if obj.max_drawdown is not None else None,
                'max_drawdown_duration': obj.max_drawdown_duration,
                'avg_intra_trade_drawdown': round(float(obj.avg_intra_trade_drawdown), 2) if obj.avg_intra_trade_drawdown is not None else None,
                'worst_intra_trade_drawdown': round(float(obj.worst_intra_trade_drawdown), 2) if obj.worst_intra_trade_drawdown is not None else None,
                'sharpe_ratio': round(float(obj.sharpe_ratio), 2) if obj.sharpe_ratio else None,
                'cagr': round(float(obj.cagr), 2) if obj.cagr else None,
                'total_return': round(float(obj.total_return), 2) if obj.total_return else None,
                'equity_curve': obj.equity_curve or [],
                'equity_curve_x': eq['x'],
                'equity_curve_y': eq['y'],
                'benchmark_equity_curve_x': bench_x,
                'benchmark_equity_curve_y': bench_y,
                'hedge_equity_curve_x': hedge_x,
                'hedge_equity_curve_y': hedge_y,
                'hedge_metrics': hedge_m,
                'hedge_error': hedge_err,
                'strategy_only_equity_curve_x': so['x'],
                'strategy_only_equity_curve_y': so['y'],
                'strategy_only_metrics': so['metrics'],
            }

        def block_from_nested(nested: dict, so_bundle_key: str):
            """Build one mode block from additional_stats[nested_key] payload."""
            if not isinstance(nested, dict):
                nested = {}
            eq = extract_equity_curve_arrays(nested.get('equity_curve'))
            so = self._strategy_only_bundle(obj, so_bundle_key)
            return {
                'total_trades': nested.get('total_trades', 0),
                'winning_trades': nested.get('winning_trades', 0),
                'losing_trades': nested.get('losing_trades', 0),
                'win_rate': round(float(nested.get('win_rate', 0)), 2) if nested.get('win_rate') else None,
                'total_pnl': round(float(nested.get('total_pnl', 0)), 2) if nested.get('total_pnl') is not None else None,
                'total_pnl_percentage': round(float(nested.get('total_pnl_percentage', 0)), 2) if nested.get('total_pnl_percentage') is not None else None,
                'average_pnl': round(float(nested.get('average_pnl', 0)), 2) if nested.get('average_pnl') is not None else None,
                'average_winner': round(float(nested.get('average_winner', 0)), 2) if nested.get('average_winner') is not None else None,
                'average_loser': round(float(nested.get('average_loser', 0)), 2) if nested.get('average_loser') is not None else None,
                'profit_factor': round(float(nested.get('profit_factor', 0)), 2) if nested.get('profit_factor') is not None else None,
                'max_drawdown': round(float(nested.get('max_drawdown', 0)), 2) if nested.get('max_drawdown') is not None else None,
                'max_drawdown_duration': nested.get('max_drawdown_duration'),
                'avg_intra_trade_drawdown': round(float(nested.get('avg_intra_trade_drawdown', 0)), 2) if nested.get('avg_intra_trade_drawdown') is not None else None,
                'worst_intra_trade_drawdown': round(float(nested.get('worst_intra_trade_drawdown', 0)), 2) if nested.get('worst_intra_trade_drawdown') is not None else None,
                'sharpe_ratio': round(float(nested.get('sharpe_ratio', 0)), 2) if nested.get('sharpe_ratio') is not None else None,
                'cagr': round(float(nested.get('cagr', 0)), 2) if nested.get('cagr') is not None else None,
                'total_return': round(float(nested.get('total_return', 0)), 2) if nested.get('total_return') is not None else None,
                'equity_curve': nested.get('equity_curve', []),
                'equity_curve_x': eq['x'],
                'equity_curve_y': eq['y'],
                'benchmark_equity_curve_x': bench_x,
                'benchmark_equity_curve_y': bench_y,
                'hedge_equity_curve_x': hedge_x,
                'hedge_equity_curve_y': hedge_y,
                'hedge_metrics': hedge_m,
                'hedge_error': hedge_err,
                'strategy_only_equity_curve_x': so['x'],
                'strategy_only_equity_curve_y': so['y'],
                'strategy_only_metrics': so['metrics'],
            }

        bench_x = self.get_benchmark_equity_curve_x(obj)
        bench_y = self.get_benchmark_equity_curve_y(obj)
        hedge_x = self.get_hedge_equity_curve_x(obj)
        hedge_y = self.get_hedge_equity_curve_y(obj)
        hedge_m = self.get_hedge_metrics(obj)
        hedge_err = self.get_hedge_error(obj)

        modes = normalize_position_modes(getattr(obj.backtest, 'position_modes', None))
        has_long = 'long' in modes
        has_short = 'short' in modes

        extra = obj.additional_stats if isinstance(obj.additional_stats, dict) else {}

        if has_short and not has_long:
            # Primary row is short only
            long_nested = extra.get('long') or {}
            return {
                'short': block_from_model('short'),
                'long': block_from_nested(long_nested, 'long'),
            }

        # Long ran (alone or with short): primary row is long
        short_nested = extra.get('short') or {}
        return {
            'long': block_from_model('long'),
            'short': block_from_nested(short_nested, 'short'),
        }


class BacktestListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for Backtest list views - excludes nested objects"""
    strategy_name = serializers.CharField(source='strategy.name', read_only=True)
    symbols_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Backtest
        fields = [
            'id', 'name', 'strategy', 'strategy_name', 'strategy_assignment',
            'start_date', 'end_date', 'split_ratio',
            'initial_capital', 'bet_size_percentage', 'strategy_parameters',
            'hedge_enabled', 'run_strategy_only_baseline', 'hedge_config', 'position_modes',
            'status', 'error_message',
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
            'initial_capital', 'bet_size_percentage', 'strategy_parameters',
            'hedge_enabled', 'run_strategy_only_baseline', 'hedge_config', 'position_modes',
            'status', 'error_message',
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
            'initial_capital', 'bet_size_percentage', 'strategy_parameters',
            'hedge_enabled', 'run_strategy_only_baseline', 'hedge_config', 'position_modes',
            'status', 'error_message',
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
        required=False,  # Optional when broker filtering is used
        allow_empty=True
    )
    broker_id = serializers.IntegerField(required=False, allow_null=True, help_text="Optional broker ID for broker-aware symbol filtering")
    exchange_code = serializers.CharField(required=False, allow_blank=True, allow_null=True, help_text="Optional exchange code when filtering by broker")
    start_date = serializers.DateTimeField(required=False, allow_null=True)
    end_date = serializers.DateTimeField(required=False, allow_null=True)
    split_ratio = serializers.FloatField(required=False, default=0.7, min_value=0.0, max_value=1.0)
    strategy_parameters = serializers.JSONField(required=False, default=dict)
    initial_capital = serializers.DecimalField(required=False, default=10000.0, max_digits=20, decimal_places=2, min_value=Decimal('0.01'))
    bet_size_percentage = serializers.FloatField(required=False, default=100.0, min_value=0.1, max_value=100.0, help_text="Percentage of available capital to bet per trade")
    hedge_enabled = serializers.BooleanField(required=False, default=False)
    run_strategy_only_baseline = serializers.BooleanField(
        required=False,
        default=True,
        help_text="When hedge_enabled, whether to also run the strategy-only baseline for comparison",
    )
    hedge_config = serializers.JSONField(required=False, default=dict)
    position_modes = serializers.ListField(
        child=serializers.ChoiceField(choices=['long', 'short']),
        required=False,
        allow_empty=False,
        help_text="Subset of ['long','short']; omit to run both. At least one when provided.",
    )

    def validate(self, data):
        """Validate that either symbol_tickers is provided or broker_id is provided"""
        symbol_tickers = data.get('symbol_tickers', [])
        broker_id = data.get('broker_id')
        
        if not broker_id and (not symbol_tickers or len(symbol_tickers) == 0):
            raise serializers.ValidationError(
                "Either 'symbol_tickers' must be provided, or 'broker_id' must be provided for broker-based filtering."
            )

        data['position_modes'] = normalize_position_modes(data.get('position_modes'))

        return data


class HedgePreviewSerializer(serializers.Serializer):
    """Request body for hybrid VIX hedge preview (no backtest required)."""
    start_date = serializers.DateTimeField(required=True)
    end_date = serializers.DateTimeField(required=True)
    initial_capital = serializers.DecimalField(
        required=False,
        default=10000.0,
        max_digits=20,
        decimal_places=2,
        min_value=Decimal('0.01'),
    )
    hedge_config = serializers.JSONField(required=False, default=dict)
    use_yahoo_only = serializers.BooleanField(
        required=False,
        default=True,
        help_text="If true (default), fetch SPY/VIXM/VIXY/^VIX from Yahoo only; do not read DB.",
    )


class HedgeLabSettingsWriteSerializer(serializers.Serializer):
    """Persist hedge lab overrides (known keys only)."""

    hedge_config = serializers.JSONField(required=True)

    def validate_hedge_config(self, value):
        from .services.hybrid_vix_hedge import filter_hedge_config_user_keys

        if not isinstance(value, dict):
            raise serializers.ValidationError("hedge_config must be an object")
        return filter_hedge_config_user_keys(value)

