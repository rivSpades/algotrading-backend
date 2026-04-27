"""
API Views for Backtest Engine
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.pagination import PageNumberPagination
from django.shortcuts import get_object_or_404
from django.db.models import Q
from .models import Backtest, Trade, BacktestStatistics, SymbolBacktestRun, SymbolBacktestTrade, SymbolBacktestStatistics
from .serializers import (
    BacktestSerializer, BacktestListSerializer, BacktestDetailSerializer, BacktestCreateSerializer,
    TradeSerializer, BacktestStatisticsSerializer, HedgePreviewSerializer, HedgeLabSettingsWriteSerializer,
    SymbolBacktestTradeSerializer, SymbolBacktestStatisticsSerializer, SymbolBacktestRunSerializer,
)
from .services.hybrid_vix_hedge import (
    simulate_hybrid_vix_hedge,
    merge_lab_overrides_with_request,
    resolved_hedge_config_for_backtest,
    merge_defaults_into_hedge_config,
    get_hedge_lab_saved_overrides,
    compute_hedge_panic_snapshot,
)
import json
import logging

from market_data.models import Symbol
from .services.create_backtest import create_backtest_from_validated_data

logger = logging.getLogger(__name__)


class TradePagination(PageNumberPagination):
    """Pagination for trades"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BacktestPagination(PageNumberPagination):
    """Pagination for backtests"""
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BacktestViewSet(viewsets.ModelViewSet):
    """ViewSet for Backtest"""
    queryset = Backtest.objects.all()
    serializer_class = BacktestSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'strategy__name']
    ordering_fields = ['created_at', 'updated_at', 'completed_at']
    ordering = ['-created_at']
    pagination_class = BacktestPagination
    
    def get_queryset(self):
        """Optimize queryset based on action - list vs detail"""
        queryset = Backtest.objects.all()
        
        if self.action == 'list':
            # For list views, only prefetch strategy (needed for strategy.name)
            # Don't prefetch symbols, trades, statistics to improve performance
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        elif self.action == 'retrieve':
            # For detail views (retrieve), only prefetch strategy - trades and statistics are fetched separately
            # This dramatically improves performance for backtests with thousands of symbols
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        else:
            # For other actions (update, partial_update, etc.), may need more data
            # But still avoid prefetching trades/statistics unless absolutely necessary
            queryset = queryset.select_related('strategy', 'strategy_assignment')
        
        # Filter backtests by strategy if strategy parameter is provided
        strategy_id = self.request.query_params.get('strategy', None)
        if strategy_id:
            queryset = queryset.filter(strategy_id=strategy_id)
        return queryset
    
    def get_serializer_class(self):
        if self.action == 'create':
            return BacktestCreateSerializer
        elif self.action == 'list':
            return BacktestListSerializer
        elif self.action == 'retrieve':
            # Use lightweight serializer for detail views - excludes trades and statistics
            # Trades: fetched via /backtests/{id}/trades/ endpoint
            # Statistics: fetched via /backtests/{id}/statistics/optimized/ endpoint
            # Symbols: fetched via /backtests/{id}/symbols/ endpoint
            return BacktestDetailSerializer
        return BacktestSerializer
    
    def create(self, request, *args, **kwargs):
        """Create and start a new backtest"""
        serializer = BacktestCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return create_backtest_from_validated_data(serializer.validated_data)
    
    @action(detail=True, methods=['get'])
    def trades(self, request, pk=None):
        """Get trades for a backtest with pagination and filtering
        
        Query parameters:
        - symbol: Filter by symbol ticker
        - mode: Filter by backtest position mode ('long', 'short') — matches
          ``metadata.position_mode`` stored when trades are saved (one run per mode).
        - no_pagination: Set to 'true' to disable pagination (not recommended for large datasets)
        """
        backtest = self.get_object()
        trades = backtest.trades.select_related('symbol').all()
        
        # Filter by symbol ticker if provided
        symbol_ticker = request.query_params.get('symbol', None)
        if symbol_ticker:
            trades = trades.filter(symbol__ticker=symbol_ticker)
        
        mode = request.query_params.get('mode', None)
        if mode:
            mode = mode.lower()
            if mode not in ['long', 'short']:
                mode = None  # Invalid mode, ignore it
        
        # Filter by position mode (Celery task stores this on each Trade.metadata)
        if mode in ('long', 'short'):
            trades = trades.filter(metadata__position_mode=mode)
        
        # Order by entry timestamp
        trades = trades.order_by('entry_timestamp')
        
        # If symbol filter is present, inject independent bet_amounts from symbol statistics
        # This ensures "Total Invested" matches the independent equity curve for individual symbol views
        # Use the mode parameter to get the correct mode's independent_bet_amounts, but don't filter trades
        independent_bet_amounts = {}
        if symbol_ticker:
            try:
                symbol = Symbol.objects.get(ticker=symbol_ticker)
                symbol_stats = backtest.statistics.filter(symbol=symbol).first()
                if symbol_stats and symbol_stats.additional_stats:
                    # Default to long (main stats row) for independent_bet_amounts lookup
                    stats_mode = mode if mode else 'long'
                    mode_stats = symbol_stats.additional_stats.get(stats_mode, {})
                    independent_bet_amounts = mode_stats.get('independent_bet_amounts', {})
            except Symbol.DoesNotExist:
                pass
        
        # Convert trades queryset to list so we can modify metadata
        trades_list = list(trades)
        
        # Inject independent bet_amount into trade metadata if available
        # Only inject for trades that match the requested mode (if mode is specified)
        # This ensures each mode's trades get the correct mode's independent_bet_amounts
        if independent_bet_amounts:
            from django.utils import timezone as tz
            for trade in trades_list:
                # If mode is specified, only inject for trades matching that mode
                # Otherwise, inject for all trades (when mode is not specified, default to long-mode stats)
                trade_metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
                trade_mode = trade_metadata.get('position_mode')
                
                # Check if this trade matches the requested mode
                should_inject = True
                if mode and trade_mode:
                    should_inject = (trade_mode == mode)
                
                if should_inject and trade.entry_timestamp:
                    # Convert entry_timestamp to ISO string format for lookup
                    entry_ts_str = trade.entry_timestamp.isoformat()
                    # Also try without timezone info in case keys are stored differently
                    entry_ts_str_naive = trade.entry_timestamp.replace(tzinfo=None).isoformat()
                    
                    independent_bet = independent_bet_amounts.get(entry_ts_str) or independent_bet_amounts.get(entry_ts_str_naive)
                    if independent_bet is not None:
                        # Create a copy of metadata and add independent bet_amount
                        if not isinstance(trade.metadata, dict):
                            trade.metadata = {}
                        else:
                            trade.metadata = dict(trade.metadata)  # Create a copy
                        trade.metadata['independent_bet_amount'] = independent_bet
        
        # Check if pagination should be disabled
        no_pagination = request.query_params.get('no_pagination', 'false').lower() == 'true'
        
        if no_pagination:
            # Return all trades without pagination (not recommended for large datasets)
            serializer = TradeSerializer(trades_list, many=True)
            return Response(serializer.data)
        
        # Apply pagination (default behavior) - need to paginate the list
        paginator = TradePagination()
        page = paginator.paginate_queryset(trades_list, request)
        if page is not None:
            serializer = TradeSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        
        # Fallback to non-paginated response if pagination not requested
        serializer = TradeSerializer(trades_list, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'], url_path='preview-hedge')
    def preview_hedge(self, request):
        """Run hybrid VIX hedge + SPY buy-and-hold simulation for a date range (fast preview)."""
        ser = HedgePreviewSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        vd = ser.validated_data
        merged_cfg = merge_lab_overrides_with_request(vd.get('hedge_config'))
        result = simulate_hybrid_vix_hedge(
            vd['start_date'],
            vd['end_date'],
            float(vd['initial_capital']),
            merged_cfg,
            yahoo_only=bool(vd.get('use_yahoo_only', True)),
        )
        return Response(result)

    @action(detail=False, methods=['get'], url_path='hedge-panic-snapshot')
    def hedge_panic_snapshot(self, request):
        """Current VIX z-stress and level vs the hybrid hedge panic rule (same as live overlay)."""
        yahoo_only = str(request.query_params.get('use_yahoo_only', 'true')).lower() in (
            '1', 'true', 'yes', '',
        )
        user_cfg = {}
        raw = request.query_params.get('hedge_config')
        if raw:
            try:
                user_cfg = json.loads(raw)
            except json.JSONDecodeError:
                return Response({'error': 'Invalid hedge_config JSON'}, status=400)
            if not isinstance(user_cfg, dict):
                return Response({'error': 'hedge_config must be a JSON object'}, status=400)

        dep_id = request.query_params.get('deployment')
        if dep_id:
            from live_trading.models import StrategyDeployment

            dep = get_object_or_404(StrategyDeployment, pk=dep_id)
            if not getattr(dep, 'hedge_enabled', False):
                return Response(
                    {
                        'error': (
                            'Hedge is disabled for this deployment; omit deployment= '
                            'or pick a deployment with hedge enabled.'
                        ),
                    },
                    status=400,
                )
            effective = resolved_hedge_config_for_backtest(dep.hedge_config or {})
        else:
            effective = resolved_hedge_config_for_backtest(
                merge_lab_overrides_with_request(user_cfg)
            )

        include_chart = str(
            request.query_params.get('include_chart', 'true'),
        ).lower() not in ('0', 'false', 'no')
        try:
            chart_tail_days = int(request.query_params.get('chart_tail_days', '90'))
        except (TypeError, ValueError):
            chart_tail_days = 90
        chart_tail_days = max(1, min(chart_tail_days, 500))

        result = compute_hedge_panic_snapshot(
            effective,
            yahoo_only=yahoo_only,
            include_chart=include_chart,
            chart_tail_days=chart_tail_days,
        )
        return Response(result)

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get all statistics for a backtest"""
        backtest = self.get_object()
        stats = backtest.statistics.all()
        serializer = BacktestStatisticsSerializer(stats, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='statistics/optimized')
    def statistics_optimized(self, request, pk=None):
        """Get optimized statistics for a backtest - organized by mode (LONG/SHORT)"""
        backtest = self.get_object()

        # Get portfolio stats (symbol=None)
        portfolio_stats = backtest.statistics.filter(symbol__isnull=True).first()

        result = {
            'portfolio': None,
            'symbols': []
        }

        if portfolio_stats:
            serializer = BacktestStatisticsSerializer(portfolio_stats)
            result['portfolio'] = {
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
                'benchmark_equity_curve_x': serializer.data.get('benchmark_equity_curve_x', []),
                'benchmark_equity_curve_y': serializer.data.get('benchmark_equity_curve_y', []),
                'benchmark_ticker': serializer.data.get('benchmark_ticker'),
                'benchmark_error': serializer.data.get('benchmark_error'),
                'hedge_equity_curve_x': serializer.data.get('hedge_equity_curve_x', []),
                'hedge_equity_curve_y': serializer.data.get('hedge_equity_curve_y', []),
                'hedge_metrics': serializer.data.get('hedge_metrics', {}),
                'hedge_error': serializer.data.get('hedge_error'),
                'hedge_enabled': backtest.hedge_enabled,
            }

        # Get symbol-level stats
        symbol_stats = backtest.statistics.filter(symbol__isnull=False).select_related('symbol')
        for stat in symbol_stats:
            serializer = BacktestStatisticsSerializer(stat)
            result['symbols'].append({
                'symbol_ticker': serializer.data.get('symbol_ticker'),
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
            })

        return Response(result)

    @action(detail=True, methods=['get'], url_path='symbol-list')
    def symbols(self, request, pk=None):
        """Get paginated list of symbols associated with this backtest (with search support)"""
        from market_data.serializers import SymbolListSerializer

        # Get backtest - ensure we have the object
        # Use pk from kwargs if available, otherwise use pk parameter
        lookup_pk = self.kwargs.get('pk', pk)
        try:
            backtest = self.get_object()
        except Exception:
            # Try to get backtest directly if get_object() fails
            from .models import Backtest
            backtest = Backtest.objects.get(pk=lookup_pk)

        # Get unique symbols from the backtest (ManyToMany relationship)
        symbols_queryset = backtest.symbols.all().order_by('ticker')

        # Apply search filter if provided
        search = request.query_params.get('search', None)
        if search:
            # Trim whitespace and ensure search is not empty
            search = search.strip()
            if search:
                symbols_queryset = symbols_queryset.filter(ticker__icontains=search)
            else:
                # If search is only whitespace, treat as no search
                search = None

        # Apply pagination
        paginator = PageNumberPagination()
        paginator.page_size = 20
        paginator.page_size_query_param = 'page_size'
        paginator.max_page_size = 100

        page = paginator.paginate_queryset(symbols_queryset, request)
        if page is not None:
            # Serialize symbols to return full symbol objects
            serializer = SymbolListSerializer(page, many=True)
            # Get pagination metadata
            paginated_response = paginator.get_paginated_response(serializer.data)
            return paginated_response

        # Fallback if pagination not requested - return all as array
        serializer = SymbolListSerializer(symbols_queryset, many=True)
        return Response({'results': serializer.data, 'count': len(serializer.data), 'next': None, 'previous': None})

    @action(detail=True, methods=['get'], url_path=r'symbol/(?P<ticker>[^/]+)/$')
    def by_symbol(self, request, pk=None, ticker=None):
        """Get backtest statistics for a specific symbol"""
        backtest = self.get_object()
        try:
            symbol = Symbol.objects.get(ticker=ticker)
            stats = backtest.statistics.filter(symbol=symbol).first()
            if stats:
                serializer = BacktestStatisticsSerializer(stats)
                return Response(serializer.data)
            return Response(
                {'error': f'No statistics found for symbol {ticker} in this backtest'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class SymbolBacktestRunViewSet(viewsets.ModelViewSet):
    """ViewSet for single-symbol SymbolBacktestRun."""

    queryset = SymbolBacktestRun.objects.all().select_related('strategy', 'symbol', 'broker')
    serializer_class = SymbolBacktestRunSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'strategy__name', 'symbol__ticker']
    ordering_fields = ['created_at', 'updated_at', 'completed_at']
    ordering = ['-created_at']
    pagination_class = BacktestPagination

    def get_queryset(self):
        qs = super().get_queryset()
        strategy_id = self.request.query_params.get('strategy', None)
        if strategy_id:
            qs = qs.filter(strategy_id=strategy_id)
        ticker = self.request.query_params.get('ticker', None)
        if ticker:
            qs = qs.filter(symbol__ticker__iexact=ticker)
        parameter_set = self.request.query_params.get('parameter_set', None)
        if parameter_set:
            qs = qs.filter(parameter_set_id=parameter_set)
        return qs

    @action(detail=True, methods=['get'])
    def trades(self, request, pk=None):
        """Get trades for a symbol run with pagination and filtering (same params as backtests)."""
        run = self.get_object()
        trades = run.trades.select_related('symbol').all()

        symbol_ticker = request.query_params.get('symbol', None)
        if symbol_ticker:
            trades = trades.filter(symbol__ticker__iexact=symbol_ticker)

        mode = request.query_params.get('mode', None)
        if mode:
            trades = trades.filter(metadata__position_mode__iexact=mode)

        no_pagination = request.query_params.get('no_pagination', '').lower() in ('1', 'true', 'yes')
        if no_pagination:
            serializer = SymbolBacktestTradeSerializer(trades, many=True)
            return Response(serializer.data)

        paginator = TradePagination()
        page = paginator.paginate_queryset(trades, request)
        serializer = SymbolBacktestTradeSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        run = self.get_object()
        stats = run.statistics.all()
        serializer = SymbolBacktestStatisticsSerializer(stats, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='statistics/optimized')
    def statistics_optimized(self, request, pk=None):
        """Optimized statistics for a symbol run (same shape as backtest statistics/optimized)."""
        run = self.get_object()
        portfolio_stats = run.statistics.filter(symbol__isnull=True).first()
        result = {'portfolio': None, 'symbols': []}
        if portfolio_stats:
            serializer = SymbolBacktestStatisticsSerializer(portfolio_stats)
            result['portfolio'] = {
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
                'benchmark_equity_curve_x': serializer.data.get('benchmark_equity_curve_x', []),
                'benchmark_equity_curve_y': serializer.data.get('benchmark_equity_curve_y', []),
                'benchmark_ticker': serializer.data.get('benchmark_ticker'),
                'benchmark_error': serializer.data.get('benchmark_error'),
                'hedge_equity_curve_x': serializer.data.get('hedge_equity_curve_x', []),
                'hedge_equity_curve_y': serializer.data.get('hedge_equity_curve_y', []),
                'hedge_metrics': serializer.data.get('hedge_metrics', {}),
                'hedge_error': serializer.data.get('hedge_error'),
                'hedge_enabled': run.hedge_enabled,
            }

        symbol_stats = run.statistics.filter(symbol__isnull=False).select_related('symbol')
        for stat in symbol_stats:
            serializer = SymbolBacktestStatisticsSerializer(stat)
            result['symbols'].append({
                'symbol_ticker': serializer.data.get('symbol_ticker'),
                'stats_by_mode': serializer.data.get('stats_by_mode', {}),
                'equity_curve_x': serializer.data.get('equity_curve_x', []),
                'equity_curve_y': serializer.data.get('equity_curve_y', []),
            })
        return Response(result)

    @action(detail=False, methods=['get', 'put'], url_path='hedge-lab-settings')
    def hedge_lab_settings(self, request):
        """GET: saved overrides + full effective defaults. PUT: save lab overrides."""
        if request.method == 'GET':
            saved = get_hedge_lab_saved_overrides()
            effective = merge_defaults_into_hedge_config(saved)
            return Response(
                {
                    'hedge_config': saved,
                    'effective_config': effective,
                }
            )
        ser = HedgeLabSettingsWriteSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        from .models import HedgeLabSettings

        row = HedgeLabSettings.get_solo()
        row.hedge_config = ser.validated_data['hedge_config']
        row.save(update_fields=['hedge_config', 'updated_at'])
        saved = get_hedge_lab_saved_overrides()
        effective = merge_defaults_into_hedge_config(saved)
        return Response(
            {
                'hedge_config': saved,
                'effective_config': effective,
            }
        )


class TradeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Trade (read-only)"""
    queryset = Trade.objects.all().select_related('backtest', 'symbol')
    serializer_class = TradeSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['entry_timestamp', 'pnl']
    ordering = ['entry_timestamp']


class BacktestStatisticsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for BacktestStatistics (read-only)"""
    queryset = BacktestStatistics.objects.all().select_related('backtest', 'symbol')
    serializer_class = BacktestStatisticsSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['created_at', 'total_pnl', 'win_rate']
    ordering = ['-created_at']
