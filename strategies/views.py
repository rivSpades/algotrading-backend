"""
API Views for Strategies
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from django.db import transaction
from django.db.models import Q, Count
from .models import StrategyDefinition, StrategyAssignment
from .serializers import StrategyDefinitionSerializer, StrategyAssignmentSerializer
from market_data.models import Symbol
from market_data.serializers import SymbolListSerializer
from backtest_engine.models import Backtest, SymbolBacktestRun, SymbolBacktestTrade, SymbolBacktestStatistics
from backtest_engine.position_modes import normalize_position_modes
from backtest_engine.services.create_backtest import create_backtest_from_validated_data
from backtest_engine.tasks import run_symbol_backtest_run_task, bulk_symbol_runs_queue_task
from live_trading.models import SymbolBrokerAssociation, Broker


def _snapshot_item(_snap_row):
    raise NotImplementedError("Legacy Backtest snapshot items removed; use SymbolBacktestRun endpoints.")


class StrategyDefinitionViewSet(viewsets.ModelViewSet):
    """ViewSet for StrategyDefinition"""
    queryset = StrategyDefinition.objects.all()
    serializer_class = StrategyDefinitionSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'description_short', 'description_long']
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['name']

    # Legacy /symbol-backtest endpoints removed (single-symbol runs are now stored in SymbolBacktestRun).

    def _symbol_run_item(self, run_row: SymbolBacktestRun):
        return {
            'id': run_row.id,
            'run_id': run_row.id,
            'status': run_row.status,
            'created_at': run_row.created_at,
            'completed_at': run_row.completed_at,
            'updated_at': run_row.updated_at,
            'error_message': run_row.error_message or '',
            'parameters': {
                'name': run_row.name or '',
                'split_ratio': float(run_row.split_ratio),
                'initial_capital': str(run_row.initial_capital),
                'bet_size_percentage': float(run_row.bet_size_percentage),
                'strategy_parameters': run_row.strategy_parameters or {},
                'position_modes': run_row.position_modes or [],
                'hedge_enabled': bool(run_row.hedge_enabled),
                'hedge_config': run_row.hedge_config or {},
                'run_strategy_only_baseline': bool(run_row.run_strategy_only_baseline),
                'start_date': run_row.start_date.isoformat() if run_row.start_date else None,
                'end_date': run_row.end_date.isoformat() if run_row.end_date else None,
                'broker_id': run_row.broker_id,
            },
        }

    @action(detail=True, methods=['get'], url_path=r'symbol-runs/(?P<ticker>[^/.]+)')
    def symbol_runs(self, request, pk=None, ticker=None):
        """All stored single-symbol runs for this strategy + ticker (newest first)."""
        strategy = self.get_object()
        try:
            sym = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            return Response({'error': f'Symbol {ticker} not found'}, status=status.HTTP_404_NOT_FOUND)
        rows = (
            SymbolBacktestRun.objects.filter(strategy=strategy, symbol=sym)
            .order_by('-created_at')
        )
        runs = [self._symbol_run_item(r) for r in rows]
        return Response({'runs': runs, 'ticker': ticker})

    @action(detail=True, methods=['post'], url_path=r'symbol-runs/(?P<ticker>[^/.]+)/run')
    def run_symbol_run(self, request, pk=None, ticker=None):
        """Queue a new SymbolBacktestRun for this strategy + ticker."""
        strategy = self.get_object()
        try:
            sym = Symbol.objects.get(ticker=ticker, status='active')
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found or is not active'},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            body = dict(request.data)
        except (TypeError, ValueError):
            body = request.data.dict() if hasattr(request.data, 'dict') else {}

        # Merge parameters like portfolio creation does
        merged_params = (strategy.default_parameters or {}).copy()
        assignment = StrategyAssignment.objects.filter(strategy=strategy, symbol=sym).first()
        if assignment:
            merged_params.update(assignment.parameters or {})
        else:
            global_assignment = StrategyAssignment.objects.filter(strategy=strategy, symbol__isnull=True).first()
            if global_assignment:
                merged_params.update(global_assignment.parameters or {})
        merged_params.update(body.get('strategy_parameters') or {})

        name = body.get('name') or f'{strategy.name} — {ticker}'
        start_date = body.get('start_date') or None
        end_date = body.get('end_date') or None

        broker = None
        broker_id = body.get('broker_id')
        if broker_id:
            try:
                broker = Broker.objects.get(id=int(broker_id))
            except Exception:
                broker = None

        run = SymbolBacktestRun.objects.create(
            name=name,
            strategy=strategy,
            symbol=sym,
            broker=broker,
            start_date=start_date or timezone.now().replace(year=1900, month=1, day=1),
            end_date=end_date or timezone.now(),
            split_ratio=body.get('split_ratio', 0.7),
            initial_capital=body.get('initial_capital', 10000.0),
            bet_size_percentage=body.get('bet_size_percentage', 100.0),
            strategy_parameters=merged_params,
            hedge_enabled=bool(body.get('hedge_enabled', False)),
            run_strategy_only_baseline=bool(body.get('run_strategy_only_baseline', True)),
            hedge_config=body.get('hedge_config') or {},
            position_modes=normalize_position_modes(body.get('position_modes')),
            status='pending',
        )
        task = run_symbol_backtest_run_task.delay(run.id)
        return Response(
            {
                'id': run.id,
                'task_id': task.id,
                'status': run.status,
                'ticker': ticker,
            },
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=['post'], url_path=r'symbol-runs/(?P<ticker>[^/.]+)/recalculate')
    def recalculate_symbol_run(self, request, pk=None, ticker=None):
        """
        Re-run an existing SymbolBacktestRun in place (same row id).

        Body: { "run_id": <int> } — must belong to this strategy and ticker.
        Clears stored trades/statistics for that run, sets status to pending, queues Celery.
        """
        strategy = self.get_object()
        try:
            sym = Symbol.objects.get(ticker=ticker, status='active')
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found or is not active'},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            body = dict(request.data)
        except (TypeError, ValueError):
            body = request.data.dict() if hasattr(request.data, 'dict') else {}

        run_id = body.get('run_id')
        if run_id is None:
            return Response({'error': 'run_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            run_id = int(run_id)
        except (TypeError, ValueError):
            return Response({'error': 'run_id must be an integer'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            run = SymbolBacktestRun.objects.select_related('symbol', 'strategy').get(
                id=run_id, strategy=strategy, symbol=sym
            )
        except SymbolBacktestRun.DoesNotExist:
            return Response(
                {'error': f'Run {run_id} not found for this strategy and {ticker}'},
                status=status.HTTP_404_NOT_FOUND,
            )

        if run.status in ('pending', 'running'):
            return Response(
                {'error': f'Run {run_id} is already {run.status}; wait for it to finish before recalculating.'},
                status=status.HTTP_409_CONFLICT,
            )

        with transaction.atomic():
            SymbolBacktestTrade.objects.filter(run=run).delete()
            SymbolBacktestStatistics.objects.filter(run=run).delete()
            run.status = 'pending'
            run.error_message = ''
            run.completed_at = None
            run.save()

        task = run_symbol_backtest_run_task.delay(run.id)
        return Response(
            {
                'id': run.id,
                'task_id': task.id,
                'status': run.status,
                'ticker': ticker,
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=['post'], url_path=r'symbol-runs/run-bulk')
    def run_symbol_runs_bulk(self, request, pk=None):
        """
        Queue multiple SymbolBacktestRun jobs at once.

        Body:
        - symbol_tickers: list[str] (optional if select_all_linked is true)
        - select_all_linked: bool (optional) -> when true, expand to all broker-linked active symbols
        - broker_id: int (required when select_all_linked is true)
        - exchange_code: str (optional)
        - plus standard run fields (split_ratio, initial_capital, etc.)
        """
        strategy = self.get_object()
        try:
            body = dict(request.data)
        except (TypeError, ValueError):
            body = request.data.dict() if hasattr(request.data, 'dict') else {}

        select_all_linked = bool(body.pop('select_all_linked', False))
        symbol_tickers = body.pop('symbol_tickers', None)
        broker_id = body.get('broker_id')
        exchange_code = body.get('exchange_code', '') or ''

        tickers = []
        if select_all_linked:
            if not broker_id:
                return Response({'error': 'broker_id is required when select_all_linked=true'}, status=status.HTTP_400_BAD_REQUEST)
            try:
                broker = Broker.objects.get(id=broker_id)
            except Broker.DoesNotExist:
                return Response({'error': f'Broker with id {broker_id} not found'}, status=status.HTTP_404_NOT_FOUND)

            qs = SymbolBrokerAssociation.objects.filter(
                broker=broker,
                symbol__status='active',
            ).filter(Q(long_active=True) | Q(short_active=True))

            pmodes = body.get('position_modes') or ['long', 'short']
            has_long = 'long' in pmodes
            has_short = 'short' in pmodes
            if has_long and not has_short:
                qs = qs.filter(long_active=True)
            elif has_short and not has_long:
                qs = qs.filter(short_active=True)

            if exchange_code:
                qs = qs.filter(symbol__exchange__code__iexact=exchange_code)

            tickers = list(qs.values_list('symbol__ticker', flat=True).order_by('symbol__ticker'))
        else:
            if symbol_tickers is None:
                return Response({'error': "Provide symbol_tickers or set select_all_linked=true"}, status=status.HTTP_400_BAD_REQUEST)
            tickers = [str(t).strip().upper() for t in (symbol_tickers or []) if str(t).strip()]

        if not tickers:
            return Response({'error': 'No tickers to run'}, status=status.HTTP_400_BAD_REQUEST)

        task = bulk_symbol_runs_queue_task.delay(strategy.id, tickers, dict(body))
        return Response(
            {
                'task_id': task.id,
                'queued_count': len(tickers),
                'tickers': tickers[:200],
            },
            status=status.HTTP_202_ACCEPTED,
        )

    # NOTE: Cannot be `symbol-runs/summary` because it conflicts with
    # `symbol-runs/(?P<ticker>...)` (router resolves "summary" as a ticker).
    @action(detail=True, methods=['get'], url_path='symbol-runs-summary')
    def symbol_runs_summary(self, request, pk=None):
        """Distinct tickers that have at least one stored SymbolBacktestRun for this strategy."""
        strategy = self.get_object()
        counts = (
            SymbolBacktestRun.objects.filter(strategy=strategy)
            .values('symbol__ticker')
            .annotate(snapshot_count=Count('id'))
        )
        count_map = {c['symbol__ticker']: c['snapshot_count'] for c in counts}
        runs = (
            SymbolBacktestRun.objects.filter(strategy=strategy)
            .select_related('symbol')
            .order_by('symbol__ticker', '-created_at')
        )
        rows = []
        seen = set()
        for r in runs:
            t = r.symbol.ticker
            if t in seen:
                continue
            seen.add(t)
            sym_data = dict(SymbolListSerializer(r.symbol).data)
            sym_data['snapshot_count'] = count_map.get(t, 0)
            sym_data['latest_run_id'] = r.id
            sym_data['latest_run_status'] = r.status
            sym_data['snapshot_updated_at'] = r.updated_at.isoformat() if r.updated_at else None
            rows.append(sym_data)
        rows.sort(key=lambda x: x.get('ticker') or '')
        return Response({'symbols': rows})

    @action(detail=True, methods=['delete'], url_path='symbol-runs')
    def delete_all_symbol_runs(self, request, pk=None):
        """Delete every stored SymbolBacktestRun for this strategy (all symbols, all runs)."""
        strategy = self.get_object()
        qs = SymbolBacktestRun.objects.filter(strategy=strategy)
        count = qs.count()
        if count == 0:
            return Response({'deleted_runs': 0, 'message': 'No symbol runs to delete.'}, status=status.HTTP_200_OK)
        with transaction.atomic():
            total_deleted, _by_model = qs.delete()
        return Response(
            {
                'deleted_runs': count,
                'deleted_total_objects': total_deleted,
                'message': f'Removed {count} symbol run(s) for this strategy.',
            },
            status=status.HTTP_200_OK,
        )

    # Legacy run_symbol_backtest removed.

    # Legacy bulk snapshot endpoint removed; use /symbol-runs/run-bulk.

    # Legacy snapshot-symbols and symbol-snapshots endpoints removed; use /symbol-runs/summary and DELETE /symbol-runs.


class StrategyAssignmentViewSet(viewsets.ModelViewSet):
    """ViewSet for StrategyAssignment"""
    queryset = StrategyAssignment.objects.all().select_related('strategy', 'symbol')
    serializer_class = StrategyAssignmentSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    ordering_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        queryset = super().get_queryset()
        symbol_ticker = self.request.query_params.get('symbol_ticker', None)
        
        if symbol_ticker:
            # Filter for symbol-specific assignments OR global assignments
            queryset = queryset.filter(Q(symbol__ticker=symbol_ticker) | Q(symbol__isnull=True))
        else:
            # If no symbol_ticker, only show global assignments
            queryset = queryset.filter(symbol__isnull=True)
        
        return queryset
    
    @action(detail=False, methods=['get'], url_path='symbol/(?P<ticker>[^/.]+)')
    def by_symbol(self, request, ticker=None):
        """Get all strategy assignments for a specific symbol (including global)"""
        try:
            symbol = Symbol.objects.get(ticker=ticker)
            assignments = StrategyAssignment.objects.filter(
                Q(symbol=symbol) | Q(symbol__isnull=True)
            ).select_related('strategy', 'symbol')
            serializer = self.get_serializer(assignments, many=True)
            return Response(serializer.data)
        except Symbol.DoesNotExist:
            return Response(
                {'error': f'Symbol {ticker} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
