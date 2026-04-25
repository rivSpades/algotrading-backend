"""
API Views for Strategies
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.filters import SearchFilter, OrderingFilter
from django.utils import timezone
from django.db import transaction
from django.utils.dateparse import parse_datetime
from django.db.models import Q, Count, Max, OuterRef, Subquery
from .models import StrategyDefinition, StrategyAssignment
from .serializers import StrategyDefinitionSerializer, StrategyAssignmentSerializer
from market_data.models import Symbol
from market_data.serializers import SymbolListSerializer
from backtest_engine.models import (
    Backtest,
    SymbolBacktestRun,
    SymbolBacktestTrade,
    SymbolBacktestStatistics,
    SymbolBacktestParameterSet,
)
from backtest_engine.position_modes import normalize_position_modes
from backtest_engine.services.create_backtest import create_backtest_from_validated_data
from backtest_engine.tasks import run_symbol_backtest_run_task, bulk_symbol_runs_queue_task
from backtest_engine.parameter_sets import build_symbol_run_parameter_payload, signature_for_payload
from backtest_engine.position_modes import normalize_position_modes
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
            'parameter_set': run_row.parameter_set_id,
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
            # Exclude orphan rows created during earlier failures (e.g. before parameter_set signature was set).
            # All valid runs should have `parameter_set` populated.
            SymbolBacktestRun.objects.filter(strategy=strategy, symbol=sym)
            .exclude(parameter_set__isnull=True)
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

        def _parse_dt(val, default_dt):
            if val is None or val == '':
                return default_dt
            if hasattr(val, 'utcoffset'):
                dt = val
            else:
                s = str(val).strip()
                # DRF/JS often send ISO with 'Z'; parse_datetime may require +00:00.
                if s.endswith('Z'):
                    s = s[:-1] + '+00:00'
                dt = parse_datetime(s)
                if dt is None:
                    return default_dt
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt, timezone=timezone.utc)
            return dt

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
            parameter_set=None,  # set below after signature computed
            start_date=_parse_dt(start_date, timezone.now().replace(year=1900, month=1, day=1)),
            end_date=_parse_dt(end_date, timezone.now()),
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
        payload = build_symbol_run_parameter_payload(
            strategy_id=strategy.id,
            broker_id=broker.id if broker else None,
            start_date=run.start_date,
            end_date=run.end_date,
            split_ratio=run.split_ratio,
            initial_capital=run.initial_capital,
            bet_size_percentage=run.bet_size_percentage,
            strategy_parameters=run.strategy_parameters,
            position_modes=run.position_modes,
            hedge_enabled=run.hedge_enabled,
            run_strategy_only_baseline=run.run_strategy_only_baseline,
            hedge_config=run.hedge_config,
        )
        sig = signature_for_payload(payload)
        ps, _created = SymbolBacktestParameterSet.objects.get_or_create(
            signature=sig,
            defaults={'strategy': strategy, 'broker': broker, 'parameters': payload, 'label': str(name)[:200]},
        )
        if not ps.label and name:
            ps.label = str(name)[:200]
            ps.save(update_fields=['label'])
        run.parameter_set = ps
        run.save(update_fields=['parameter_set'])
        task = run_symbol_backtest_run_task.delay(run.id)
        return Response(
            {
                'id': run.id,
                'parameter_set': ps.signature,
                'parameter_set_label': ps.label,
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
                'parameter_set': run.parameter_set_id,
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
        """Distinct tickers that have at least one stored SymbolBacktestRun for this strategy (paginated)."""
        strategy = self.get_object()
        parameter_set = request.query_params.get('parameter_set') or None
        # Do NOT use query param `search`: SearchFilter on this viewset also uses it and would
        # filter the StrategyDefinition queryset so get_object() 404s. Prefer `ticker_q`.
        search = (request.query_params.get('ticker_q') or request.query_params.get('q') or '').strip()
        base_qs = SymbolBacktestRun.objects.filter(strategy=strategy)
        if parameter_set:
            base_qs = base_qs.filter(parameter_set_id=parameter_set)
        if search:
            base_qs = base_qs.filter(symbol__ticker__icontains=search)

        # One DB row per ticker, including "latest" run fields via subquery.
        latest_run_qs = base_qs.filter(symbol__ticker=OuterRef('symbol__ticker')).order_by('-created_at')
        summary_qs = (
            base_qs.values('symbol__ticker')
            .annotate(
                snapshot_count=Count('id'),
                latest_run_id=Subquery(latest_run_qs.values('id')[:1]),
                latest_run_status=Subquery(latest_run_qs.values('status')[:1]),
                snapshot_updated_at=Subquery(latest_run_qs.values('updated_at')[:1]),
                parameter_set=Subquery(latest_run_qs.values('parameter_set_id')[:1]),
                parameter_set_label=Subquery(latest_run_qs.values('parameter_set__label')[:1]),
                latest_created_at=Max('created_at'),
            )
            .order_by('symbol__ticker')
        )

        # Paginate.
        from rest_framework.pagination import PageNumberPagination

        paginator = PageNumberPagination()
        paginator.page_size_query_param = 'page_size'
        paginator.max_page_size = 200
        page = paginator.paginate_queryset(summary_qs, request)
        page_rows = list(page or [])

        tickers = [r.get('symbol__ticker') for r in page_rows if r.get('symbol__ticker')]
        symbols = Symbol.objects.filter(ticker__in=tickers)
        sym_map = {s.ticker: s for s in symbols}

        results = []
        for row in page_rows:
            t = row.get('symbol__ticker')
            sym = sym_map.get(t)
            if not sym:
                continue
            sym_data = dict(SymbolListSerializer(sym).data)
            sym_data['snapshot_count'] = row.get('snapshot_count', 0)
            sym_data['latest_run_id'] = row.get('latest_run_id')
            sym_data['latest_run_status'] = row.get('latest_run_status')
            su = row.get('snapshot_updated_at')
            sym_data['snapshot_updated_at'] = su.isoformat() if hasattr(su, 'isoformat') and su else None
            sym_data['parameter_set'] = row.get('parameter_set')
            sym_data['parameter_set_label'] = row.get('parameter_set_label') or ''
            results.append(sym_data)

        return paginator.get_paginated_response(results)

    @action(detail=True, methods=['get'], url_path='symbol-run-parameter-sets')
    def symbol_run_parameter_sets(self, request, pk=None):
        """List parameter sets for this strategy (for UI dropdowns)."""
        strategy = self.get_object()
        qs = (
            SymbolBacktestParameterSet.objects.filter(strategy=strategy)
            .order_by('-created_at')
        )
        rows = [
            {
                'signature': ps.signature,
                'label': ps.label or '',
                'created_at': ps.created_at.isoformat() if ps.created_at else None,
            }
            for ps in qs
        ]
        return Response({'parameter_sets': rows})

    @action(detail=True, methods=['delete'], url_path=r'symbol-run-parameter-sets/(?P<signature>[0-9a-f]{64})')
    def delete_symbol_run_parameter_set(self, request, pk=None, signature=None):
        """Delete a parameter set AND all SymbolBacktestRuns linked to it for this strategy."""
        strategy = self.get_object()
        try:
            ps = SymbolBacktestParameterSet.objects.get(signature=signature, strategy=strategy)
        except SymbolBacktestParameterSet.DoesNotExist:
            return Response({'error': 'Parameter set not found'}, status=status.HTTP_404_NOT_FOUND)

        qs = SymbolBacktestRun.objects.filter(strategy=strategy, parameter_set_id=ps.signature)
        run_count = qs.count()
        with transaction.atomic():
            total_deleted, _by_model = qs.delete()
            ps.delete()
        return Response(
            {
                'deleted_runs': run_count,
                'deleted_total_objects': total_deleted + 1,  # plus parameter set row
                'parameter_set': signature,
                'message': f'Deleted parameter set and {run_count} run(s).',
            },
            status=status.HTTP_200_OK,
        )

    @action(detail=True, methods=['get'], url_path=r'symbol-run-parameter-sets/(?P<signature>[0-9a-f]{64})/sharpe-heatmap')
    def symbol_run_parameter_set_sharpe_heatmap(self, request, pk=None, signature=None):
        """
        Per-symbol risk/return metrics for a parameter set (used for charts):
        sharpe + max_drawdown for LONG/SHORT.
        Uses the latest run per symbol under this parameter_set.
        """
        strategy = self.get_object()
        try:
            ps = SymbolBacktestParameterSet.objects.get(signature=signature, strategy=strategy)
        except SymbolBacktestParameterSet.DoesNotExist:
            return Response({'error': 'Parameter set not found'}, status=status.HTTP_404_NOT_FOUND)

        runs = (
            SymbolBacktestRun.objects.filter(strategy=strategy, parameter_set=ps)
            .select_related('symbol')
            .order_by('symbol__ticker', '-created_at')
        )
        latest_by_ticker = {}
        for r in runs:
            t = r.symbol.ticker
            if t not in latest_by_ticker:
                latest_by_ticker[t] = r

        # Load symbol-level stats rows for those runs.
        run_ids = [r.id for r in latest_by_ticker.values()]
        stats_rows = (
            SymbolBacktestStatistics.objects.filter(run_id__in=run_ids, symbol__isnull=False)
            .select_related('run', 'symbol')
        )
        # IMPORTANT: do not key only by run_id; in rare cases duplicates can exist.
        # Always match the stats row for the run's symbol.
        stats_by_run_symbol = {(s.run_id, s.symbol_id): s for s in stats_rows}

        cells = []
        for t in sorted(latest_by_ticker.keys()):
            run = latest_by_ticker[t]
            stats = stats_by_run_symbol.get((run.id, run.symbol_id))
            if not stats:
                cells.append(
                    {
                        'ticker': t,
                        'run_id': run.id,
                        'long': {'sharpe': None, 'max_drawdown': None},
                        'short': {'sharpe': None, 'max_drawdown': None},
                    }
                )
                continue
            modes = normalize_position_modes(run.position_modes)
            primary = 'long' if 'long' in modes else 'short'
            secondary = 'short' if primary == 'long' else 'long'

            primary_sharpe = float(stats.sharpe_ratio) if stats.sharpe_ratio is not None else None
            primary_dd = float(stats.max_drawdown) if stats.max_drawdown is not None else None
            extra = stats.additional_stats if isinstance(stats.additional_stats, dict) else {}
            sec_block = extra.get(secondary) or {}
            secondary_sharpe = None
            if isinstance(sec_block, dict) and sec_block.get('sharpe_ratio') is not None:
                try:
                    secondary_sharpe = float(sec_block.get('sharpe_ratio'))
                except (TypeError, ValueError):
                    secondary_sharpe = None
            secondary_dd = None
            if isinstance(sec_block, dict) and sec_block.get('max_drawdown') is not None:
                try:
                    secondary_dd = float(sec_block.get('max_drawdown'))
                except (TypeError, ValueError):
                    secondary_dd = None

            long_metrics = (
                {'sharpe': primary_sharpe, 'max_drawdown': primary_dd}
                if primary == 'long'
                else {'sharpe': secondary_sharpe, 'max_drawdown': secondary_dd}
            )
            short_metrics = (
                {'sharpe': primary_sharpe, 'max_drawdown': primary_dd}
                if primary == 'short'
                else {'sharpe': secondary_sharpe, 'max_drawdown': secondary_dd}
            )

            cells.append({'ticker': t, 'run_id': run.id, 'long': long_metrics, 'short': short_metrics})

        return Response(
            {
                'parameter_set': ps.signature,
                'label': ps.label or '',
                'cells': cells,
            }
        )

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
