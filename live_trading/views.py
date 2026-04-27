"""API Views for Live Trading.

The viewsets here cover:
- Brokers and symbol/broker associations
- Live trades (read-only)
- Strategy deployments (V2) — CRUD, preview-symbols, lifecycle, events
"""

import logging

from celery.result import AsyncResult
from django.db import transaction
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.filters import OrderingFilter, SearchFilter
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from algo_trading_backend.celery import app as celery_app

from market_data.models import Symbol
from market_data.models import ExchangeSchedule
from backtest_engine.models import SymbolBacktestParameterSet
from strategies.models import StrategyDefinition

from .models import (
    Broker,
    DeploymentEvent,
    DeploymentSymbol,
    LiveTrade,
    StrategyDeployment,
    SymbolBrokerAssociation,
)
from .serializers import (
    BrokerSerializer,
    BrokerSymbolLinkSerializer,
    DeploymentEventSerializer,
    DeploymentSymbolSerializer,
    LiveTradeSerializer,
    StrategyDeploymentCoreSerializer,
    StrategyDeploymentCreateSerializer,
    StrategyDeploymentListSerializer,
    StrategyDeploymentPreviewSerializer,
    SymbolBrokerAssociationSerializer,
)
from backtest_engine.services.hybrid_vix_hedge import resolved_hedge_config_for_backtest

from .services import (
    EngineNotRegistered,
    build_symbol_candidates,
    compute_deployment_metrics,
    evaluate_deployment,
    evaluate_deployment_for_promotion,
    evaluate_deployment_symbol,
    exit_open_trades_for_deployment,
    log_event,
    manual_close_live_trade,
    promote_to_real_money,
    queue_snapshot_recalc,
    reconcile_deployment_symbols,
    select_default_symbols,
    update_open_trades,
)
from .services.hedge_inherit import inherit_hedge_from_symbol_runs
from .tasks import reconcile_close_until_sync_task

logger = logging.getLogger(__name__)


def _find_next_occurrence(*, now, open_utc, weekdays):
    """Return the next datetime (UTC) matching open_utc on allowed weekdays."""
    from datetime import datetime, timedelta, timezone as dt_tz

    today = now.date()
    for i in range(0, 10):
        d = today + timedelta(days=i)
        if weekdays and d.isoweekday() not in weekdays:
            continue
        candidate = timezone.make_aware(datetime.combine(d, open_utc), timezone=dt_tz.utc)
        if candidate > now:
            return candidate
    return None


def _find_prev_occurrence(*, now, open_utc, weekdays):
    """Return the most recent datetime (UTC) <= now matching open_utc on allowed weekdays."""
    from datetime import datetime, timedelta, timezone as dt_tz

    today = now.date()
    for i in range(0, 10):
        d = today - timedelta(days=i)
        if weekdays and d.isoweekday() not in weekdays:
            continue
        candidate = timezone.make_aware(datetime.combine(d, open_utc), timezone=dt_tz.utc)
        if candidate <= now:
            return candidate
    return None


class BrokerPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class BrokerSymbolPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class DeploymentEventPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200


class DeploymentSymbolPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = 'page_size'
    max_page_size = 200


class LiveTradePagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 200


def _actor_for(request) -> tuple[str, str]:
    """Return (actor_type, actor_id) extracted from a DRF request."""
    user = getattr(request, 'user', None)
    if user is not None and getattr(user, 'is_authenticated', False):
        return 'user', str(getattr(user, 'id', ''))
    return 'system', ''


class BrokerViewSet(viewsets.ModelViewSet):
    """ViewSet for Broker."""

    queryset = Broker.objects.all()
    serializer_class = BrokerSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'code']
    ordering_fields = ['name', 'code', 'created_at']
    ordering = ['name']
    pagination_class = BrokerPagination

    def filter_queryset(self, queryset):
        if self.action == 'symbols':
            return queryset
        return super().filter_queryset(queryset)

    @action(detail=True, methods=['get'], url_path='symbols')
    def symbols(self, request, pk=None):
        broker = get_object_or_404(Broker, pk=pk)
        associations = SymbolBrokerAssociation.objects.filter(broker=broker).select_related(
            'symbol', 'symbol__exchange',
        )

        search_query = (
            request.query_params.get('search', None)
            or request.query_params.get('symbol_search', None)
        )
        if search_query:
            associations = associations.filter(
                Q(symbol__ticker__icontains=search_query)
                | Q(symbol__name__icontains=search_query),
            )

        paginator = BrokerSymbolPagination()
        page = paginator.paginate_queryset(associations, request)
        if page is not None:
            serializer = SymbolBrokerAssociationSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = SymbolBrokerAssociationSerializer(associations, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'], url_path='link-symbols')
    def link_symbols(self, request, pk=None):
        broker = self.get_object()
        serializer = BrokerSymbolLinkSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data
        symbol_tickers = data.get('symbol_tickers', [])
        exchange_code = data.get('exchange_code', '')
        link_all_available = data.get('link_all_available', False)
        verify_capabilities = data.get('verify_capabilities', True)

        from .tasks import link_broker_symbols_task

        try:
            task = link_broker_symbols_task.delay(
                broker_id=broker.id,
                symbol_tickers=symbol_tickers if symbol_tickers else None,
                exchange_code=exchange_code if exchange_code else None,
                link_all_available=link_all_available,
                verify_capabilities=verify_capabilities,
            )
            logger.info('Started symbol linking task for broker %s, task_id: %s', broker.id, task.id)
            return Response(
                {
                    'message': 'Symbol linking task started',
                    'task_id': task.id,
                    'broker_id': broker.id,
                    'broker_name': broker.name,
                },
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            logger.error('Error starting symbol linking task: %s', e)
            return Response(
                {'error': f'Error starting symbol linking task: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=True, methods=['post'], url_path='reverify-symbols')
    def reverify_symbols(self, request, pk=None):
        broker = self.get_object()
        from .tasks import reverify_broker_symbol_associations_task

        try:
            task = reverify_broker_symbol_associations_task.delay(broker_id=broker.id)
            logger.info('Started reverify symbols task for broker %s, task_id: %s', broker.id, task.id)
            return Response(
                {
                    'message': 'Symbol re-verification task started',
                    'task_id': task.id,
                    'broker_id': broker.id,
                    'broker_name': broker.name,
                },
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            logger.error('Error starting reverify symbols task: %s', e)
            return Response(
                {'error': f'Error starting reverify task: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _get_adapter_for_type(self, broker, deployment_type):
        from .adapters.factory import get_broker_adapter

        if deployment_type == 'paper':
            if not broker.has_paper_trading_credentials():
                return None, Response(
                    {'error': 'Paper trading credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            return get_broker_adapter(broker, paper_trading=True), None
        if deployment_type == 'real_money':
            if not broker.has_real_money_credentials():
                return None, Response(
                    {'error': 'Real money credentials not configured.'},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            return get_broker_adapter(broker, paper_trading=False), None
        return None, Response(
            {'error': 'Invalid deployment_type. Must be "paper" or "real_money".'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    @action(detail=True, methods=['post'], url_path='test-connection')
    def test_connection(self, request, pk=None):
        broker = self.get_object()
        deployment_type = request.data.get('deployment_type', 'paper')
        adapter, err = self._get_adapter_for_type(broker, deployment_type)
        if err is not None:
            return err
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}. Only ALPACA is currently supported.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            is_valid = adapter.verify_credentials()
            field = 'paper_trading_active' if deployment_type == 'paper' else 'real_money_active'
            setattr(broker, field, bool(is_valid))
            broker.save()
            if is_valid:
                return Response({
                    'success': True,
                    'message': f'{deployment_type.replace("_", " ").title()} connection test successful.',
                    'deployment_type': deployment_type,
                    'is_active': True,
                })
            return Response({
                'success': False,
                'message': f'{deployment_type.replace("_", " ").title()} connection test failed.',
                'deployment_type': deployment_type,
                'is_active': False,
            }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            field = 'paper_trading_active' if deployment_type == 'paper' else 'real_money_active'
            setattr(broker, field, False)
            broker.save()
            return Response(
                {'error': f'Connection test failed: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(detail=True, methods=['get'], url_path='account-balance')
    def account_balance(self, request, pk=None):
        broker = self.get_object()
        deployment_type = request.query_params.get('deployment_type', 'paper')
        adapter, err = self._get_adapter_for_type(broker, deployment_type)
        if err is not None:
            return err
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            balance = adapter.get_account_balance()
            equity = adapter.get_account_equity()
            return Response({
                'balance': str(balance),
                'equity': str(equity),
                'deployment_type': deployment_type,
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to get account balance: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(detail=True, methods=['get'], url_path='check-symbol')
    def check_symbol(self, request, pk=None):
        broker = self.get_object()
        symbol = request.query_params.get('symbol', '').upper()
        deployment_type = request.query_params.get('deployment_type', 'paper')

        if not symbol:
            return Response(
                {'error': 'Symbol parameter is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        adapter, err = self._get_adapter_for_type(broker, deployment_type)
        if err is not None:
            return err
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            is_tradable = adapter.is_symbol_tradable(symbol)
            capabilities = adapter.get_symbol_capabilities(symbol)
            current_price = adapter.get_current_price(symbol)
            return Response({
                'symbol': symbol,
                'is_tradable': is_tradable,
                'capabilities': {
                    'long_supported': capabilities.get('long_supported', False),
                    'short_supported': capabilities.get('short_supported', False),
                },
                'current_price': str(current_price) if current_price else None,
                'deployment_type': deployment_type,
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to check symbol: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @action(detail=True, methods=['get'], url_path='positions')
    def positions(self, request, pk=None):
        broker = self.get_object()
        deployment_type = request.query_params.get('deployment_type', 'paper')
        adapter, err = self._get_adapter_for_type(broker, deployment_type)
        if err is not None:
            return err
        if not adapter:
            return Response(
                {'error': f'Broker adapter not found for code: {broker.code}'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            positions = adapter.get_all_positions()
            positions_data = [{
                'symbol': pos.symbol,
                'quantity': str(pos.quantity),
                'average_price': str(pos.average_price),
                'current_price': str(pos.current_price),
                'unrealized_pnl': str(pos.unrealized_pnl),
                'position_type': pos.position_type,
            } for pos in positions]
            return Response({
                'positions': positions_data,
                'count': len(positions_data),
                'deployment_type': deployment_type,
            })
        except Exception as e:
            return Response(
                {'error': f'Failed to get positions: {str(e)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )


class SymbolBrokerAssociationViewSet(viewsets.ModelViewSet):
    """ViewSet for SymbolBrokerAssociation."""

    queryset = SymbolBrokerAssociation.objects.select_related('symbol', 'broker').all()
    serializer_class = SymbolBrokerAssociationSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['symbol__ticker', 'broker__name', 'broker__code']
    ordering_fields = ['symbol__ticker', 'broker__name', 'updated_at']
    ordering = ['broker__name', 'symbol__ticker']

    def get_queryset(self):
        queryset = super().get_queryset()

        broker_id = self.request.query_params.get('broker', None)
        if broker_id:
            queryset = queryset.filter(broker_id=broker_id)

        symbol_ticker = self.request.query_params.get('symbol', None)
        if symbol_ticker:
            queryset = queryset.filter(symbol__ticker=symbol_ticker)

        long_active = self.request.query_params.get('long_active', None)
        if long_active is not None:
            queryset = queryset.filter(long_active=long_active.lower() == 'true')

        short_active = self.request.query_params.get('short_active', None)
        if short_active is not None:
            queryset = queryset.filter(short_active=short_active.lower() == 'true')

        return queryset


class LiveTradeViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for LiveTrade (read-only)."""

    queryset = LiveTrade.objects.select_related(
        'deployment', 'deployment__strategy', 'symbol',
    ).all()
    serializer_class = LiveTradeSerializer
    pagination_class = LiveTradePagination
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['symbol__ticker', 'deployment__name', 'broker_order_id']
    ordering_fields = ['entry_timestamp', 'exit_timestamp', 'pnl']
    ordering = ['-entry_timestamp']

    def get_queryset(self):
        queryset = super().get_queryset()

        deployment_id = self.request.query_params.get('deployment', None)
        if deployment_id:
            queryset = queryset.filter(deployment_id=deployment_id)

        symbol_ticker = self.request.query_params.get('symbol', None)
        if symbol_ticker:
            queryset = queryset.filter(symbol__ticker=symbol_ticker)

        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)

        deployment_type = self.request.query_params.get('deployment_type', None)
        if deployment_type:
            queryset = queryset.filter(deployment__deployment_type=deployment_type)

        omit_hedge = str(self.request.query_params.get('omit_hedge_legs', '')).lower() in (
            '1', 'true', 'yes',
        )
        if omit_hedge and deployment_id:
            # Strategy rows only; nest hedge legs on each row via `hedge_legs`.
            # Do NOT use exclude(metadata__is_hedge_leg=True): for rows where the
            # key is missing, JSON lookups are NULL and SQL "NOT (x = true)" drops
            # them, which hid every normal (non-hedge) trade and showed an empty list.
            queryset = queryset.filter(
                Q(metadata__is_hedge_leg__isnull=True)
                | Q(metadata__is_hedge_leg=False)
                | ~Q(metadata__has_key='is_hedge_leg')
            )

        return queryset

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        if getattr(self, 'action', None) != 'list':
            return ctx
        dep = self.request.query_params.get('deployment')
        omit_hedge = str(self.request.query_params.get('omit_hedge_legs', '')).lower() in (
            '1', 'true', 'yes',
        )
        if dep and omit_hedge:
            by_parent: dict = {}
            for h in (
                LiveTrade.objects.filter(
                    deployment_id=dep,
                    metadata__is_hedge_leg=True,
                )
                .select_related('symbol', 'deployment', 'deployment__strategy')
            ):
                raw_pid = (h.metadata or {}).get('hedge_parent_live_trade_id')
                if raw_pid is None:
                    continue
                try:
                    pid = int(raw_pid)
                except (TypeError, ValueError):
                    continue
                by_parent.setdefault(pid, []).append(h)
            ctx['hedge_by_parent'] = by_parent
        return ctx

    @action(detail=True, methods=['post'], url_path='manual-close')
    def manual_close(self, request, pk=None):
        """Manually submit a market close for a single `LiveTrade`.

        Optional JSON body: ``{"force": true}`` to skip broker position pre-check and
        submit using app ``trade_type``/``quantity`` (when the broker reports no row
        but the trade is still open in the app, or wrong account until fixed).
        """
        trade = self.get_object()
        actor_type, actor_id = _actor_for(request)

        def _truthy(val):
            if val is None:
                return False
            if isinstance(val, bool):
                return val
            return str(val).lower() in ('1', 'true', 'yes')

        trust_db = _truthy(request.data.get('force')) or _truthy(
            request.query_params.get('force')
        )

        outcome = manual_close_live_trade(
            trade,
            actor_type=actor_type,
            actor_id=actor_id,
            trust_db=trust_db,
        )
        payload = outcome.to_dict()
        if (
            not trust_db
            and outcome.live_trade_id
            and outcome.status not in ('failed', 'skipped')
        ):
            async_result = reconcile_close_until_sync_task.delay(outcome.live_trade_id)
            payload['reconcile_task_id'] = async_result.id
        return Response(payload)

    @action(detail=False, methods=['get'], url_path='reconcile-close-status')
    def reconcile_close_status(self, request):
        """Poll Celery result for `reconcile_close_until_sync_task` (browser → wait for close)."""
        task_id = request.query_params.get('task_id')
        if not task_id:
            return Response(
                {'error': 'task_id query parameter is required'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        r = AsyncResult(task_id, app=celery_app)
        if r.state == 'PENDING':
            return Response({'ready': False, 'state': r.state})
        if r.state == 'FAILURE':
            err = str(r.result) if r.result is not None else 'task failed'
            return Response(
                {'ready': True, 'state': r.state, 'status': 'error', 'error': err},
            )
        if r.state == 'SUCCESS':
            result = r.result if isinstance(r.result, dict) else {'value': r.result}
            return Response({'ready': True, 'state': r.state, 'result': result})
        return Response(
            {
                'ready': False,
                'state': r.state,
                'meta': r.info if isinstance(r.info, dict) else None,
            },
        )


# ---------------------------------------------------------------------------
# Strategy Deployment (v2)
# ---------------------------------------------------------------------------


class StrategyDeploymentPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 200


class StrategyDeploymentViewSet(viewsets.ModelViewSet):
    """CRUD + lifecycle endpoints for `StrategyDeployment`."""

    queryset = StrategyDeployment.objects.select_related(
        'strategy', 'broker', 'parameter_set', 'parent_deployment',
    ).all()
    pagination_class = StrategyDeploymentPagination
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['name', 'strategy__name', 'broker__name', 'parameter_set__label']
    ordering_fields = ['created_at', 'updated_at', 'started_at', 'last_signal_at']
    ordering = ['-created_at']

    def get_serializer_class(self):
        if self.action == 'list':
            return StrategyDeploymentListSerializer
        if self.action == 'create':
            return StrategyDeploymentCreateSerializer
        if self.action == 'preview_symbols':
            return StrategyDeploymentPreviewSerializer
        return StrategyDeploymentCoreSerializer

    def get_queryset(self):
        qs = super().get_queryset()

        strategy_id = self.request.query_params.get('strategy')
        if strategy_id:
            qs = qs.filter(strategy_id=strategy_id)

        deployment_type = self.request.query_params.get('deployment_type')
        if deployment_type:
            qs = qs.filter(deployment_type=deployment_type)

        status_filter = self.request.query_params.get('status')
        if status_filter:
            qs = qs.filter(status=status_filter)

        broker_id = self.request.query_params.get('broker')
        if broker_id:
            qs = qs.filter(broker_id=broker_id)

        parameter_set = self.request.query_params.get('parameter_set')
        if parameter_set:
            qs = qs.filter(parameter_set_id=parameter_set)

        return qs

    # ------------------------------------------------------------------
    # Create / lifecycle
    # ------------------------------------------------------------------

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated = serializer.validated_data
        symbol_overrides = validated.pop('symbol_overrides', []) or []
        use_default_symbols = validated.pop('use_default_symbols', True)
        hedge_enabled_req = validated.pop('hedge_enabled', None)
        hedge_config_req = validated.pop('hedge_config', None)
        position_mode = validated.get('position_mode', 'long')
        parameter_set = validated['parameter_set']
        strategy = validated['strategy']

        inherited_hedge_en, inherited_hedge_cfg = inherit_hedge_from_symbol_runs(
            strategy, parameter_set,
        )
        if hedge_enabled_req is not None:
            final_hedge_en = bool(hedge_enabled_req)
        else:
            final_hedge_en = bool(inherited_hedge_en)
        if not final_hedge_en:
            final_hedge_cfg = {}
        elif hedge_config_req is not None:
            final_hedge_cfg = resolved_hedge_config_for_backtest(hedge_config_req)
        else:
            final_hedge_cfg = resolved_hedge_config_for_backtest(
                dict(inherited_hedge_cfg or {}),
            )

        with transaction.atomic():
            deployment = StrategyDeployment.objects.create(
                deployment_type='paper',
                status='pending',
                strategy_parameters=validated.get(
                    'strategy_parameters',
                    parameter_set.parameters or {},
                ),
                hedge_enabled=final_hedge_en,
                hedge_config=final_hedge_cfg,
                **{k: v for k, v in validated.items() if k != 'strategy_parameters'},
            )

            if use_default_symbols and not symbol_overrides:
                candidates = select_default_symbols(parameter_set, position_mode)
                deployment_symbols = self._materialize_default_symbols(
                    deployment, candidates, position_mode,
                )
            else:
                deployment_symbols = self._materialize_overrides(
                    deployment, parameter_set, position_mode, symbol_overrides,
                )

        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            event_type='deploy_created',
            actor_type=actor_type,
            actor_id=actor_id,
            message=(
                f"Created paper deployment for {deployment.strategy.name} "
                f"with {len(deployment_symbols)} symbol(s)"
            ),
            context={
                'parameter_set': parameter_set.signature,
                'broker': deployment.broker.code,
                'position_mode': position_mode,
                'symbol_count': len(deployment_symbols),
                'used_default_symbols': use_default_symbols and not symbol_overrides,
                'hedge_enabled': final_hedge_en,
                'hedge_inherited': hedge_enabled_req is None,
            },
        )
        for ds in deployment_symbols:
            log_event(
                deployment,
                deployment_symbol=ds,
                event_type='symbol_added',
                actor_type=actor_type,
                actor_id=actor_id,
                message=f"Symbol {ds.symbol.ticker} enrolled (priority={ds.priority}, color={ds.color_overall})",
                context={
                    'ticker': ds.symbol.ticker,
                    'priority': ds.priority,
                    'tier': ds.tier,
                    'color_long': ds.color_long,
                    'color_short': ds.color_short,
                    'color_overall': ds.color_overall,
                },
            )

        detail = StrategyDeploymentCoreSerializer(deployment).data
        return Response(detail, status=status.HTTP_201_CREATED)

    def _materialize_default_symbols(self, deployment, candidates, position_mode):
        rows = []
        for index, candidate in enumerate(candidates):
            ds = DeploymentSymbol.objects.create(
                deployment=deployment,
                symbol=candidate.symbol,
                position_mode=position_mode,
                status='active',
                sharpe_long=candidate.sharpe_long,
                sharpe_short=candidate.sharpe_short,
                max_dd_long=candidate.max_dd_long,
                max_dd_short=candidate.max_dd_short,
                total_trades_long=candidate.total_trades_long,
                total_trades_short=candidate.total_trades_short,
                color_long=candidate.color_long,
                color_short=candidate.color_short,
                color_overall=candidate.color_overall,
                tier=candidate.tier,
                priority=index,
                last_evaluated_at=timezone.now(),
            )
            rows.append(ds)
        return rows

    def _materialize_overrides(self, deployment, parameter_set, position_mode, overrides):
        # Build the candidate map from parameter set so overrides inherit colors/tier.
        # Symbol's primary key is its ticker, so symbol_id == ticker for all uses here.
        candidates = build_symbol_candidates(parameter_set, position_mode).candidates
        by_ticker = {c.symbol.ticker.upper(): c for c in candidates}

        rows = []
        for index, override in enumerate(overrides):
            ticker = (override.get('ticker') or override.get('symbol_id') or '').upper()
            symbol = None
            if ticker:
                symbol = Symbol.objects.filter(ticker=ticker).first()
            if not symbol:
                continue
            candidate = by_ticker.get(symbol.ticker.upper())
            kwargs = {
                'deployment': deployment,
                'symbol': symbol,
                'position_mode': override.get('position_mode', position_mode),
                'status': 'active',
                'priority': override.get('priority', index),
                'last_evaluated_at': timezone.now(),
            }
            if candidate is not None:
                kwargs.update(
                    sharpe_long=candidate.sharpe_long,
                    sharpe_short=candidate.sharpe_short,
                    max_dd_long=candidate.max_dd_long,
                    max_dd_short=candidate.max_dd_short,
                    total_trades_long=candidate.total_trades_long,
                    total_trades_short=candidate.total_trades_short,
                    color_long=candidate.color_long,
                    color_short=candidate.color_short,
                    color_overall=candidate.color_overall,
                    tier=candidate.tier,
                )
            rows.append(DeploymentSymbol.objects.create(**kwargs))
        return rows

    @action(detail=False, methods=['post'], url_path='preview-symbols')
    def preview_symbols(self, request):
        """Return the green-default selection without persisting anything."""
        serializer = StrategyDeploymentPreviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        signature = serializer.validated_data['parameter_set']
        position_mode = serializer.validated_data['position_mode']
        default_only = serializer.validated_data['default_only']

        try:
            ps = SymbolBacktestParameterSet.objects.select_related('strategy').get(signature=signature)
        except SymbolBacktestParameterSet.DoesNotExist:
            return Response(
                {'error': 'Parameter set not found'},
                status=status.HTTP_404_NOT_FOUND,
            )

        result = build_symbol_candidates(ps, position_mode)
        payload = result.to_preview_payload(default_only=default_only)
        payload['strategy'] = ps.strategy_id
        payload['parameter_set_label'] = ps.label or ''
        return Response(payload)

    @action(
        detail=False,
        methods=['get'],
        url_path='hedge-inherit-preview',
    )
    def hedge_inherit_preview(self, request):
        """What hybrid VIX hedge the backend would use when `hedge_enabled` is omitted on create."""
        strategy_id = request.query_params.get('strategy')
        ps_sig = request.query_params.get('parameter_set') or request.query_params.get('parameter_set_signature')
        if not strategy_id or not ps_sig:
            return Response(
                {'error': 'Query params `strategy` and `parameter_set` (parameter set signature) are required.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            strategy = StrategyDefinition.objects.get(pk=strategy_id)
        except StrategyDefinition.DoesNotExist:
            return Response(
                {'error': 'Strategy not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            ps = SymbolBacktestParameterSet.objects.get(signature=ps_sig)
        except SymbolBacktestParameterSet.DoesNotExist:
            return Response(
                {'error': 'Parameter set not found for that signature.'},
                status=status.HTTP_404_NOT_FOUND,
            )
        he, cfg = inherit_hedge_from_symbol_runs(strategy, ps)
        return Response({
            'hedge_enabled': he,
            'hedge_config': cfg,
            'resolved_hedge_config': resolved_hedge_config_for_backtest(cfg) if he else {},
        })

    @action(detail=True, methods=['post'], url_path='activate')
    def activate(self, request, pk=None):
        deployment = self.get_object()
        if deployment.status not in ('pending', 'paused'):
            return Response(
                {'error': f"Cannot activate deployment in status '{deployment.status}'."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        deployment.status = 'active'
        if not deployment.activated_at:
            deployment.activated_at = timezone.now()
        if not deployment.started_at:
            deployment.started_at = timezone.now()
        deployment.save(update_fields=['status', 'activated_at', 'started_at', 'updated_at'])
        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            event_type='deploy_activated',
            actor_type=actor_type,
            actor_id=actor_id,
            message='Deployment activated',
        )
        return Response(StrategyDeploymentCoreSerializer(deployment).data)

    @action(detail=True, methods=['post'], url_path='pause')
    def pause(self, request, pk=None):
        deployment = self.get_object()
        if deployment.status not in ('active', 'evaluating'):
            return Response(
                {'error': f"Cannot pause deployment in status '{deployment.status}'."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        deployment.status = 'paused'
        deployment.save(update_fields=['status', 'updated_at'])
        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            event_type='deploy_paused',
            actor_type=actor_type,
            actor_id=actor_id,
            message='Deployment paused',
        )
        return Response(StrategyDeploymentCoreSerializer(deployment).data)

    @action(detail=True, methods=['post'], url_path='stop')
    def stop(self, request, pk=None):
        deployment = self.get_object()
        if deployment.status == 'stopped':
            return Response(
                {'error': 'Deployment already stopped.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        actor_type, actor_id = _actor_for(request)

        exit_summary = exit_open_trades_for_deployment(
            deployment,
            actor_type=actor_type,
            actor_id=actor_id,
        )
        deployment.status = 'stopped'
        deployment.save(update_fields=['status', 'updated_at'])
        log_event(
            deployment,
            event_type='deploy_stopped',
            actor_type=actor_type,
            actor_id=actor_id,
            message='Deployment stopped (exit-all submitted)',
            context={'exit': exit_summary},
        )
        detail = StrategyDeploymentCoreSerializer(deployment).data
        detail['stop_exit'] = exit_summary
        return Response(detail)

    def destroy(self, request, *args, **kwargs):
        """Delete a deployment only when it has no open trades.

        Safety: deletion is blocked if any `LiveTrade(status='open')` exists for
        the deployment (including exit orders that were submitted but not yet
        reflected as closed).
        """
        deployment = self.get_object()
        open_trades = deployment.live_trades.filter(status='open').count()
        if open_trades > 0:
            return Response(
                {'error': 'Cannot delete deployment while trades are still open.', 'open_trades': open_trades},
                status=status.HTTP_409_CONFLICT,
            )
        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            event_type='deploy_deleted',
            actor_type=actor_type,
            actor_id=actor_id,
            message='Deployment deleted',
        )
        return super().destroy(request, *args, **kwargs)

    @action(detail=True, methods=['post'], url_path='evaluate')
    def evaluate(self, request, pk=None):
        """Score a deployment against its `evaluation_criteria` and persist
        the breakdown on `evaluation_results`.

        Body (all optional):
          - `transition_status` (bool, default True) — when True, also flips
            the deployment's status to `passed` / `failed`.
        """
        deployment = self.get_object()
        if deployment.deployment_type != 'paper':
            return Response(
                {'error': 'Only paper deployments can be evaluated.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        transition_status = bool(request.data.get('transition_status', True))
        actor_type, actor_id = _actor_for(request)

        result = evaluate_deployment_for_promotion(
            deployment,
            actor_type=actor_type,
            actor_id=actor_id,
            save=True,
            transition_status=transition_status,
        )

        return Response({
            'deployment_id': deployment.id,
            'status': deployment.status,
            'can_promote_to_real_money': deployment.can_promote_to_real_money(),
            'evaluation': result.to_dict(),
        })

    @action(detail=True, methods=['get'], url_path='evaluation-preview')
    def evaluation_preview(self, request, pk=None):
        """Return current metrics and the criteria check breakdown without
        persisting anything. Used by the frontend to render a 'before you
        promote' summary panel.
        """
        deployment = self.get_object()
        actor_type, actor_id = _actor_for(request)
        result = evaluate_deployment_for_promotion(
            deployment,
            actor_type=actor_type,
            actor_id=actor_id,
            save=False,
            transition_status=False,
        )
        return Response({
            'deployment_id': deployment.id,
            'current_status': deployment.status,
            'can_promote_to_real_money': deployment.can_promote_to_real_money(),
            'metrics': compute_deployment_metrics(deployment),
            'criteria': dict(deployment.evaluation_criteria or {}),
            'evaluation': result.to_dict(),
        })

    @action(detail=True, methods=['post'], url_path='promote-to-real-money')
    def promote_to_real_money(self, request, pk=None):
        """Promote a paper deployment to a sibling real-money deployment.

        Body (all optional):
          - `name_suffix` (str, default ' (Live)') — appended to the new
            deployment's name to distinguish it from the paper sibling.
        """
        deployment = self.get_object()
        if deployment.deployment_type != 'paper':
            return Response(
                {'error': 'Only paper deployments can be promoted.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        actor_type, actor_id = _actor_for(request)

        if not deployment.broker.has_real_money_credentials():
            return Response(
                {
                    'error': (
                        'Deployment cannot be promoted. Requires a broker with '
                        'configured real-money credentials.'
                    ),
                    'current_status': deployment.status,
                    'broker_has_real_credentials': deployment.broker.has_real_money_credentials(),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        name_suffix = request.data.get('name_suffix') or ' (Live)'

        try:
            with transaction.atomic():
                real = promote_to_real_money(
                    deployment,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    name_suffix=str(name_suffix),
                )
        except ValueError as exc:
            return Response(
                {'error': str(exc), 'current_status': deployment.status},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                'paper_deployment_id': deployment.id,
                'real_deployment_id': real.id,
                'real_deployment': StrategyDeploymentCoreSerializer(real).data,
            },
            status=status.HTTP_201_CREATED,
        )

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------

    @action(detail=True, methods=['get'], url_path='symbols')
    def list_symbols(self, request, pk=None):
        deployment = self.get_object()
        qs = (
            deployment.deployment_symbols.select_related('symbol', 'symbol__exchange')
            .order_by('priority', 'symbol__ticker')
        )
        paginator = DeploymentSymbolPagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        if page is not None:
            serializer = DeploymentSymbolSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        return Response(DeploymentSymbolSerializer(qs, many=True).data)

    @action(detail=True, methods=['post'], url_path=r'symbols/(?P<deployment_symbol_id>\d+)/disable')
    def disable_symbol(self, request, pk=None, deployment_symbol_id=None):
        deployment = self.get_object()
        ds = get_object_or_404(
            DeploymentSymbol, pk=deployment_symbol_id, deployment=deployment,
        )
        ds.status = 'disabled'
        ds.save(update_fields=['status', 'updated_at'])
        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            deployment_symbol=ds,
            event_type='symbol_disabled',
            actor_type=actor_type,
            actor_id=actor_id,
            message=f"Symbol {ds.symbol.ticker} disabled (manual)",
        )
        return Response(DeploymentSymbolSerializer(ds).data)

    @action(detail=True, methods=['post'], url_path=r'symbols/(?P<deployment_symbol_id>\d+)/enable')
    def enable_symbol(self, request, pk=None, deployment_symbol_id=None):
        deployment = self.get_object()
        ds = get_object_or_404(
            DeploymentSymbol, pk=deployment_symbol_id, deployment=deployment,
        )
        ds.status = 'active'
        ds.save(update_fields=['status', 'updated_at'])
        actor_type, actor_id = _actor_for(request)
        log_event(
            deployment,
            deployment_symbol=ds,
            event_type='symbol_enabled',
            actor_type=actor_type,
            actor_id=actor_id,
            message=f"Symbol {ds.symbol.ticker} re-enabled (manual)",
        )
        return Response(DeploymentSymbolSerializer(ds).data)

    # ------------------------------------------------------------------
    # Manual engine fire (testing helper)
    # ------------------------------------------------------------------

    @action(detail=True, methods=['post'], url_path='fire-now')
    def fire_now(self, request, pk=None):
        """Manually run the live engine for this deployment.

        Body (all optional):
        - `deployment_symbol_ids`: list of `DeploymentSymbol` ids to limit the
          fire to. When omitted, all active symbols are evaluated in priority
          order.
        - `include_disabled`: if true, include disabled symbols too.

        Returns the per-symbol engine evaluations and a summary so the
        frontend / a developer can verify signal logic without waiting for
        the next scheduled tick.
        """

        deployment = self.get_object()
        if deployment.status not in ('pending', 'active', 'evaluating', 'paused'):
            return Response(
                {
                    'error': (
                        f"Cannot fire engine while deployment status is "
                        f"'{deployment.status}'."
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        only_ids = request.data.get('deployment_symbol_ids') or None
        if only_ids is not None and not isinstance(only_ids, list):
            return Response(
                {'error': 'deployment_symbol_ids must be a list of integers.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        include_disabled = bool(request.data.get('include_disabled', False))
        place_orders = bool(request.data.get('place_orders', False))
        actor_type, actor_id = _actor_for(request)

        try:
            evaluations = evaluate_deployment(
                deployment,
                actor_type=actor_type,
                actor_id=actor_id or 'manual',
                only_symbol_ids=only_ids,
                include_disabled=include_disabled,
                place_orders=place_orders,
            )
        except EngineNotRegistered as exc:
            return Response(
                {'error': str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        actionable = sum(
            1 for ev in evaluations if ev.signal and ev.signal.is_actionable
        )
        return Response({
            'deployment_id': deployment.id,
            'fired_at': timezone.now().isoformat(),
            'symbol_count': len(evaluations),
            'actionable_count': actionable,
            'place_orders': place_orders,
            'evaluations': [ev.to_dict() for ev in evaluations],
        })

    @action(
        detail=True,
        methods=['post'],
        url_path=r'symbols/(?P<deployment_symbol_id>\d+)/fire-now',
    )
    def fire_symbol_now(self, request, pk=None, deployment_symbol_id=None):
        """Run the engine for a single deployment symbol synchronously."""
        deployment = self.get_object()
        ds = get_object_or_404(
            DeploymentSymbol, pk=deployment_symbol_id, deployment=deployment,
        )
        place_orders = bool(request.data.get('place_orders', False))
        actor_type, actor_id = _actor_for(request)
        try:
            evaluation = evaluate_deployment_symbol(
                ds,
                actor_type=actor_type,
                actor_id=actor_id or 'manual',
                place_orders=place_orders,
            )
        except EngineNotRegistered as exc:
            return Response(
                {'error': str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(evaluation.to_dict())

    @action(detail=True, methods=['post'], url_path='update-positions')
    def update_positions(self, request, pk=None):
        """Reconcile open `LiveTrade` rows against broker-reported positions."""
        deployment = self.get_object()
        actor_type, actor_id = _actor_for(request)
        summary = update_open_trades(
            deployment,
            actor_type=actor_type,
            actor_id=actor_id or 'manual',
        )
        return Response(summary)

    @action(detail=True, methods=['post'], url_path='recalc-snapshots')
    def recalc_snapshots(self, request, pk=None):
        """Queue per-symbol snapshot recalc backtests for this deployment.

        Body (all optional):
        - `tickers`: list of symbol tickers to limit the recalc to.
        - `enqueue`: if false, only resolve the candidate runs without
          dispatching the celery task (useful for tests).
        - `reconcile`: if true (default), immediately run a reconciliation
          pass against whatever snapshot stats are currently committed.
        """
        deployment = self.get_object()
        actor_type, actor_id = _actor_for(request)
        only_tickers = request.data.get('tickers')
        enqueue = bool(request.data.get('enqueue', True))
        do_reconcile = bool(request.data.get('reconcile', True))

        if only_tickers is not None and not isinstance(only_tickers, list):
            return Response(
                {'error': 'tickers must be a list of strings.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        recalc = queue_snapshot_recalc(
            deployment,
            actor_type=actor_type,
            actor_id=actor_id or 'manual',
            only_tickers=only_tickers,
            enqueue=enqueue,
        )
        result = {'recalc': recalc}
        if do_reconcile:
            summary = reconcile_deployment_symbols(
                deployment,
                actor_type=actor_type,
                actor_id=actor_id or 'manual',
            )
            result['reconcile'] = summary.to_dict()
        return Response(result)

    # ------------------------------------------------------------------
    # Events / signals / statistics
    # ------------------------------------------------------------------

    def _events_queryset(self, deployment, request):
        qs = DeploymentEvent.objects.filter(deployment=deployment).select_related(
            'deployment_symbol__symbol',
        )
        event_type = request.query_params.get('event_type')
        if event_type:
            qs = qs.filter(event_type=event_type)
        actor_type = request.query_params.get('actor_type')
        if actor_type:
            qs = qs.filter(actor_type=actor_type)
        level = request.query_params.get('level')
        if level:
            qs = qs.filter(level=level)
        since = request.query_params.get('since')
        if since:
            qs = qs.filter(created_at__gte=since)
        return qs

    @action(detail=True, methods=['get'], url_path='events')
    def events(self, request, pk=None):
        deployment = self.get_object()
        qs = self._events_queryset(deployment, request)
        paginator = DeploymentEventPagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        if page is not None:
            serializer = DeploymentEventSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        return Response(DeploymentEventSerializer(qs, many=True).data)

    @action(detail=True, methods=['get'], url_path='signals')
    def signals(self, request, pk=None):
        deployment = self.get_object()
        qs = DeploymentEvent.objects.filter(
            deployment=deployment,
            event_type__in=['signal_evaluated', 'order_placed', 'order_filled', 'order_failed'],
        ).select_related('deployment_symbol__symbol')
        since = request.query_params.get('since')
        if since:
            qs = qs.filter(created_at__gte=since)
        paginator = DeploymentEventPagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        if page is not None:
            serializer = DeploymentEventSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)
        return Response(DeploymentEventSerializer(qs, many=True).data)

    @action(detail=True, methods=['get'], url_path='statistics')
    def statistics(self, request, pk=None):
        deployment = self.get_object()
        trades_qs = LiveTrade.objects.filter(deployment=deployment).select_related('symbol')
        open_trades_qs = trades_qs.filter(status='open')
        closed_trades_qs = trades_qs.filter(status='closed').exclude(pnl__isnull=True)

        from django.db.models import Sum

        realized_pnl = closed_trades_qs.aggregate(total=Sum('pnl'))['total'] or 0
        open_count = open_trades_qs.count()
        closed_count = closed_trades_qs.count()
        winning_trades = closed_trades_qs.filter(is_winner=True).count()
        win_rate = (winning_trades / closed_count) if closed_count else None

        # “Total invested” approximated as current exposure from open trades.
        total_invested_open = 0
        for t in open_trades_qs:
            try:
                total_invested_open += (t.entry_price or 0) * (t.quantity or 0)
            except Exception:
                continue

        # Equity curve from realised PnL over time (closed trades by exit time).
        equity_curve = []
        equity = float(deployment.initial_capital or 0)
        for t in closed_trades_qs.order_by('exit_timestamp', 'id'):
            equity += float(t.pnl or 0)
            equity_curve.append(
                {
                    'timestamp': (t.exit_timestamp.isoformat() if t.exit_timestamp else t.entry_timestamp.isoformat()),
                    'equity': equity,
                    'pnl': float(t.pnl or 0),
                    'ticker': t.symbol.ticker,
                    'trade_id': t.id,
                }
            )

        symbol_count = deployment.deployment_symbols.count()
        active_symbol_count = deployment.deployment_symbols.filter(status='active').count()

        metrics = compute_deployment_metrics(deployment)
        return Response({
            'status': deployment.status,
            'symbol_count': symbol_count,
            'active_symbol_count': active_symbol_count,
            'open_trades': open_count,
            'closed_trades': closed_count,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'current_pnl': str(realized_pnl),
            'total_pnl': str(realized_pnl),
            'total_invested_open': str(total_invested_open),
            'equity_curve': equity_curve,
            'metrics': metrics,
            'last_signal_at': deployment.last_signal_at,
        })


class DeploymentEventViewSet(viewsets.ReadOnlyModelViewSet):
    """Global read-only feed for `DeploymentEvent` rows."""

    queryset = DeploymentEvent.objects.select_related(
        'deployment', 'deployment__strategy', 'deployment_symbol__symbol',
    ).all()
    serializer_class = DeploymentEventSerializer
    pagination_class = DeploymentEventPagination
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['message', 'event_type', 'actor_id']
    ordering_fields = ['created_at']
    ordering = ['-created_at']

    def get_queryset(self):
        qs = super().get_queryset()
        deployment = self.request.query_params.get('deployment')
        if deployment:
            qs = qs.filter(deployment_id=deployment)
        event_type = self.request.query_params.get('event_type')
        if event_type:
            qs = qs.filter(event_type=event_type)
        actor_type = self.request.query_params.get('actor_type')
        if actor_type:
            qs = qs.filter(actor_type=actor_type)
        level = self.request.query_params.get('level')
        if level:
            qs = qs.filter(level=level)
        since = self.request.query_params.get('since')
        if since:
            qs = qs.filter(created_at__gte=since)
        return qs


class MarketOpenProgressViewSet(viewsets.ViewSet):
    """Grouped exchange open countdown/progress for the dashboard."""

    def list(self, request):
        from datetime import datetime, timezone as dt_tz

        now = timezone.now()
        schedules = (
            ExchangeSchedule.objects.filter(active=True)
            .select_related('exchange')
            .order_by('open_utc', 'exchange__code')
        )

        groups = {}
        for s in schedules:
            gk = s.open_group_key()
            groups.setdefault(gk, []).append(s)

        out = []
        for gk, rows in groups.items():
            sample = rows[0]
            open_utc = sample.open_utc
            close_utc = sample.close_utc
            weekdays = sample.weekday_list() or [1, 2, 3, 4, 5]

            next_open = _find_next_occurrence(now=now, open_utc=open_utc, weekdays=weekdays)
            prev_open = _find_prev_occurrence(now=now, open_utc=open_utc, weekdays=weekdays)
            if next_open is None or prev_open is None:
                continue

            # Determine whether the market is currently open for this group.
            session_date = prev_open.date()
            close_at = timezone.make_aware(datetime.combine(session_date, close_utc), timezone=dt_tz.utc)
            is_open = prev_open <= now < close_at
            seconds_to_close = max(0, int((close_at - now).total_seconds())) if is_open else None

            total = max(1.0, (next_open - prev_open).total_seconds())
            elapsed = (now - prev_open).total_seconds()
            progress = min(1.0, max(0.0, elapsed / total))
            seconds_to_open = max(0, int((next_open - now).total_seconds()))

            out.append(
                {
                    'open_group_key': gk,
                    'open_utc': open_utc.strftime('%H:%M'),
                    'close_utc': close_utc.strftime('%H:%M'),
                    'weekdays': weekdays,
                    'exchanges': sorted({r.exchange.code for r in rows}),
                    'next_open_at': next_open.isoformat(),
                    'seconds_to_open': seconds_to_open,
                    'progress': (None if is_open else round(progress, 6)),
                    'is_open': bool(is_open),
                    'close_at': close_at.isoformat(),
                    'seconds_to_close': seconds_to_close,
                }
            )

        out.sort(key=lambda r: (r['seconds_to_open'], r['open_group_key']))
        return Response({'results': out, 'as_of': now.isoformat()})
