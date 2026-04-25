"""
Shared backtest creation + Celery queue (used by BacktestViewSet and strategy symbol snapshots).
"""

import logging
from django.db.models import Q
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response

from backtest_engine.models import Backtest
from backtest_engine.serializers import BacktestSerializer
from backtest_engine.tasks import run_backtest_task
from backtest_engine.services.hybrid_vix_hedge import resolved_hedge_config_for_backtest
from market_data.models import Symbol
from strategies.models import StrategyDefinition, StrategyAssignment

logger = logging.getLogger(__name__)


def create_backtest_and_enqueue(data):
    """
    Internal helper used by both HTTP endpoints and bulk Celery launchers.

    ``data`` must match BacktestCreateSerializer.validated_data shape.
    Returns: (backtest, celery_task_id)
    Raises: ValueError for expected validation-ish errors.
    """
    try:
        strategy = StrategyDefinition.objects.get(id=data['strategy_id'])
    except StrategyDefinition.DoesNotExist:
        raise ValueError(f'Strategy with id {data["strategy_id"]} not found')

    symbols = []
    symbol_tickers = data.get('symbol_tickers', [])
    broker_id = data.get('broker_id')
    exchange_code = data.get('exchange_code', '')

    if broker_id:
        from live_trading.models import Broker, SymbolBrokerAssociation
        from market_data.models import Exchange

        try:
            broker = Broker.objects.get(id=broker_id)
        except Broker.DoesNotExist:
            raise ValueError(f'Broker with id {broker_id} not found')

        associations = SymbolBrokerAssociation.objects.filter(
            broker=broker,
            symbol__status='active',
        ).filter(Q(long_active=True) | Q(short_active=True))

        pmodes_create = data.get('position_modes') or ['long', 'short']
        has_long = 'long' in pmodes_create
        has_short = 'short' in pmodes_create
        if has_long and not has_short:
            associations = associations.filter(long_active=True)
        elif has_short and not has_long:
            associations = associations.filter(short_active=True)

        if exchange_code:
            try:
                exchange = Exchange.objects.get(code=exchange_code)
                associations = associations.filter(symbol__exchange=exchange)
            except Exchange.DoesNotExist:
                raise ValueError(f'Exchange with code {exchange_code} not found')

        symbols = [assoc.symbol for assoc in associations.select_related('symbol')]

        if symbol_tickers and len(symbol_tickers) > 0:
            symbols = [s for s in symbols if s.ticker in symbol_tickers]

        if not symbols:
            exchange_text = f' on exchange {exchange_code}' if exchange_code else ''
            raise ValueError(f'No symbols found for broker {broker.name}{exchange_text}')
    else:
        if not symbol_tickers or len(symbol_tickers) == 0:
            symbols = list(Symbol.objects.filter(status='active'))
            if not symbols:
                raise ValueError(
                    'No active symbols found. Please provide symbol_tickers or use broker-based filtering.'
                )
        else:
            for ticker in symbol_tickers:
                try:
                    symbol = Symbol.objects.get(ticker=ticker, status='active')
                    symbols.append(symbol)
                except Symbol.DoesNotExist:
                    raise ValueError(f'Symbol {ticker} not found or is not active')

    strategy_assignment = None
    if len(symbols) == 1:
        strategy_assignment = StrategyAssignment.objects.filter(
            strategy=strategy,
            symbol=symbols[0],
        ).first()

    if not strategy_assignment:
        strategy_assignment = StrategyAssignment.objects.filter(
            strategy=strategy,
            symbol__isnull=True,
        ).first()

    strategy_parameters = strategy.default_parameters.copy()
    if strategy_assignment:
        strategy_parameters.update(strategy_assignment.parameters)
    strategy_parameters.update(data.get('strategy_parameters', {}))

    start_date = data.get('start_date')
    end_date = data.get('end_date')
    if not start_date:
        start_date = None
    if not end_date:
        end_date = timezone.now()

    broker = None
    if broker_id:
        from live_trading.models import Broker

        try:
            broker = Broker.objects.get(id=broker_id)
        except Broker.DoesNotExist:
            broker = None

    pmodes = data.get('position_modes') or ['long', 'short']

    backtest = Backtest.objects.create(
        name=data.get('name', ''),
        strategy=strategy,
        strategy_assignment=strategy_assignment,
        broker=broker,
        start_date=start_date,
        end_date=end_date,
        split_ratio=data.get('split_ratio', 0.7),
        initial_capital=data.get('initial_capital', 10000.0),
        bet_size_percentage=data.get('bet_size_percentage', 100.0),
        strategy_parameters=strategy_parameters,
        hedge_enabled=bool(data.get('hedge_enabled', False)),
        run_strategy_only_baseline=bool(data.get('run_strategy_only_baseline', True)),
        hedge_config=(
            resolved_hedge_config_for_backtest(data.get('hedge_config'))
            if bool(data.get('hedge_enabled', False))
            else {}
        ),
        position_modes=pmodes,
        status='pending',
    )
    backtest.symbols.set(symbols)

    task = run_backtest_task.delay(backtest.id)
    logger.info('Started backtest task for backtest %s, task_id: %s', backtest.id, task.id)
    return backtest, task.id


def create_backtest_from_validated_data(data):
    """
    Create a Backtest row, attach symbols, enqueue run_backtest_task.

    ``data`` must match BacktestCreateSerializer.validated_data shape.
    Returns a DRF Response (201 or error status).
    """
    try:
        backtest, task_id = create_backtest_and_enqueue(data)
        serializer = BacktestSerializer(backtest)
        response_data = serializer.data
        response_data['task_id'] = task_id
        return Response(response_data, status=status.HTTP_201_CREATED)
    except ValueError as ve:
        msg = str(ve)
        http_status = status.HTTP_400_BAD_REQUEST
        if msg.startswith('Strategy with id') or msg.startswith('Broker with id') or msg.startswith('Symbol '):
            http_status = status.HTTP_404_NOT_FOUND
        return Response({'error': msg}, status=http_status)
    except Exception as e:
        logger.error('Error starting backtest task: %s', str(e))
        return Response(
            {'error': f'Error starting backtest: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
