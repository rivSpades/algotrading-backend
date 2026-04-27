"""Celery tasks for live trading.

Includes:
- Broker-symbol linking and re-verification (carried over from earlier phases).
- `market_open_fanout` — per `ExchangeSchedule.open_group_key` beat: refreshes
  SPY + VIXM + VIXY + ^VIX (hedge / dashboard) OHLCV **once**, then enqueues
  `deployment_market_open` in symbol chunks per matching deployment (see
  `LIVE_TRADING_MARKET_OPEN_SYMBOLS_PER_TASK`).
- `deployment_market_open` — refreshes OHLCV for each enrolled symbol for that
  open group, then runs the live engine.

Weekend snapshot recalc and order-placement tasks land in later phases but
the placeholder beat for the weekend job is registered alongside the market
open ones (see `live_trading.services.scheduling`).
"""

import logging
import time

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.db import close_old_connections, connections
from django.shortcuts import get_object_or_404
from django.utils import timezone

from market_data.models import Exchange, Symbol

from .models import (
    Broker,
    DeploymentSymbol,
    LiveTrade,
    StrategyDeployment,
    SymbolBrokerAssociation,
)
from .services import (
    EngineNotRegistered,
    deployment_symbols_for_open_group,
    deployments_for_open_group,
    evaluate_deployment_for_promotion,
    evaluate_deployment_symbol,
    get_adapter_for_deployment,
    get_engine_instance,
    log_event,
    queue_snapshot_recalc,
    reconcile_deployment_symbols,
    update_open_trades,
)

logger = logging.getLogger(__name__)

# Cap wall time spent in synchronous OHLCV `.apply()` subtasks so market-open always reaches
# engine evaluation (yfinance / Alpaca can otherwise block the worker for many minutes).
OHLCV_REFRESH_BUDGET_SEC = int(8 * 60)
HEDGE_BENCH_OHLCV_BUDGET_SEC = int(5 * 60)


@shared_task(bind=True, name='live_trading.link_broker_symbols')
def link_broker_symbols_task(
    self,
    broker_id,
    symbol_tickers=None,
    exchange_code=None,
    link_all_available=False,
    verify_capabilities=True,
):
    """Link symbols to a broker asynchronously.

    Args:
        broker_id: Broker primary key.
        symbol_tickers: optional list of tickers to link.
        exchange_code: optional exchange code to link all symbols from.
        link_all_available: if True, discover broker-tradable symbols and
            link those that already exist locally and have no association.
        verify_capabilities: if True, query the broker for long/short support.
    """
    try:
        broker = Broker.objects.get(id=broker_id)

        symbols_to_link = []
        broker_symbols_data = None

        if link_all_available:
            from .adapters.factory import get_broker_adapter

            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': 10,
                    'message': f'Getting tradable symbols from broker {broker.name}...',
                },
            )

            adapter = get_broker_adapter(broker, paper_trading=True)
            if not adapter and broker.has_real_money_credentials():
                adapter = get_broker_adapter(broker, paper_trading=False)

            if not adapter:
                return {
                    'status': 'error',
                    'error': 'Broker must have at least paper trading or real money credentials configured and active',
                }

            try:
                broker_symbols_data = adapter.get_all_symbols_with_capabilities()
                broker_symbols = list(broker_symbols_data.keys())

                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': 30,
                        'message': f'Found {len(broker_symbols)} tradable symbols, filtering...',
                    },
                )

                db_symbols = Symbol.objects.filter(ticker__in=broker_symbols)
                symbols_with_broker = SymbolBrokerAssociation.objects.values_list(
                    'symbol_id', flat=True,
                ).distinct()
                symbols_to_link = [
                    s for s in db_symbols if s.ticker not in symbols_with_broker
                ]

                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': 50,
                        'message': f'Found {len(symbols_to_link)} symbols to link',
                    },
                )
            except Exception as e:
                logger.error('Error getting tradable symbols from broker: %s', e)
                return {
                    'status': 'error',
                    'error': f'Failed to get tradable symbols from broker: {str(e)}',
                }
        else:
            if symbol_tickers:
                symbols = Symbol.objects.filter(ticker__in=symbol_tickers)
                symbols_to_link.extend(symbols)

            if exchange_code:
                exchange = get_object_or_404(Exchange, code=exchange_code)
                exchange_symbols = Symbol.objects.filter(exchange=exchange)
                symbols_to_link.extend(exchange_symbols)

            symbols_to_link = list(set(symbols_to_link))

            if not symbols_to_link:
                return {
                    'status': 'error',
                    'error': 'No symbols found to link',
                }

            existing_associations = SymbolBrokerAssociation.objects.filter(
                broker=broker,
                symbol__in=symbols_to_link,
            ).values_list('symbol_id', flat=True)

            existing_tickers = set(existing_associations)
            symbols_to_link = [s for s in symbols_to_link if s.ticker not in existing_tickers]

        if not symbols_to_link:
            return {
                'status': 'success',
                'message': 'No symbols to link (all already linked)',
                'created': 0,
                'skipped': 0,
                'total': 0,
            }

        total_symbols = len(symbols_to_link)
        created_count = 0
        failed_count = 0

        if verify_capabilities and not link_all_available:
            from .adapters.factory import get_broker_adapter

            adapter = get_broker_adapter(broker, paper_trading=True)
            if not adapter and broker.has_real_money_credentials():
                adapter = get_broker_adapter(broker, paper_trading=False)

            if adapter and hasattr(adapter, 'get_all_symbols_with_capabilities'):
                try:
                    broker_symbols_data = adapter.get_all_symbols_with_capabilities()
                except Exception as e:
                    logger.warning('Could not get bulk capabilities: %s', e)
                    broker_symbols_data = None
        elif not verify_capabilities:
            adapter = None

        associations_to_create = []

        for index, symbol in enumerate(symbols_to_link):
            try:
                long_active = False
                short_active = False

                if verify_capabilities:
                    if broker_symbols_data and symbol.ticker in broker_symbols_data:
                        capabilities = broker_symbols_data[symbol.ticker]
                        long_active = capabilities.get('long_supported', False)
                        short_active = capabilities.get('short_supported', False)
                    elif adapter:
                        try:
                            capabilities = adapter.get_symbol_capabilities(symbol.ticker)
                            long_active = capabilities.get('long_supported', False)
                            short_active = capabilities.get('short_supported', False)
                        except Exception as e:
                            logger.error('Error verifying capabilities for %s: %s', symbol.ticker, e)

                associations_to_create.append(
                    SymbolBrokerAssociation(
                        symbol=symbol,
                        broker=broker,
                        long_active=long_active,
                        short_active=short_active,
                        verified_at=timezone.now() if verify_capabilities and (long_active or short_active) else None,
                    ),
                )

                if (index + 1) % 100 == 0 or (index + 1) == total_symbols:
                    progress = 50 + int((index + 1) / total_symbols * 50)
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'progress': progress,
                            'message': f'Prepared {index + 1}/{total_symbols} symbols for linking...',
                        },
                    )
            except Exception as e:
                logger.error('Error preparing symbol %s: %s', symbol.ticker, e)
                failed_count += 1

        if associations_to_create:
            try:
                SymbolBrokerAssociation.objects.bulk_create(
                    associations_to_create,
                    batch_size=500,
                    ignore_conflicts=False,
                )
                created_count = len(associations_to_create)
                logger.info('Bulk created %s symbol-broker associations', created_count)
            except Exception as e:
                logger.error('Error bulk creating associations: %s', e)
                for assoc in associations_to_create:
                    try:
                        assoc.save()
                        created_count += 1
                    except Exception as e2:
                        logger.error('Error creating association for %s: %s', assoc.symbol.ticker, e2)
                        failed_count += 1

        return {
            'status': 'success',
            'message': f'Processed {total_symbols} symbols',
            'created': created_count,
            'failed': failed_count,
            'total': total_symbols,
            'broker_id': broker_id,
            'broker_name': broker.name,
        }

    except Broker.DoesNotExist:
        return {'status': 'error', 'error': f'Broker with id {broker_id} not found'}
    except Exception as e:
        logger.error('Error in link_broker_symbols_task: %s', e, exc_info=True)
        return {'status': 'error', 'error': str(e)}


@shared_task(bind=True, name='live_trading.reverify_broker_symbol_associations')
def reverify_broker_symbol_associations_task(self, broker_id):
    """Re-fetch broker tradability for every association on this broker."""
    try:
        broker = Broker.objects.get(id=broker_id)
        associations = list(
            SymbolBrokerAssociation.objects.filter(broker=broker).select_related('symbol'),
        )
        if not associations:
            return {
                'status': 'success',
                'updated': 0,
                'failed': 0,
                'total': 0,
                'message': 'No linked symbols to re-verify',
                'broker_id': broker_id,
                'broker_name': broker.name,
            }

        self.update_state(
            state='PROGRESS',
            meta={'progress': 5, 'message': f'Re-verifying {len(associations)} linked symbols…'},
        )

        from .adapters.factory import get_broker_adapter

        adapter = get_broker_adapter(broker, paper_trading=True)
        if not adapter and broker.has_real_money_credentials():
            adapter = get_broker_adapter(broker, paper_trading=False)
        if not adapter:
            return {
                'status': 'error',
                'error': 'Broker must have paper or real-money credentials configured to verify capabilities',
            }

        broker_symbols_data = None
        if hasattr(adapter, 'get_all_symbols_with_capabilities'):
            try:
                broker_symbols_data = adapter.get_all_symbols_with_capabilities()
            except Exception as e:
                logger.warning('Could not get bulk capabilities for reverify: %s', e)

        now = timezone.now()
        failed_count = 0
        total = len(associations)

        for index, assoc in enumerate(associations):
            ticker = assoc.symbol.ticker
            long_active = False
            short_active = False
            try:
                if broker_symbols_data and ticker in broker_symbols_data:
                    capabilities = broker_symbols_data[ticker]
                    long_active = bool(capabilities.get('long_supported', False))
                    short_active = bool(capabilities.get('short_supported', False))
                else:
                    capabilities = adapter.get_symbol_capabilities(ticker)
                    long_active = bool(capabilities.get('long_supported', False))
                    short_active = bool(capabilities.get('short_supported', False))
            except Exception as e:
                logger.error('Error re-verifying capabilities for %s: %s', ticker, e)
                failed_count += 1

            assoc.long_active = long_active
            assoc.short_active = short_active
            assoc.verified_at = now if (long_active or short_active) else None
            assoc.updated_at = now

            if (index + 1) % 100 == 0 or (index + 1) == total:
                progress = 5 + int((index + 1) / total * 90)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'message': f'Re-verified {index + 1}/{total} symbols…',
                    },
                )

        SymbolBrokerAssociation.objects.bulk_update(
            associations,
            ['long_active', 'short_active', 'verified_at', 'updated_at'],
            batch_size=500,
        )

        return {
            'status': 'success',
            'updated': total,
            'failed': failed_count,
            'total': total,
            'broker_id': broker_id,
            'broker_name': broker.name,
        }
    except Broker.DoesNotExist:
        return {'status': 'error', 'error': f'Broker with id {broker_id} not found'}
    except Exception as e:
        logger.error('Error in reverify_broker_symbol_associations_task: %s', e, exc_info=True)
        return {'status': 'error', 'error': str(e)}


# ---------------------------------------------------------------------------
# Live trading scheduled tasks (Phase 4)
# ---------------------------------------------------------------------------


@shared_task(
    bind=True,
    name='live_trading.market_open_fanout',
    soft_time_limit=12 * 60,
    time_limit=14 * 60,
)
def market_open_fanout_task(self, open_group_key: str) -> dict:
    """Fan-out task fired once per `ExchangeSchedule.open_group_key`.

    Refreshes SPY, VIXM, VIXY, and ^VIX (once per run, always) so dashboard /
    model series are fresh, then looks up every active deployment that has at
    least one active `DeploymentSymbol` whose exchange matches the open group
    and dispatches `deployment_market_open_task` in symbol-ID chunks so the full
    ordered universe is covered (see `LIVE_TRADING_MARKET_OPEN_SYMBOLS_PER_TASK`).
    """

    actor_id = self.request.id or 'manual'
    started_at = timezone.now()
    chunk_size = int(
        getattr(
            settings,
            'LIVE_TRADING_MARKET_OPEN_SYMBOLS_PER_TASK',
            250,
        )
    )
    if chunk_size < 1:
        chunk_size = 1

    deployments_qs, exchange_codes = deployments_for_open_group(open_group_key)
    deployments = list(deployments_qs)
    deployment_ids = [d.id for d in deployments]

    log_event(
        None,
        event_type='task_tick',
        actor_type='task',
        actor_id=actor_id,
        message=f"market_open_fanout for {open_group_key}",
        context={
            'open_group_key': open_group_key,
            'exchanges': exchange_codes,
            'deployments': deployment_ids,
            'started_at': started_at.isoformat(),
        },
    )

    # Benchmark / vol-sleeve series: always first (dashboard + any hedged backtests),
    # independent of per-deployment `hedge_enabled`, then per-deployment symbols.
    _refresh_hedge_bench_ohlcv(open_group_key=open_group_key, actor_id=actor_id)

    chunk_tasks_dispatched = 0
    for dep in deployments:
        id_list = list(
            deployment_symbols_for_open_group(
                dep, open_group_key,
            ).values_list('id', flat=True),
        )
        if not id_list:
            continue
        for i in range(0, len(id_list), chunk_size):
            chunk = id_list[i : i + chunk_size]
            try:
                deployment_market_open_task.delay(
                    dep.id,
                    open_group_key,
                    only_symbol_ids=chunk,
                )
                chunk_tasks_dispatched += 1
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    'Failed to dispatch deployment_market_open chunk for deployment=%s: %s',
                    dep.id, exc,
                )

    connections.close_all()
    return {
        'status': 'success',
        'open_group_key': open_group_key,
        'exchanges': exchange_codes,
        'deployment_count': len(deployment_ids),
        'chunk_tasks_dispatched': chunk_tasks_dispatched,
        # Same count as `chunk_tasks_dispatched` (was one task per deployment).
        'dispatched': chunk_tasks_dispatched,
        'symbols_per_task': chunk_size,
    }


@shared_task(
    bind=True,
    name='live_trading.deployment_market_open',
    # OHLCV refresh is capped by OHLCV_REFRESH_BUDGET_SEC but engine+orders on many
    # symbols can still run long; keep hard > soft so workers are not SIGKILL'd.
    soft_time_limit=25 * 60,
    time_limit=28 * 60,
)
def deployment_market_open_task(
    self,
    deployment_id: int,
    open_group_key: str,
    *,
    refresh_ohlcv: bool = True,
    only_symbol_ids=None,
) -> dict:
    """Per-deployment market-open tick (one Celery run = one symbol chunk or full run).

    **Where this fits in the pipeline**
    1. Celery Beat runs the periodic task for an `open_group_key` (see
       `ExchangeSchedule` / `market_open_fanout_task`).
    2. `market_open_fanout_task` finds deployments that have active symbols on
       exchanges in that open group and dispatches one
       `deployment_market_open_task` **per symbol chunk** (default chunk size
       ``LIVE_TRADING_MARKET_OPEN_SYMBOLS_PER_TASK``) so the whole universe is
       covered, not a single 50-row cap.
    3. If this task is invoked **without** ``only_symbol_ids`` and the
       deployment has more symbols than the chunk size, it **orchestrates** by
       enqueueing chunk tasks and returns immediately (no double work on that
       coordinator run).
    4. This task always runs for one ``(deployment_id, open_group_key)``; when
       ``only_symbol_ids`` is set, it processes exactly that ordered slice.

    **Symbol order**
    Active symbols come from `deployment_symbols_for_open_group`, which orders
    rows ``ORDER BY priority ASC, symbol__ticker ASC``.

    **Steps inside this task**
    1. Load `StrategyDeployment` or return error.
    2. If status is not ``active`` or ``evaluating``, log and return skipped.
    3. Build the ordered `DeploymentSymbol` queryset for `open_group_key` (or
       orchestrate chunks if the full set is too large and ``only_symbol_ids``
       is unset), evaluate to a list.
    4. If no symbols, return success with empty evaluations.
    5. If ``refresh_ohlcv`` is true, refresh daily OHLCV for those symbols
       so the engine sees current bars (including today’s open when needed).
       (SPY / VIXM / VIXY / ^VIX are refreshed once per open-group in
       ``market_open_fanout_task`` before this task runs.)
    6. Resolve broker adapter (or None → dry run, no orders).
    7. Build the live engine instance for the deployment
       (`get_engine_instance`).
    8. For each `DeploymentSymbol` **in the list order above**, call
       `evaluate_deployment_symbol` (errors on one symbol do not stop the
       rest); count actionable signals; optionally place orders when an
       adapter exists.
    9. Return a summary dict (evaluations, counts, timestamps).
    """

    actor_id = self.request.id or 'manual'
    chunk_size = int(
        getattr(
            settings,
            'LIVE_TRADING_MARKET_OPEN_SYMBOLS_PER_TASK',
            250,
        )
    )
    if chunk_size < 1:
        chunk_size = 1

    try:
        deployment = StrategyDeployment.objects.select_related(
            'strategy', 'broker', 'parameter_set',
        ).get(id=deployment_id)
    except StrategyDeployment.DoesNotExist:
        logger.warning('deployment_market_open: deployment %s not found', deployment_id)
        return {
            'status': 'error',
            'error': f'Deployment {deployment_id} not found',
            'deployment_id': deployment_id,
        }

    if deployment.status not in ('active', 'evaluating'):
        log_event(
            deployment,
            event_type='task_tick',
            actor_type='task',
            actor_id=actor_id,
            level='warning',
            message=(
                f"Skipping market-open tick: status='{deployment.status}'"
            ),
            context={'open_group_key': open_group_key},
        )
        return {
            'status': 'skipped',
            'reason': f"deployment status '{deployment.status}'",
            'deployment_id': deployment_id,
        }

    base_qs = deployment_symbols_for_open_group(deployment, open_group_key)
    if only_symbol_ids is not None and len(only_symbol_ids) > 0:
        qs = base_qs.filter(id__in=list(only_symbol_ids))
    else:
        # Full-universe manual/shell: if larger than one chunk, fan out like beat.
        try:
            full_id_list = list(base_qs.values_list('id', flat=True))
        except Exception:  # pragma: no cover - defensive
            full_id_list = []
        if len(full_id_list) > chunk_size:
            n_chunks = 0
            for i in range(0, len(full_id_list), chunk_size):
                sub = full_id_list[i : i + chunk_size]
                try:
                    deployment_market_open_task.delay(
                        deployment_id,
                        open_group_key,
                        only_symbol_ids=sub,
                    )
                    n_chunks += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        'deployment_market_open orchestration failed for chunk: %s',
                        exc,
                    )
            log_event(
                deployment,
                event_type='task_tick',
                actor_type='task',
                actor_id=actor_id,
                message=(
                    f"deployment_market_open orchestrated {n_chunks} chunk(s) "
                    f"for {len(full_id_list)} symbol(s) "
                    f"(symbols_per_task={chunk_size})"
                ),
                context={
                    'open_group_key': open_group_key,
                    'orchestrated': True,
                    'chunks': n_chunks,
                    'symbol_count_total': len(full_id_list),
                    'symbols_per_task': chunk_size,
                },
            )
            return {
                'status': 'orchestrated',
                'deployment_id': deployment_id,
                'open_group_key': open_group_key,
                'chunks': n_chunks,
                'symbols_total': len(full_id_list),
                'symbols_per_task': chunk_size,
            }
        qs = base_qs

    try:
        total_available = base_qs.count()
    except Exception:  # pragma: no cover - defensive
        total_available = None
    deployment_symbols = list(qs)

    log_event(
        deployment,
        event_type='task_tick',
        actor_type='task',
        actor_id=actor_id,
        message=(
            f"deployment_market_open for {deployment.strategy.name} "
            f"({len(deployment_symbols)} symbol(s))"
        ),
        context={
            'open_group_key': open_group_key,
            'symbol_count': len(deployment_symbols),
            'symbol_count_universe': total_available,
            'symbol_chunk': bool(only_symbol_ids),
            'symbols_per_task': chunk_size,
            'tickers': [ds.symbol.ticker for ds in deployment_symbols],
        },
    )

    if not deployment_symbols:
        return {
            'status': 'success',
            'deployment_id': deployment_id,
            'open_group_key': open_group_key,
            'symbol_count': 0,
            'evaluations': [],
        }

    if refresh_ohlcv:
        _refresh_ohlcv_for_symbols(deployment, deployment_symbols, actor_id=actor_id)

    broker_adapter = get_adapter_for_deployment(deployment)
    if broker_adapter is None:
        log_event(
            deployment,
            event_type='error',
            actor_type='task',
            actor_id=actor_id,
            level='warning',
            message=(
                f"No broker adapter for {deployment.broker.code}; "
                f"running engine in dry-run (no orders will be placed)."
            ),
            context={'open_group_key': open_group_key},
        )

    try:
        engine = get_engine_instance(deployment, broker_adapter=broker_adapter)
    except EngineNotRegistered as exc:
        log_event(
            deployment,
            event_type='error',
            actor_type='task',
            actor_id=actor_id,
            level='error',
            message=str(exc),
            error=exc,
            context={'open_group_key': open_group_key},
        )
        return {
            'status': 'error',
            'error': str(exc),
            'deployment_id': deployment_id,
        }

    fire_at = timezone.now()
    evaluations: list[dict] = []
    actionable_count = 0
    n_sym = len(deployment_symbols)
    try:
        for i, ds in enumerate(deployment_symbols):
            # Heartbeat every 5 symbols so long stretches without logs are rarer
            # (the last 4 symbols of each block used to have no log → looked "hung").
            if i and (i % 5 == 0):
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'deployment_id': deployment_id,
                        'open_group_key': open_group_key,
                        'processed': i,
                        'total': n_sym,
                        'last_ticker': deployment_symbols[i - 1].symbol.ticker,
                    },
                )
                log_event(
                    deployment,
                    event_type='task_tick',
                    actor_type='task',
                    actor_id=actor_id,
                    message=f"deployment_market_open progress {i}/{n_sym}",
                    context={
                        'open_group_key': open_group_key,
                        'processed': i,
                        'total': n_sym,
                        'last_ticker': deployment_symbols[i - 1].symbol.ticker,
                    },
                )

            try:
                evaluation = evaluate_deployment_symbol(
                    ds,
                    fire_at=fire_at,
                    actor_type='task',
                    actor_id=actor_id,
                    engine=engine,
                    broker_adapter=broker_adapter,
                    place_orders=broker_adapter is not None,
                )
            except Exception:
                # Logged inside evaluate_deployment_symbol; keep going so one
                # bad symbol doesn't cancel the whole tick.
                continue
            try:
                evaluations.append(evaluation.to_dict())
            except Exception as exc:  # pragma: no cover - bad context types
                logger.exception(
                    'deployment_market_open: evaluation.to_dict() failed for %s',
                    ds.symbol.ticker,
                )
                evaluations.append(
                    {
                        'deployment_symbol_id': evaluation.deployment_symbol_id,
                        'ticker': evaluation.ticker,
                        'error': f'to_dict failed: {exc}',
                    }
                )
            if evaluation.signal and evaluation.signal.is_actionable:
                actionable_count += 1

        # After last symbol: explicit phase marker (vs silent gap before DB close / return).
        self.update_state(
            state='PROGRESS',
            meta={
                'deployment_id': deployment_id,
                'open_group_key': open_group_key,
                'processed': n_sym,
                'total': n_sym,
                'phase': 'evaluation_complete',
            },
        )
        log_event(
            deployment,
            event_type='task_tick',
            actor_type='task',
            actor_id=actor_id,
            message=f"deployment_market_open evaluation complete ({n_sym} symbols, {actionable_count} actionable)",
            context={
                'open_group_key': open_group_key,
                'total': n_sym,
                'actionable_count': actionable_count,
            },
        )
    except SoftTimeLimitExceeded:
        log_event(
            deployment,
            event_type='task_tick',
            actor_type='task',
            actor_id=actor_id,
            level='warning',
            message='deployment_market_open soft time limit hit; returning partial results',
            context={
                'open_group_key': open_group_key,
                'processed': len(evaluations),
                'total': n_sym,
            },
        )

    try:
        close_old_connections()
    except Exception:  # pragma: no cover
        logger.exception('close_old_connections after deployment_market_open')

    return {
        'status': 'success',
        'deployment_id': deployment_id,
        'open_group_key': open_group_key,
        'symbol_count': len(deployment_symbols),
        'evaluation_count': len(evaluations),
        'actionable_count': actionable_count,
        'fired_at': fire_at.isoformat(),
        'partial': len(evaluations) < len(deployment_symbols),
    }


@shared_task(bind=True, name='live_trading.update_positions')
def update_positions_task(self, deployment_id: int) -> dict:
    """Reconcile open `LiveTrade` rows for a deployment against the broker.

    Currently invoked manually (Phase 5 ships the skeleton; an interval beat
    can be registered later if/when a deployment opts into broker polling).
    """

    actor_id = self.request.id or 'manual'
    try:
        deployment = StrategyDeployment.objects.select_related('broker').get(
            id=deployment_id,
        )
    except StrategyDeployment.DoesNotExist:
        return {
            'status': 'error',
            'error': f'Deployment {deployment_id} not found',
            'deployment_id': deployment_id,
        }

    if deployment.status not in ('active', 'evaluating', 'paused'):
        return {
            'status': 'skipped',
            'reason': f"deployment status '{deployment.status}'",
            'deployment_id': deployment_id,
        }

    summary = update_open_trades(
        deployment,
        actor_type='task',
        actor_id=actor_id,
    )
    summary['deployment_id'] = deployment_id
    connections.close_all()
    return summary


@shared_task(
    bind=True,
    name='live_trading.reconcile_close_until_sync',
    soft_time_limit=100,
    time_limit=110,
)
def reconcile_close_until_sync_task(self, main_live_trade_id: int) -> dict:
    """Poll `update_open_trades` until the main leg and any hedge children are not open.

    Used after a manual (or other) market close so the UI can wait for Alpaca
    async fills and DB rows to match broker reality.
    """
    from .services.order_service import _hedge_leg_child_trades, get_adapter_for_deployment

    actor_id = self.request.id or 'reconcile_close'
    max_iterations = 70
    sleep_sec = 1.4

    for i in range(max_iterations):
        close_old_connections()
        try:
            main = LiveTrade.objects.select_related(
                'symbol', 'deployment', 'deployment__broker', 'deployment_symbol',
            ).get(id=main_live_trade_id)
        except LiveTrade.DoesNotExist:
            return {
                'status': 'error',
                'error': 'trade_not_found',
                'main_live_trade_id': main_live_trade_id,
            }

        dep = main.deployment
        adapter = get_adapter_for_deployment(dep)
        if adapter is None:
            return {
                'status': 'error',
                'error': 'no_broker_adapter',
                'main_live_trade_id': main_live_trade_id,
            }

        update_open_trades(
            dep,
            broker_adapter=adapter,
            actor_type='task',
            actor_id=actor_id,
        )

        main.refresh_from_db()
        open_hedges = [h for h in _hedge_leg_child_trades(main) if h.status == 'open']

        if main.status != 'open' and not open_hedges:
            return {
                'status': 'success',
                'main_live_trade_id': main_live_trade_id,
                'main_status': main.status,
                'iterations': i + 1,
            }

        if self.request.id:
            self.update_state(
                state='PROGRESS',
                meta={
                    'phase': 'reconcile_close',
                    'iteration': i + 1,
                    'max_iterations': max_iterations,
                    'main_status': main.status,
                    'hedge_open': len(open_hedges),
                },
            )
        time.sleep(sleep_sec)

    main.refresh_from_db()
    open_hedges = [h for h in _hedge_leg_child_trades(main) if h.status == 'open']
    return {
        'status': 'timeout',
        'main_live_trade_id': main_live_trade_id,
        'main_status': main.status,
        'hedge_open': len(open_hedges),
        'iterations': max_iterations,
    }


def _refresh_hedge_bench_ohlcv(*, open_group_key: str, actor_id: str) -> None:
    """SPY, VIXM, VIXY, ^VIX — once per ``market_open_fanout`` (before any deployment).

    Always runs (not gated on ``hedge_enabled``) so the vol-hedge dashboard and
    DB-backed model series stay fresh. Failures are logged; they never block
    individual deployment work.
    """
    from backtest_engine.services.hybrid_vix_hedge import HEDGE_TICKERS
    from market_data.tasks import fetch_ohlcv_data_task

    today = timezone.now().date().isoformat()
    bench_t0 = time.monotonic()
    for ticker in HEDGE_TICKERS:
        if (time.monotonic() - bench_t0) > HEDGE_BENCH_OHLCV_BUDGET_SEC:
            logger.warning(
                'hedge_bench_ohlcv budget exceeded open_group_key=%s after %s',
                open_group_key,
                HEDGE_BENCH_OHLCV_BUDGET_SEC,
            )
            log_event(
                None,
                event_type='task_tick',
                actor_type='task',
                actor_id=actor_id,
                level='warning',
                message=(
                    f'OHLCV hedge bench stopped early (time budget {HEDGE_BENCH_OHLCV_BUDGET_SEC}s)'
                ),
                context={
                    'open_group_key': open_group_key,
                    'scope': 'hedge_bench_ohlcv',
                    'today': today,
                },
            )
            break
        try:
            result = fetch_ohlcv_data_task.apply(
                kwargs={
                    'ticker': ticker,
                    'period': '1mo',
                    'replace_existing': False,
                    # Hedge bench tickers are treated as Yahoo-backed reference series.
                    'provider_code': 'YAHOO',
                },
            )
            value = result.result if hasattr(result, 'result') else result
            if isinstance(value, dict) and value.get('status') == 'failed':
                log_event(
                    None,
                    event_type='task_tick',
                    actor_type='task',
                    actor_id=actor_id,
                    level='warning',
                    message=(
                        f"OHLCV refresh failed (hedge bench) for {ticker}: "
                        f"{value.get('message')}"
                    ),
                    context={
                        'ticker': ticker,
                        'open_group_key': open_group_key,
                        'scope': 'hedge_bench_ohlcv',
                        'today': today,
                        'fetch_result': value,
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception('hedge_bench_ohlcv failed ticker=%s', ticker, exc)
            log_event(
                None,
                event_type='task_tick',
                actor_type='task',
                actor_id=actor_id,
                level='warning',
                message=f"OHLCV refresh raised (hedge bench) for {ticker}: {exc}",
                error=exc,
                context={
                    'ticker': ticker,
                    'open_group_key': open_group_key,
                    'scope': 'hedge_bench_ohlcv',
                    'today': today,
                },
            )
    logger.info(
        'hedge_bench_ohlcv complete open_group_key=%s tickers=%s',
        open_group_key,
        list(HEDGE_TICKERS),
    )


def _refresh_ohlcv_for_symbols(deployment, deployment_symbols, *, actor_id: str) -> None:
    """Synchronously refresh daily OHLCV for each deployment enrolled symbol.

    Hedge benchmark tickers (``SPY``, ``VIXM``, ``VIXY``, ``^VIX``) are updated
    in :func:`_refresh_hedge_bench_ohlcv` from the fanout task, before this
    per-deployment pass.

    Uses `apply()` so each refresh runs in-process; failures are captured in
    the audit log but never abort the surrounding tick.
    """
    from market_data.tasks import fetch_ohlcv_data_multiple_symbols_task, fetch_ohlcv_data_task

    # Gap-Up/Gap-Down needs enough daily history to compute RollingSTD_{std_period}.
    # Refresh more than 1mo so we don't end up with mostly-null STD and "invalid_std" holds.
    std_period = int((deployment.strategy_parameters or {}).get('std_period', 90) or 90)
    refresh_period = '6mo' if std_period >= 60 else '3mo'

    today = timezone.now().date().isoformat()
    refresh_t0 = time.monotonic()

    def _refresh_over_budget() -> bool:
        return (time.monotonic() - refresh_t0) > OHLCV_REFRESH_BUDGET_SEC

    # Group by provider so we can reuse market_data batching/rate-limits (esp. Alpaca free tier).
    by_provider: dict[str, list] = {}
    for ds in deployment_symbols:
        code = getattr(getattr(ds.symbol, 'provider', None), 'code', None) or 'YAHOO'
        by_provider.setdefault(code.upper(), []).append(ds)

    for provider_code, dss in by_provider.items():
        if _refresh_over_budget():
            log_event(
                deployment,
                event_type='task_tick',
                actor_type='task',
                actor_id=actor_id,
                level='warning',
                message=(
                    f'OHLCV refresh time budget ({OHLCV_REFRESH_BUDGET_SEC}s) hit; '
                    f'skipping remaining providers for this deployment'
                ),
                context={
                    'provider_code': provider_code,
                    'today': today,
                    'scope': 'deployment_ohlcv_refresh',
                },
            )
            break
        tickers = [ds.symbol.ticker for ds in dss]
        if (
            provider_code == 'ALPACA'
            and len(tickers) > 1
            and not _refresh_over_budget()
        ):
            try:
                result = fetch_ohlcv_data_multiple_symbols_task.apply(
                    kwargs={
                        'tickers': tickers,
                        'period': refresh_period,
                        'replace_existing': False,
                        'provider_code': provider_code,
                    }
                )
                value = result.result if hasattr(result, 'result') else result
                per = (value or {}).get('result') if isinstance(value, dict) else None
                if isinstance(per, dict):
                    for ds in dss:
                        t = ds.symbol.ticker
                        r = per.get(t) or {}
                        if isinstance(r, dict) and r.get('errors'):
                            log_event(
                                deployment,
                                deployment_symbol=ds,
                                event_type='error',
                                actor_type='task',
                                actor_id=actor_id,
                                level='warning',
                                message=f"OHLCV refresh failed for {t}: {r.get('message') or 'bulk fetch error'}",
                                context={
                                    'ticker': t,
                                    'today': today,
                                    'provider_code': provider_code,
                                    'fetch_result': r,
                                    'bulk': True,
                                },
                            )
                    continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    'OHLCV bulk refresh raised for deployment=%s provider=%s',
                    deployment.id, provider_code,
                )
                # Fall back to per-ticker refresh below.

        for ds in dss:
            if _refresh_over_budget():
                log_event(
                    deployment,
                    event_type='task_tick',
                    actor_type='task',
                    actor_id=actor_id,
                    level='warning',
                    message=(
                        f'OHLCV refresh time budget ({OHLCV_REFRESH_BUDGET_SEC}s) hit; '
                        f'skipping further symbols (last provider {provider_code})'
                    ),
                    context={
                        'provider_code': provider_code,
                        'today': today,
                        'scope': 'deployment_ohlcv_refresh',
                    },
                )
                return
            ticker = ds.symbol.ticker
            try:
                result = fetch_ohlcv_data_task.apply(
                    kwargs={
                        'ticker': ticker,
                        'period': refresh_period,
                        'replace_existing': False,
                        'provider_code': provider_code,
                    },
                )
                value = result.result if hasattr(result, 'result') else result
                if isinstance(value, dict) and value.get('status') == 'failed':
                    log_event(
                        deployment,
                        deployment_symbol=ds,
                        event_type='error',
                        actor_type='task',
                        actor_id=actor_id,
                        level='warning',
                        message=(
                            f"OHLCV refresh failed for {ticker}: "
                            f"{value.get('message')}"
                        ),
                        context={
                            'ticker': ticker,
                            'today': today,
                            'provider_code': provider_code,
                            'fetch_result': value,
                        },
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    'OHLCV refresh raised for deployment=%s ticker=%s',
                    deployment.id, ticker,
                )
                log_event(
                    deployment,
                    deployment_symbol=ds,
                    event_type='error',
                    actor_type='task',
                    actor_id=actor_id,
                    level='warning',
                    message=f"OHLCV refresh raised for {ticker}: {exc}",
                    error=exc,
                    context={'ticker': ticker, 'today': today, 'provider_code': provider_code},
                )


@shared_task(bind=True, name='live_trading.weekend_snapshot_recalc')
def weekend_snapshot_recalc_task(
    self,
    *,
    only_deployment_ids=None,
    enqueue_recalcs: bool = True,
    reconcile: bool = True,
) -> dict:
    """Weekly job that re-runs each deployment's snapshot backtests.

    Workflow:
    1. Iterate over every active/evaluating deployment (or the supplied
       `only_deployment_ids`).
    2. For each deployment, dispatch `weekend_recalc_for_deployment_task` so
       the actual recalc + reconcile runs in parallel across the worker
       pool. The fan-out task itself is intentionally light so it does not
       hold a worker for the full Saturday window.
    """

    actor_id = self.request.id or 'manual'
    started_at = timezone.now()

    qs = StrategyDeployment.objects.filter(
        status__in=('active', 'evaluating', 'paused'),
    )
    if only_deployment_ids:
        qs = qs.filter(id__in=list(only_deployment_ids))
    deployment_ids = list(qs.values_list('id', flat=True))

    log_event(
        None,
        event_type='task_tick',
        actor_type='task',
        actor_id=actor_id,
        message=(
            f"weekend_snapshot_recalc for {len(deployment_ids)} deployment(s)"
        ),
        context={
            'deployment_ids': deployment_ids,
            'enqueue_recalcs': enqueue_recalcs,
            'reconcile': reconcile,
            'started_at': started_at.isoformat(),
        },
    )

    dispatched = 0
    for deployment_id in deployment_ids:
        try:
            weekend_recalc_for_deployment_task.delay(
                deployment_id,
                enqueue_recalcs=enqueue_recalcs,
                reconcile=reconcile,
            )
            dispatched += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                'Failed to dispatch weekend recalc for deployment=%s', deployment_id,
            )
            log_event(
                None,
                event_type='error',
                actor_type='task',
                actor_id=actor_id,
                level='error',
                message=f'weekend dispatch failed for deployment {deployment_id}: {exc}',
                error=exc,
                context={'deployment_id': deployment_id},
            )

    connections.close_all()
    return {
        'status': 'success',
        'deployment_count': len(deployment_ids),
        'dispatched': dispatched,
        'started_at': started_at.isoformat(),
    }


@shared_task(bind=True, name='live_trading.weekend_recalc_for_deployment')
def weekend_recalc_for_deployment_task(
    self,
    deployment_id: int,
    *,
    enqueue_recalcs: bool = True,
    reconcile: bool = True,
) -> dict:
    """Per-deployment portion of the weekend recalc job.

    Triggers the per-symbol `backtest_engine.run_symbol_backtest_run` tasks
    (when `enqueue_recalcs=True`) and then runs the reconciliation pass which
    transitions `DeploymentSymbol.status` based on the latest snapshot
    stats (color downgrade -> flag-or-disable, upgrade -> enable).

    Note: the recalc tasks run asynchronously, so the reconciliation pass
    that follows operates on whatever snapshot stats are currently committed
    to the DB. In production we expect a chord/group hand-off, but the
    plain reconcile is safe to call repeatedly — it is idempotent.
    """

    actor_id = self.request.id or 'manual'
    try:
        deployment = StrategyDeployment.objects.select_related(
            'parameter_set', 'broker', 'strategy',
        ).get(id=deployment_id)
    except StrategyDeployment.DoesNotExist:
        return {
            'status': 'error',
            'error': f'Deployment {deployment_id} not found',
        }

    recalc_summary = queue_snapshot_recalc(
        deployment,
        actor_type='task',
        actor_id=actor_id,
        enqueue=enqueue_recalcs,
    )

    reconcile_summary = None
    if reconcile:
        reconcile_summary = reconcile_deployment_symbols(
            deployment,
            actor_type='task',
            actor_id=actor_id,
        ).to_dict()

    connections.close_all()
    return {
        'status': 'success',
        'deployment_id': deployment_id,
        'recalc': recalc_summary,
        'reconcile': reconcile_summary,
    }


@shared_task(bind=True, name='live_trading.evaluate_deployment')
def evaluate_deployment_task(
    self,
    deployment_id: int,
    *,
    transition_status: bool = True,
) -> dict:
    """Score a paper deployment against its `evaluation_criteria`.

    Persists the breakdown to `evaluation_results` and (when
    `transition_status=True`) flips the deployment to `passed`/`failed`.
    """

    actor_id = self.request.id or 'manual'
    try:
        deployment = StrategyDeployment.objects.select_related('broker').get(
            id=deployment_id,
        )
    except StrategyDeployment.DoesNotExist:
        return {
            'status': 'error',
            'error': f'Deployment {deployment_id} not found',
        }

    result = evaluate_deployment_for_promotion(
        deployment,
        actor_type='task',
        actor_id=actor_id,
        save=True,
        transition_status=transition_status,
    )

    connections.close_all()
    return {
        'status': 'success',
        'deployment_id': deployment_id,
        'evaluation': result.to_dict(),
        'new_status': deployment.status,
    }
