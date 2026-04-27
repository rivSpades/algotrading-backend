"""ExchangeSchedule bootstrap + Celery Beat registration helpers.

The platform groups deployments by their snapshot symbols' exchanges so a
single Celery Beat schedule can fan out signals for every exchange that
shares the same open minute (e.g. NYSE / NASDAQ both open at 13:30 UTC).

Two responsibilities live here:

1. `ensure_default_exchange_schedules()` — seed `ExchangeSchedule` rows for a
   handful of major venues (NYSE, NASDAQ, AMEX, ARCA, BATS, LSE, XETR, TSE,
   HKEX). Idempotent: re-running it never duplicates rows.

2. `sync_market_open_periodic_tasks()` — read every active `ExchangeSchedule`,
   group them by `open_group_key` (open time + weekday mask), and ensure
   exactly one `PeriodicTask` per group calling
   `live_trading.tasks.market_open_fanout` with that key. Stale tasks owned
   by this system are removed when their group disappears.

Both helpers are pure-Python; the management command
`live_trading bootstrap_market_schedules` and tests are the entry points.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import time
from typing import Iterable, Optional

from django.db import transaction

from market_data.models import Exchange, ExchangeSchedule

logger = logging.getLogger(__name__)


MARKET_OPEN_TASK_PATH = 'live_trading.market_open_fanout'
WEEKEND_RECALC_TASK_PATH = 'live_trading.weekend_snapshot_recalc'

# Market-open evaluation is intentionally delayed a few minutes so broker
# minute bars and provider OHLCV refreshes have time to settle.
MARKET_OPEN_DELAY_MINUTES = 3

#: Marker stored on `PeriodicTask.description` so we can safely delete the
#: rows we created without touching unrelated user-managed schedules.
LIVE_BEAT_MARKER = '[live_trading-auto] '

#: User-facing name/description for the weekend beat (relabel + `ensure_weekend_recalc_task`).
WEEKEND_BEAT_NAME = 'LiveTrading — Weekend snapshot recalc (reconcile symbols)'
WEEKEND_BEAT_DESCRIPTION = (
    LIVE_BEAT_MARKER
    + 'Weekend: queue snapshot backtests and reconcile DeploymentSymbol status/color/tier. '
    + 'Pipeline: Beat -> weekend_snapshot_recalc_task -> enqueue per-symbol snapshot tasks -> reconcile deployment symbols.'
)


@dataclass(frozen=True)
class ExchangeScheduleSpec:
    """Bootstrap definition for `ensure_default_exchange_schedules`."""

    code: str
    name: str
    country: str
    timezone: str
    open_utc: time
    close_utc: time
    weekdays: str = '1,2,3,4,5'


DEFAULT_EXCHANGE_SCHEDULES: tuple[ExchangeScheduleSpec, ...] = (
    # US equities — all share the 13:30/20:00 UTC slot
    ExchangeScheduleSpec(
        code='NYSE', name='New York Stock Exchange',
        country='United States', timezone='America/New_York',
        open_utc=time(13, 30), close_utc=time(20, 0),
    ),
    ExchangeScheduleSpec(
        code='NASDAQ', name='NASDAQ',
        country='United States', timezone='America/New_York',
        open_utc=time(13, 30), close_utc=time(20, 0),
    ),
    ExchangeScheduleSpec(
        code='AMEX', name='NYSE American',
        country='United States', timezone='America/New_York',
        open_utc=time(13, 30), close_utc=time(20, 0),
    ),
    ExchangeScheduleSpec(
        code='ARCA', name='NYSE Arca',
        country='United States', timezone='America/New_York',
        open_utc=time(13, 30), close_utc=time(20, 0),
    ),
    ExchangeScheduleSpec(
        code='BATS', name='Cboe BZX',
        country='United States', timezone='America/New_York',
        open_utc=time(13, 30), close_utc=time(20, 0),
    ),
    # London Stock Exchange — 08:00/16:30 BST/UTC slot
    ExchangeScheduleSpec(
        code='LSE', name='London Stock Exchange',
        country='United Kingdom', timezone='Europe/London',
        open_utc=time(8, 0), close_utc=time(16, 30),
    ),
    # Frankfurt (XETRA) — 07:00/15:30 UTC (winter)
    ExchangeScheduleSpec(
        code='XETR', name='Xetra (Deutsche Börse)',
        country='Germany', timezone='Europe/Berlin',
        open_utc=time(7, 0), close_utc=time(15, 30),
    ),
    # Tokyo Stock Exchange — 00:00/06:00 UTC
    ExchangeScheduleSpec(
        code='TSE', name='Tokyo Stock Exchange',
        country='Japan', timezone='Asia/Tokyo',
        open_utc=time(0, 0), close_utc=time(6, 0),
    ),
    # Hong Kong Exchange — 01:30/08:00 UTC
    ExchangeScheduleSpec(
        code='HKEX', name='Hong Kong Stock Exchange',
        country='Hong Kong', timezone='Asia/Hong_Kong',
        open_utc=time(1, 30), close_utc=time(8, 0),
    ),
)


def ensure_default_exchange_schedules(
    specs: Iterable[ExchangeScheduleSpec] = DEFAULT_EXCHANGE_SCHEDULES,
) -> tuple[int, int]:
    """Seed the default `ExchangeSchedule` rows.

    Creates the underlying `Exchange` row when missing (mirroring the rest of
    the platform that lazily creates exchanges as data flows in). Returns a
    tuple `(created, updated)` for reporting.
    """

    created = 0
    updated = 0

    for spec in specs:
        with transaction.atomic():
            exchange, exchange_created = Exchange.objects.get_or_create(
                code=spec.code,
                defaults={
                    'name': spec.name,
                    'country': spec.country,
                    'timezone': spec.timezone,
                },
            )
            # Backfill missing metadata on existing rows but never overwrite
            # user-edited values that already differ.
            dirty = False
            if not exchange.name:
                exchange.name = spec.name
                dirty = True
            if not exchange.country:
                exchange.country = spec.country
                dirty = True
            if not exchange.timezone or exchange.timezone == 'UTC':
                exchange.timezone = spec.timezone
                dirty = True
            if dirty:
                exchange.save()

            schedule, schedule_created = ExchangeSchedule.objects.get_or_create(
                exchange=exchange,
                open_utc=spec.open_utc,
                close_utc=spec.close_utc,
                weekdays=spec.weekdays,
                defaults={'active': True},
            )
            if schedule_created:
                created += 1
            elif not schedule.active:
                schedule.active = True
                schedule.save(update_fields=['active', 'updated_at'])
                updated += 1
    return created, updated


def schedules_grouped_by_open(
    only_active: bool = True,
) -> dict[str, list[ExchangeSchedule]]:
    """Return active schedules indexed by `open_group_key`."""

    qs = ExchangeSchedule.objects.select_related('exchange')
    if only_active:
        qs = qs.filter(active=True)
    grouped: dict[str, list[ExchangeSchedule]] = defaultdict(list)
    for schedule in qs:
        grouped[schedule.open_group_key()].append(schedule)
    return grouped


def _crontab_for_group(group_key: str, schedules: list[ExchangeSchedule]):
    """Return a `(minute, hour, day_of_week)` crontab spec for a group key."""

    sample = schedules[0]
    minute = sample.open_utc.minute + MARKET_OPEN_DELAY_MINUTES
    hour = sample.open_utc.hour
    if minute >= 60:
        hour += minute // 60
        minute = minute % 60
    if hour >= 24:
        hour = hour % 24
    weekdays = sample.weekday_list() or [1, 2, 3, 4, 5]
    day_of_week = ','.join(str((d % 7)) for d in weekdays)  # ISO 1-7 -> cron 1-6,0
    return minute, hour, day_of_week


def build_market_open_beat_name_and_description(
    group_key: str,
    schedules: list,
) -> tuple[str, str]:
    """Return user-facing `(name, description)` for a market-open beat row.

    `schedules` is a list of `ExchangeSchedule` for that `open_group_key`, or
    empty if none are active (legacy/orphan row still in the beat table).
    """
    if schedules:
        exchanges = ', '.join(sorted({s.exchange.code for s in schedules}))
        label_tail = f'({exchanges})'
    else:
        label_tail = '(no active schedules; check ExchangeSchedule)'

    task_name = f'LiveTrading — Market open fanout [{group_key}] {label_tail}'
    ex_for_desc = (
        ', '.join(sorted({s.exchange.code for s in schedules}))
        if schedules
        else 'none match this key'
    )
    description = (
        LIVE_BEAT_MARKER
        + 'Market open: runs once per open group, then dispatches one worker task per live deployment. '
        + f'open_group_key={group_key}; exchanges={ex_for_desc}. '
        + 'Flow: beat → market_open_fanout (refresh SPY + VIXM + VIXY + ^VIX first, always) → '
        + 'dispatch deployment_market_open per deployment (refresh that deployment’s symbols) → '
        + 'evaluate symbols in priority order → place orders when actionable.'
    )
    return task_name, description


def sync_market_open_periodic_tasks(
    *,
    delete_stale: bool = True,
) -> dict:
    """Create/update one `PeriodicTask` per open-group.

    Marks each task with the `LIVE_BEAT_MARKER` so we can safely delete only
    the rows that *we* manage when a group disappears (orphan cleanup).
    """

    from django_celery_beat.models import CrontabSchedule, PeriodicTask

    groups = schedules_grouped_by_open(only_active=True)
    seen_names: set[str] = set()
    summary = {
        'created': 0,
        'updated': 0,
        'deleted': 0,
        'task_names': [],
    }

    for group_key, schedules in groups.items():
        minute, hour, day_of_week = _crontab_for_group(group_key, schedules)
        crontab, _ = CrontabSchedule.objects.get_or_create(
            minute=str(minute),
            hour=str(hour),
            day_of_week=day_of_week,
            day_of_month='*',
            month_of_year='*',
            timezone='UTC',
        )

        task_name, description = build_market_open_beat_name_and_description(group_key, schedules)
        seen_names.add(task_name)
        kwargs_payload = json.dumps({'open_group_key': group_key})

        task, was_created = PeriodicTask.objects.update_or_create(
            name=task_name,
            defaults={
                'task': MARKET_OPEN_TASK_PATH,
                'crontab': crontab,
                'interval': None,
                'enabled': True,
                'kwargs': kwargs_payload,
                'description': description,
            },
        )
        summary['task_names'].append(task_name)
        if was_created:
            summary['created'] += 1
        else:
            summary['updated'] += 1

    if delete_stale:
        stale = PeriodicTask.objects.filter(
            task=MARKET_OPEN_TASK_PATH,
            description__startswith=LIVE_BEAT_MARKER,
        ).exclude(name__in=seen_names)
        summary['deleted'] = stale.count()
        stale.delete()

    return summary


def ensure_weekend_recalc_task(
    *,
    minute: int = 0,
    hour: int = 2,
    day_of_week: str = '6',  # Saturday in cron (0=Sun, 6=Sat)
) -> tuple[bool, object]:
    """Create the weekend snapshot recalc beat (Phase 6 will populate it).

    Returns `(created, task)`.
    """

    from django_celery_beat.models import CrontabSchedule, PeriodicTask

    crontab, _ = CrontabSchedule.objects.get_or_create(
        minute=str(minute),
        hour=str(hour),
        day_of_week=day_of_week,
        day_of_month='*',
        month_of_year='*',
        timezone='UTC',
    )
    # Merge on `task` path so a legacy row named by Python path (e.g. "live_trading.…")
    # is updated in place instead of duplicating a second row keyed by the new `name`.
    existing = list(
        PeriodicTask.objects.filter(task=WEEKEND_RECALC_TASK_PATH).order_by('id')
    )
    if len(existing) > 1:
        keep_pk = existing[0].pk
        PeriodicTask.objects.filter(task=WEEKEND_RECALC_TASK_PATH).exclude(
            pk=keep_pk
        ).delete()
        task = PeriodicTask.objects.get(pk=keep_pk)
        was_created = False
    elif len(existing) == 1:
        task = existing[0]
        was_created = False
    else:
        task = None
        was_created = True

    if task is not None:
        task.name = WEEKEND_BEAT_NAME
        task.description = WEEKEND_BEAT_DESCRIPTION
        task.crontab = crontab
        task.interval = None
        task.enabled = True
        if not task.kwargs or task.kwargs in ('', '{}', '[]'):
            task.kwargs = '{}'
        task.save()
        return was_created, task

    task = PeriodicTask.objects.create(
        name=WEEKEND_BEAT_NAME,
        task=WEEKEND_RECALC_TASK_PATH,
        crontab=crontab,
        interval=None,
        enabled=True,
        kwargs='{}',
        description=WEEKEND_BEAT_DESCRIPTION,
    )
    return was_created, task


def deployments_for_open_group(open_group_key: str):
    """Return active deployments whose enrolled symbols hit `open_group_key`.

    A deployment is included if any of its `DeploymentSymbol`s point at an
    exchange that has at least one active schedule with the given open group
    key.
    """
    from django.db.models import Q

    from ..models import StrategyDeployment

    schedules = ExchangeSchedule.objects.filter(active=True).select_related('exchange')
    matching_exchange_codes = sorted(
        {s.exchange.code for s in schedules if s.open_group_key() == open_group_key}
    )
    if not matching_exchange_codes:
        return StrategyDeployment.objects.none(), []

    qs = (
        StrategyDeployment.objects.filter(
            status__in=('active', 'evaluating'),
            deployment_symbols__status='active',
            deployment_symbols__symbol__exchange__code__in=matching_exchange_codes,
        )
        .distinct()
        .select_related('strategy', 'broker', 'parameter_set')
    )
    return qs, matching_exchange_codes


def deployment_symbols_for_open_group(
    deployment,
    open_group_key: str,
    *,
    statuses: Optional[Iterable[str]] = ('active',),
):
    """Return `DeploymentSymbol`s belonging to `deployment` that match the group."""

    schedules = ExchangeSchedule.objects.filter(active=True).select_related('exchange')
    matching_exchange_codes = sorted(
        {s.exchange.code for s in schedules if s.open_group_key() == open_group_key}
    )
    qs = deployment.deployment_symbols.select_related(
        'symbol', 'symbol__exchange',
    ).filter(symbol__exchange__code__in=matching_exchange_codes)
    if statuses is not None:
        qs = qs.filter(status__in=list(statuses))
    return qs.order_by('priority', 'symbol__ticker')
