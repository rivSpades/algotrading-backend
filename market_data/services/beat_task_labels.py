"""User-friendly `PeriodicTask.name` / `description` for the scheduler UI.

Merges duplicate rows where we used to key `update_or_create` by `name` only,
and rewrites known built-in / market-data / Celery tasks to short labels and
helpful one-line + pipeline text.

Safe to re-run. Does not change crontab/interval on weekend relabel; use
`ensure_weekend_recalc_task` in bootstrap to apply the default schedule.
"""

from __future__ import annotations

import json
from typing import Any

from django.db import transaction

from live_trading.services.scheduling import (
    MARKET_OPEN_TASK_PATH,
    WEEKEND_BEAT_NAME,
    WEEKEND_BEAT_DESCRIPTION,
    WEEKEND_RECALC_TASK_PATH,
    build_market_open_beat_name_and_description,
    schedules_grouped_by_open,
)

#: Shown in UI for import schedules created from the app (optional marker).
MARKET_DATA_BEAT_MARKER = '[market_data-auto] '

_FETCH_ALL = 'market_data.tasks.fetch_symbols_from_all_exchanges_task'
_FETCH_ONE = 'market_data.tasks.fetch_symbols_from_exchange_task'
_FETCH_MULTI = 'market_data.tasks.fetch_symbols_from_multiple_exchanges_task'
_FETCH_PATHS = {_FETCH_ALL, _FETCH_ONE, _FETCH_MULTI}

_CELERY_BACKEND_CLEANUP = 'celery.backend_cleanup'


def _kwargs_dict(pt) -> dict[str, Any]:
    k = pt.kwargs
    if k is None or k == '':
        return {}
    if isinstance(k, str):
        try:
            return json.loads(k) if k and k not in ('[]',) else {}
        except Exception:  # noqa: BLE001
            return {}
    if isinstance(k, dict):
        return k
    return {}


def _schedule_suffix(pt) -> str:
    """Uniqueness hint when several beats share the same import task (different cadence)."""
    if pt.interval_id and pt.interval is not None:
        inv = pt.interval
        return f"every {inv.every} {inv.period}"
    if pt.crontab_id and pt.crontab is not None:
        c = pt.crontab
        tz = getattr(c, 'timezone', None)
        tzs = str(tz) if tz is not None else 'UTC'
        return f'cron m={c.minute} h={c.hour} dow={c.day_of_week} {tzs}'
    return f'id {pt.pk}'


def _fetch_beat_name_and_description(pt) -> tuple[str, str] | None:
    path = (pt.task or '').strip()
    if path not in _FETCH_PATHS:
        return None
    kw = _kwargs_dict(pt)
    suffix = _schedule_suffix(pt)
    row = f' · row {pt.pk}'
    if path == _FETCH_ALL:
        name = f'Market data — Import symbols: all configured exchanges · {suffix}{row}'
        desc = (
            MARKET_DATA_BEAT_MARKER
            + 'Downloads tradable tickers for every exchange that has a schedule in the import settings. '
            + 'Run automatically or use “Run now” for a one-off import.'
        )
        return name, desc
    if path == _FETCH_ONE:
        code = (kw.get('exchange_code') or '?').upper()
        name = f'Market data — Import symbols: {code} · {suffix}{row}'
        desc = (
            MARKET_DATA_BEAT_MARKER
            + f'Pulls/updates the symbol list for {code} from the data provider. '
            + f'Schedule: {suffix}.'
        )
        return name, desc
    if path == _FETCH_MULTI:
        codes = kw.get('exchange_codes') or []
        if isinstance(codes, str):
            codes = [codes]
        elif not isinstance(codes, (list, tuple)):
            codes = []
        part = ', '.join(str(c).upper() for c in codes) if codes else 'multiple'
        name = f'Market data — Import symbols: {part} · {suffix}{row}'
        desc = (
            MARKET_DATA_BEAT_MARKER
            + f"Imports tradable tickers for the given venues ({part}). Schedule: {suffix}."
        )
        return name, desc
    return None


def _celery_cleanup_name_and_description() -> tuple[str, str]:
    name = 'Celery — Delete old task result entries (result backend cleanup)'
    desc = (
        'Celery built-in: removes expired result keys from the configured result backend. '
        'Does not change strategies or your Beat schedule.'
    )
    return name, desc


@transaction.atomic
def apply_friendly_periodic_task_labels() -> dict[str, int | str]:
    """Update names/descriptions, merge known duplicates, return simple counters.

    * Market-open: one row per `open_group_key` in kwargs; rewrites label from DB schedules.
    * Weekend: merge rows that share the weekend task path; set friendly name/description.
    * Symbol-import beats: set friendly name + marker description (includes schedule in name).
    * `celery.backend_cleanup`: human-readable label.
    """
    from django_celery_beat.models import PeriodicTask

    stats: dict[str, int | str] = {
        'market_open_merged': 0,
        'market_open_relabeled': 0,
        'weekend_merged': 0,
        'weekend_relabeled': 0,
        'fetch_relabeled': 0,
        'celery_cleanup_relabeled': 0,
    }

    groups = schedules_grouped_by_open(only_active=True)

    # —— Market open (by open_group_key) ——
    by_key: dict[str, list] = {}
    for pt in PeriodicTask.objects.filter(task=MARKET_OPEN_TASK_PATH).order_by('id'):
        kw = _kwargs_dict(pt)
        gk = kw.get('open_group_key')
        if not gk or not isinstance(gk, str):
            continue
        by_key.setdefault(gk, []).append(pt)

    for gk, rows in by_key.items():
        rows = sorted(rows, key=lambda p: p.id)
        keep = rows[0]
        schedules = list(groups.get(gk) or [])
        new_name, new_desc = build_market_open_beat_name_and_description(gk, schedules)
        for extra in rows[1:]:
            extra.delete()
            stats['market_open_merged'] = int(stats['market_open_merged']) + 1
        if keep.name != new_name or (keep.description or '') != new_desc:
            keep.name = new_name
            keep.description = new_desc
            keep.save()
            stats['market_open_relabeled'] = int(stats['market_open_relabeled']) + 1

    # —— Weekend (task path) ——
    wk = list(PeriodicTask.objects.filter(task=WEEKEND_RECALC_TASK_PATH).order_by('id'))
    if wk:
        keep = wk[0]
        for extra in wk[1:]:
            extra.delete()
            stats['weekend_merged'] = int(stats['weekend_merged']) + 1
        if keep.name != WEEKEND_BEAT_NAME or (keep.description or '') != WEEKEND_BEAT_DESCRIPTION:
            keep.name = WEEKEND_BEAT_NAME
            keep.description = WEEKEND_BEAT_DESCRIPTION
            keep.save()
            stats['weekend_relabeled'] = int(stats['weekend_relabeled']) + 1

    # —— Symbol import tasks ——
    for pt in (
        PeriodicTask.objects.filter(task__in=_FETCH_PATHS)
        .select_related('interval', 'crontab')
        .order_by('id')
    ):
        pair = _fetch_beat_name_and_description(pt)
        if not pair:
            continue
        new_name, new_desc = pair
        if pt.name != new_name or (pt.description or '') != new_desc:
            pt.name = new_name
            pt.description = new_desc
            pt.save()
            stats['fetch_relabeled'] = int(stats['fetch_relabeled']) + 1

    # —— Celery backend cleanup ——
    n0, d0 = _celery_cleanup_name_and_description()
    for pt in PeriodicTask.objects.filter(task=_CELERY_BACKEND_CLEANUP):
        if pt.name != n0 or (pt.description or '') != d0:
            pt.name = n0
            pt.description = d0
            pt.save()
            stats['celery_cleanup_relabeled'] = int(stats['celery_cleanup_relabeled']) + 1

    return stats


def relabel_fetch_periodic_task_by_id(periodic_task_id: int) -> bool:
    """Apply import-beat naming to a row right after `PeriodicTask` create (adds `row id`)."""
    from django_celery_beat.models import PeriodicTask

    pt = PeriodicTask.objects.select_related('interval', 'crontab').filter(
        pk=periodic_task_id
    ).first()
    if not pt:
        return False
    pair = _fetch_beat_name_and_description(pt)
    if not pair:
        return False
    n, d = pair
    if pt.name == n and (pt.description or '') == d:
        return False
    pt.name, pt.description = n, d
    pt.save()
    return True
