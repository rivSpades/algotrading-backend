"""Management command: bootstrap exchange schedules and Celery Beat entries.

Usage::

    python manage.py bootstrap_market_schedules
    python manage.py bootstrap_market_schedules --skip-beat
    python manage.py bootstrap_market_schedules --skip-weekend

Idempotent. Safe to re-run after editing `DEFAULT_EXCHANGE_SCHEDULES` or
adding new `ExchangeSchedule` rows manually — the command will create the
missing per-open-group `PeriodicTask` rows and clean up stale ones it owns.
"""

from django.core.management.base import BaseCommand

from market_data.services.beat_task_labels import apply_friendly_periodic_task_labels
from live_trading.services.scheduling import (
    ensure_default_exchange_schedules,
    ensure_weekend_recalc_task,
    sync_market_open_periodic_tasks,
)


class Command(BaseCommand):
    help = (
        'Bootstrap default ExchangeSchedule rows and synchronise the '
        'live-trading Celery Beat schedule (market-open + weekend recalc).'
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            '--skip-beat',
            action='store_true',
            help='Only seed ExchangeSchedule rows; do not touch django_celery_beat.',
        )
        parser.add_argument(
            '--skip-weekend',
            action='store_true',
            help='Do not register the weekend snapshot recalc beat.',
        )
        parser.add_argument(
            '--keep-stale',
            action='store_true',
            help='Keep auto-generated PeriodicTask rows even if their group is gone.',
        )

    def handle(self, *args, **options) -> None:
        created, updated = ensure_default_exchange_schedules()
        self.stdout.write(self.style.SUCCESS(
            f'ExchangeSchedule: {created} created, {updated} reactivated.'
        ))

        if options['skip_beat']:
            self.stdout.write('Skipping Celery Beat sync (--skip-beat).')
            return

        beat_summary = sync_market_open_periodic_tasks(
            delete_stale=not options['keep_stale'],
        )
        self.stdout.write(self.style.SUCCESS(
            'Market-open beats: '
            f"{beat_summary['created']} created, "
            f"{beat_summary['updated']} updated, "
            f"{beat_summary['deleted']} stale removed."
        ))
        for name in beat_summary['task_names']:
            self.stdout.write(f'  - {name}')

        if options['skip_weekend']:
            self.stdout.write('Skipping weekend recalc beat (--skip-weekend).')
            return

        was_created, weekend_task = ensure_weekend_recalc_task()
        verb = 'created' if was_created else 'updated'
        self.stdout.write(self.style.SUCCESS(
            f'Weekend recalc beat {verb}: {weekend_task.name}'
        ))

        label_stats = apply_friendly_periodic_task_labels()
        self.stdout.write(self.style.SUCCESS(
            f'Beat labels: {label_stats!r}'
        ))
