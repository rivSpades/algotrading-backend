"""Rewrite django-celery-beat `PeriodicTask` display names to user-friendly text."""

from django.core.management.base import BaseCommand

from market_data.services.beat_task_labels import apply_friendly_periodic_task_labels


class Command(BaseCommand):
    help = (
        'Set human-readable names and descriptions on scheduled (Beat) tasks: '
        'market-open, weekend, symbol imports, and celery.backend_cleanup.'
    )

    def handle(self, *args, **options):
        stats = apply_friendly_periodic_task_labels()
        self.stdout.write(self.style.SUCCESS(f'Refresh complete: {stats!r}'))
