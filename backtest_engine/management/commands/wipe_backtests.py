"""
Dangerous: wipe all portfolio backtests from the database.

This deletes EVERY Backtest row (and cascades to related trades/statistics/etc).
It is intended for development reset while refactoring models.
"""

from django.core.management.base import BaseCommand, CommandError

from backtest_engine.models import Backtest


class Command(BaseCommand):
    help = "Delete ALL Backtest rows (portfolio + legacy snapshot runs). Requires --yes-really."

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes-really",
            action="store_true",
            help="Actually delete all Backtest rows (destructive).",
        )

    def handle(self, *args, **options):
        total = Backtest.objects.count()
        self.stdout.write(f"Backtest rows: {total}")

        if total == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to delete."))
            return

        if not options.get("yes_really"):
            raise CommandError(
                "Refusing to wipe backtests without --yes-really. "
                "Re-run with: python manage.py wipe_backtests --yes-really"
            )

        deleted_total, _by_model = Backtest.objects.all().delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {total} Backtest row(s). Total deleted objects (incl cascades): {deleted_total}"
            )
        )

