from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from backtest_engine.models import SymbolBacktestRun


class Command(BaseCommand):
    help = (
        "Delete orphan SymbolBacktestRun rows (parameter_set IS NULL). "
        "Trades/statistics will cascade via FK(run)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print counts but do not delete anything.",
        )
        parser.add_argument(
            "--older-than-minutes",
            type=int,
            default=0,
            help=(
                "Only delete orphans whose created_at is older than N minutes. "
                "Use this to avoid deleting runs being created right now."
            ),
        )

    def handle(self, *args, **options):
        dry_run = bool(options.get("dry_run"))
        older_than_minutes = int(options.get("older_than_minutes") or 0)

        qs = SymbolBacktestRun.objects.filter(parameter_set__isnull=True)
        if older_than_minutes > 0:
            cutoff = timezone.now() - timedelta(minutes=older_than_minutes)
            qs = qs.filter(created_at__lt=cutoff)

        total = qs.count()
        # Do a lightweight breakdown manually (avoid DB-specific group-by complexity for a maintenance command).
        status_counts = {}
        for s in qs.values_list("status", flat=True):
            status_counts[s] = status_counts.get(s, 0) + 1

        self.stdout.write(self.style.WARNING(f"Found {total} orphan SymbolBacktestRun rows (parameter_set IS NULL)."))
        if older_than_minutes > 0:
            self.stdout.write(f"Filter: created_at < now - {older_than_minutes} minutes")
        if status_counts:
            self.stdout.write("By status: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))

        if dry_run:
            self.stdout.write(self.style.SUCCESS("Dry-run: no deletions performed."))
            return

        if total == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to delete."))
            return

        with transaction.atomic():
            deleted_total, by_model = qs.delete()

        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted_total} total objects."))
        for model_label, count in sorted(by_model.items()):
            self.stdout.write(f"- {model_label}: {count}")
