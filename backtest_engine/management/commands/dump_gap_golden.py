"""
Dump Gap golden QA fixtures (synthetic OHLCV + long/short expected results).

For the **backend agent** after intentional backtest-engine / Gap changes — not for
the QA agent to "fix" a failing golden. Review the trade/stats/equity JSON diff
before committing.

Usage (backend agent):
  python manage.py dump_gap_golden
"""

from django.core.management.base import BaseCommand

from backtest_engine.tests.golden_harness import dump_golden_files


class Command(BaseCommand):
    help = 'Regenerate Gap golden QA fixtures (OHLCV + long/short result JSON).'

    def handle(self, *args, **options):
        ohlcv_path, long_path, short_path = dump_golden_files()
        self.stdout.write(self.style.SUCCESS(f'Wrote {ohlcv_path}'))
        self.stdout.write(self.style.SUCCESS(f'Wrote {long_path}'))
        self.stdout.write(self.style.SUCCESS(f'Wrote {short_path}'))
        # Print brief trade counts for review
        import json

        for label, path in (('long', long_path), ('short', short_path)):
            data = json.loads(path.read_text())
            n = len(data.get('trades') or [])
            pnl = (data.get('stats') or {}).get('scalars', {}).get('total_pnl')
            self.stdout.write(f'  {label}: {n} trade(s), total_pnl={pnl}')
