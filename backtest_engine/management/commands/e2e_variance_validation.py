"""
E2E: wipe backtests, create global test + portfolio + MC, validate Run 0 == Results PnL.
"""

import time

from django.core.management.base import BaseCommand
from django.db import transaction

from backtest_engine.models import (
    Backtest,
    BacktestStatistics,
    PortfolioMonteCarloPath,
    PortfolioMonteCarloSimulation,
    SymbolBacktestParameterSet,
    SymbolBacktestRun,
    Trade,
)
from backtest_engine.services.create_portfolio_from_parameter_set import create_portfolio_from_parameter_set
from backtest_engine.services.portfolio_monte_carlo import run_monte_carlo_simulation
from backtest_engine.tasks import run_symbol_backtest_run_task
from market_data.models import Symbol
from strategies.models import StrategyDefinition


class Command(BaseCommand):
    help = 'Wipe backtests, recreate global test flow, validate order-variance numbers'

    def add_arguments(self, parser):
        parser.add_argument('--yes-really', action='store_true', help='Required to wipe data')
        parser.add_argument('--mc-paths', type=int, default=20, help='Monte Carlo variant count')
        parser.add_argument('--tickers', nargs='+', default=['AAPL', 'SPY', 'TSLA'])

    def handle(self, *args, **options):
        if not options['yes_really']:
            self.stderr.write('Pass --yes-really to wipe and recreate test data.')
            return

        tickers = [t.upper() for t in options['tickers']]
        mc_paths = max(1, min(int(options['mc_paths']), 100))

        strategy = StrategyDefinition.objects.first()
        if not strategy:
            self.stderr.write('No strategy found')
            return

        self.stdout.write('=== Wiping all backtest data ===')
        with transaction.atomic():
            PortfolioMonteCarloPath.objects.all().delete()
            PortfolioMonteCarloSimulation.objects.all().delete()
            Trade.objects.all().delete()
            BacktestStatistics.objects.all().delete()
            Backtest.objects.all().delete()
            SymbolBacktestRun.objects.all().delete()
            SymbolBacktestParameterSet.objects.all().delete()

        self.stdout.write(self.style.SUCCESS('Wiped.'))

        from django.utils import timezone

        fixed_start = timezone.now().replace(year=1900, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        fixed_end = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        run_body = {
            'name': 'E2E global test',
            'start_date': fixed_start,
            'end_date': fixed_end,
            'split_ratio': 0.7,
            'initial_capital': 10000.0,
            'bet_size_percentage': 100.0,
            'position_modes': ['long'],
            'hedge_enabled': False,
        }

        self.stdout.write(f'=== Single-symbol runs: {tickers} ===')
        from backtest_engine.parameter_sets import build_symbol_run_parameter_payload, signature_for_payload

        payload = build_symbol_run_parameter_payload(
            strategy_id=strategy.id,
            broker_id=None,
            start_date=fixed_start,
            end_date=fixed_end,
            split_ratio=run_body['split_ratio'],
            initial_capital=run_body['initial_capital'],
            bet_size_percentage=run_body['bet_size_percentage'],
            strategy_parameters={},
            position_modes=['long'],
            hedge_enabled=False,
            run_strategy_only_baseline=True,
            hedge_config={},
        )
        sig = signature_for_payload(payload)
        ps, _ = SymbolBacktestParameterSet.objects.get_or_create(
            signature=sig,
            defaults={
                'strategy': strategy,
                'parameters': payload,
                'label': run_body['name'],
            },
        )

        for ticker in tickers:
            sym = Symbol.objects.filter(ticker=ticker, status='active').first()
            if not sym:
                self.stderr.write(f'Symbol {ticker} not found')
                return

            run = SymbolBacktestRun.objects.create(
                name=f"{run_body['name']} — {ticker}",
                strategy=strategy,
                symbol=sym,
                parameter_set=ps,
                start_date=fixed_start,
                end_date=fixed_end,
                split_ratio=run_body['split_ratio'],
                initial_capital=run_body['initial_capital'],
                bet_size_percentage=run_body['bet_size_percentage'],
                strategy_parameters=strategy.default_parameters or {},
                position_modes=['long'],
                hedge_enabled=False,
                status='pending',
            )
            self.stdout.write(f'  Running {ticker} (run id={run.id})...')
            result = run_symbol_backtest_run_task.apply(args=[run.id])
            if result.failed():
                self.stderr.write(self.style.ERROR(f'  {ticker} failed: {result.result}'))
                return
            run.refresh_from_db()
            self.stdout.write(f'  {ticker}: status={run.status}')

        self.stdout.write('=== Portfolio backtest ===')
        backtest, task_id = create_portfolio_from_parameter_set(
            strategy,
            ps,
            name='E2E portfolio',
            num_monte_carlo_paths=mc_paths,
        )
        self.stdout.write(f'  Backtest id={backtest.id}, waiting for Celery task {task_id}...')
        for _ in range(180):
            backtest.refresh_from_db()
            if backtest.status in ('completed', 'failed'):
                break
            time.sleep(2)
        if backtest.status != 'completed':
            self.stderr.write(self.style.ERROR(f'Portfolio ended with status={backtest.status}'))
            if backtest.error_message:
                self.stderr.write(backtest.error_message)
            return
        self.stdout.write(f'  Portfolio status={backtest.status}')

        stats = BacktestStatistics.objects.filter(backtest=backtest, symbol__isnull=True).first()
        results_pnl = float(stats.total_pnl) if stats else None
        trade_sum = float(sum(Trade.objects.filter(backtest=backtest).values_list('pnl', flat=True) or [0]))
        self.stdout.write(f'  Results total_pnl={results_pnl}')
        self.stdout.write(f'  Trade sum pnl={trade_sum}')
        self.stdout.write(f'  Trade count={Trade.objects.filter(backtest=backtest).count()}')

        sim = PortfolioMonteCarloSimulation.objects.filter(backtest=backtest).order_by('-id').first()
        if not sim:
            self.stderr.write('No MC simulation row')
            return

        self.stdout.write(f'=== Order variance ({mc_paths} variants) ===')
        run_monte_carlo_simulation(sim.id)
        sim.refresh_from_db()

        ref_path = PortfolioMonteCarloPath.objects.filter(simulation=sim, is_reference=True).first()
        run0 = PortfolioMonteCarloPath.objects.filter(simulation=sim, path_index=0).first()
        variants = PortfolioMonteCarloPath.objects.filter(simulation=sim, is_reference=False)

        self.stdout.write(f'  reference_profit (sim)={sim.reference_profit}')
        self.stdout.write(f'  mean_profit (all runs)={sim.mean_profit}')
        self.stdout.write(f'  Run 0 profit={run0.profit if run0 else None} is_reference={run0.is_reference if run0 else None}')
        self.stdout.write(f'  Variant count={variants.count()}')

        ok_run0 = run0 and abs(float(run0.profit) - float(results_pnl)) < 0.02
        ok_ref = sim.reference_profit is not None and abs(float(sim.reference_profit) - float(results_pnl)) < 0.02
        ok_trades = abs(trade_sum - float(results_pnl)) < 0.02

        self.stdout.write('')
        self.stdout.write('=== Validation ===')
        self.stdout.write(f'  Run 0 == Results PnL: {"PASS" if ok_run0 else "FAIL"}')
        self.stdout.write(f'  reference_profit == Results PnL: {"PASS" if ok_ref else "FAIL"}')
        self.stdout.write(f'  Trade sum == Results PnL: {"PASS" if ok_trades else "FAIL"}')

        if run0 or variants.exists():
            all_profits = []
            if run0:
                all_profits.append(float(run0.profit))
            all_profits.extend(float(v.profit) for v in variants)
            import statistics
            calc_mean = statistics.mean(all_profits)
            self.stdout.write(f'  Mean all-runs profit (recalc)={calc_mean:.2f} vs stored={sim.mean_profit}')

        if ok_run0 and ok_ref and ok_trades:
            self.stdout.write(self.style.SUCCESS('ALL CHECKS PASSED'))
        else:
            self.stderr.write(self.style.ERROR('SOME CHECKS FAILED'))

        self.stdout.write(f'\nBacktest id={backtest.id} | Parameter set={ps.signature[:12]}...')
