# Generated manually for parent-first portfolio + Monte Carlo

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0019_parameter_set_label'),
    ]

    operations = [
        migrations.AddField(
            model_name='backtest',
            name='parameter_set',
            field=models.ForeignKey(
                blank=True,
                help_text='Parent global test (parameter set) when portfolio was created from single-symbol snapshots',
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='portfolio_backtests',
                to='backtest_engine.symbolbacktestparameterset',
            ),
        ),
        migrations.AddField(
            model_name='backtest',
            name='symbol_priority_order',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Symbol tickers in priority order for same-timestamp portfolio execution (lower index runs first)',
            ),
        ),
        migrations.CreateModel(
            name='PortfolioMonteCarloSimulation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('num_paths', models.PositiveIntegerField(default=500)),
                ('status', models.CharField(
                    choices=[('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed')],
                    default='pending',
                    max_length=20,
                )),
                ('error_message', models.TextField(blank=True)),
                ('reference_symbol_order', models.JSONField(blank=True, default=list)),
                ('prob_broke', models.FloatField(blank=True, help_text='Fraction of paths with equity <= 0', null=True)),
                ('mean_profit', models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True)),
                ('median_profit', models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True)),
                ('percentile_5', models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True)),
                ('percentile_95', models.DecimalField(blank=True, decimal_places=2, max_digits=20, null=True)),
                ('profit_histogram', models.JSONField(blank=True, default=list)),
                ('sample_paths', models.JSONField(
                    blank=True,
                    default=list,
                    help_text='Up to 10 sample paths for sparklines: [{path_index, symbol_order, profit, blew_up}]',
                )),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('completed_at', models.DateTimeField(null=True, blank=True)),
                ('backtest', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='monte_carlo_simulations',
                    to='backtest_engine.backtest',
                )),
            ],
            options={
                'verbose_name': 'Portfolio Monte Carlo simulation',
                'verbose_name_plural': 'Portfolio Monte Carlo simulations',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='PortfolioMonteCarloPath',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path_index', models.PositiveIntegerField()),
                ('symbol_order', models.JSONField(blank=True, default=list)),
                ('final_equity', models.DecimalField(decimal_places=2, max_digits=20)),
                ('profit', models.DecimalField(decimal_places=2, max_digits=20)),
                ('blew_up', models.BooleanField(default=False)),
                ('simulation', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='paths',
                    to='backtest_engine.portfoliomontecarlosimulation',
                )),
            ],
            options={
                'verbose_name': 'Portfolio Monte Carlo path',
                'verbose_name_plural': 'Portfolio Monte Carlo paths',
                'ordering': ['path_index'],
                'unique_together': {('simulation', 'path_index')},
            },
        ),
        migrations.AddIndex(
            model_name='portfoliomontecarlosimulation',
            index=models.Index(fields=['backtest', '-created_at'], name='backtest_en_backtes_6a8f2d_idx'),
        ),
        migrations.AddIndex(
            model_name='portfoliomontecarlosimulation',
            index=models.Index(fields=['status'], name='backtest_en_status_8c4e1a_idx'),
        ),
        migrations.AddIndex(
            model_name='portfoliomontecarlopath',
            index=models.Index(fields=['simulation', 'path_index'], name='backtest_en_simulat_3b7c9f_idx'),
        ),
    ]
