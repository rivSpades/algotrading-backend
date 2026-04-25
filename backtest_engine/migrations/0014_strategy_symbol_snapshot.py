# Generated manually

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('market_data', '0005_add_provider_s3_fields'),
        ('strategies', '0002_strategydefinition_required_tool_configs'),
        ('backtest_engine', '0013_run_strategy_only_baseline'),
    ]

    operations = [
        migrations.AddField(
            model_name='backtest',
            name='is_strategy_symbol_snapshot',
            field=models.BooleanField(
                default=False,
                help_text='If true, this run is the stored per-(strategy,symbol) snapshot; hidden from default backtest lists.',
            ),
        ),
        migrations.CreateModel(
            name='StrategySymbolBacktestSnapshot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('updated_at', models.DateTimeField(auto_now=True)),
                (
                    'backtest',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='strategy_symbol_snapshot_row',
                        to='backtest_engine.backtest',
                    ),
                ),
                (
                    'strategy',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='symbol_backtest_snapshots',
                        to='strategies.strategydefinition',
                    ),
                ),
                (
                    'symbol',
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='strategy_symbol_snapshots',
                        to='market_data.symbol',
                    ),
                ),
            ],
            options={
                'verbose_name': 'Strategy symbol backtest snapshot',
                'verbose_name_plural': 'Strategy symbol backtest snapshots',
            },
        ),
        migrations.AddConstraint(
            model_name='strategysymbolbacktestsnapshot',
            constraint=models.UniqueConstraint(fields=('strategy', 'symbol'), name='uniq_strategy_symbol_snapshot'),
        ),
    ]
