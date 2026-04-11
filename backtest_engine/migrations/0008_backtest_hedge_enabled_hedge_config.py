# Generated manually for hybrid VIX hedge feature

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("backtest_engine", "0007_backteststatistics_intra_trade_drawdown"),
    ]

    operations = [
        migrations.AddField(
            model_name="backtest",
            name="hedge_enabled",
            field=models.BooleanField(
                default=False,
                help_text="If true, run hybrid VIX hedge simulation alongside the strategy for comparison",
            ),
        ),
        migrations.AddField(
            model_name="backtest",
            name="hedge_config",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Hybrid VIX hedge parameters (z_threshold, vix_floor, weights, windows); defaults apply when empty",
            ),
        ),
    ]
