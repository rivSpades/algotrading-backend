# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0006_backtest_broker'),
    ]

    operations = [
        migrations.AlterField(
            model_name='backteststatistics',
            name='max_drawdown',
            field=models.DecimalField(
                blank=True,
                decimal_places=4,
                help_text='Peak-to-trough drawdown on the equity curve (percentage of capital)',
                max_digits=10,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='backteststatistics',
            name='avg_intra_trade_drawdown',
            field=models.DecimalField(
                blank=True,
                decimal_places=4,
                help_text='Average intra-trade adverse excursion (percentage), from OHLCV during each open position',
                max_digits=10,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='backteststatistics',
            name='worst_intra_trade_drawdown',
            field=models.DecimalField(
                blank=True,
                decimal_places=4,
                help_text='Largest intra-trade adverse excursion (percentage) among closed trades',
                max_digits=10,
                null=True,
            ),
        ),
    ]
