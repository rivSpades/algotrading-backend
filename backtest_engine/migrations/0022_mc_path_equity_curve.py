from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0021_monte_carlo_viz_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='portfoliomontecarlopath',
            name='equity_curve',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Subsampled equity curve [{timestamp, equity}]',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlopath',
            name='is_reference',
            field=models.BooleanField(
                default=False,
                help_text='True for path 0 — the saved portfolio backtest (Results tab), not re-simulated',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='reference_profit',
            field=models.DecimalField(
                blank=True,
                decimal_places=2,
                help_text='Profit from the saved portfolio backtest (matches Results tab)',
                max_digits=20,
                null=True,
            ),
        ),
    ]
