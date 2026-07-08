# Generated manually for MC auto-chain + variance visualization fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0020_portfolio_parameter_set_monte_carlo'),
    ]

    operations = [
        migrations.AddField(
            model_name='backtest',
            name='monte_carlo_num_paths',
            field=models.PositiveIntegerField(
                default=500,
                help_text='Order-variance Monte Carlo paths to run after portfolio completes (0 = skip)',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='prob_profit_positive',
            field=models.FloatField(
                blank=True,
                help_text='Fraction of paths with profit > 0',
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='reference_equity_curve',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Portfolio backtest reference equity curve [{timestamp, equity}]',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='confidence_bands',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Per-timestamp bands [{timestamp, p5, p25, p50, p75, p95}]',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='sample_equity_curves',
            field=models.JSONField(
                blank=True,
                default=list,
                help_text='Up to 20 sample path equity curves [{path_index, points:[{timestamp, equity}]}]',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='best_path',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='worst_path',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
