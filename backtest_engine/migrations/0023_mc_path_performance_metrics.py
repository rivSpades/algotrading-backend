from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0022_mc_path_equity_curve'),
    ]

    operations = [
        migrations.AddField(
            model_name='portfoliomontecarlopath',
            name='performance_metrics',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Portfolio performance metrics for this path (same keys as Results tab)',
            ),
        ),
        migrations.AddField(
            model_name='portfoliomontecarlosimulation',
            name='mean_performance_metrics',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Mean of performance_metrics across variant paths (excludes Run 0)',
            ),
        ),
    ]
