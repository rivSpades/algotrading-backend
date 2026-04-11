# Generated manually for persisted hedge lab defaults

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("backtest_engine", "0008_backtest_hedge_enabled_hedge_config"),
    ]

    operations = [
        migrations.CreateModel(
            name="HedgeLabSettings",
            fields=[
                (
                    "singleton_key",
                    models.CharField(
                        default="default",
                        editable=False,
                        help_text="Single row key; always use HedgeLabSettings.SINGLETON_KEY",
                        max_length=32,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "hedge_config",
                    models.JSONField(
                        blank=True,
                        default=dict,
                        help_text="Partial or full hedge parameter overrides (z_threshold, vix_floor, weights, windows)",
                    ),
                ),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Hedge lab settings",
                "verbose_name_plural": "Hedge lab settings",
            },
        ),
    ]
