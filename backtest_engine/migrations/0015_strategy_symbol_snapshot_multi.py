# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0014_strategy_symbol_snapshot'),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name='strategysymbolbacktestsnapshot',
            name='uniq_strategy_symbol_snapshot',
        ),
        migrations.AddField(
            model_name='strategysymbolbacktestsnapshot',
            name='label',
            field=models.CharField(blank=True, max_length=200, help_text='Optional user-facing label for this run'),
        ),
        migrations.AddIndex(
            model_name='strategysymbolbacktestsnapshot',
            index=models.Index(fields=['strategy', 'symbol', '-updated_at'], name='bt_ss_snap_str_sym_upd_idx'),
        ),
    ]
