# Replaces CharField position_mode with JSONField position_modes (multi-select long/short).

from django.db import migrations, models


def _default_position_modes():
    return ['long', 'short']


def copy_position_mode_to_list(apps, schema_editor):
    Backtest = apps.get_model('backtest_engine', 'Backtest')
    for b in Backtest.objects.all():
        pm = getattr(b, 'position_mode', None) or 'all'
        if pm == 'long':
            modes = ['long']
        elif pm == 'short':
            modes = ['short']
        else:
            modes = ['long', 'short']
        b.position_modes = modes
        b.save(update_fields=['position_modes'])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('backtest_engine', '0010_backtest_position_mode'),
    ]

    operations = [
        migrations.AddField(
            model_name='backtest',
            name='position_modes',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.RunPython(copy_position_mode_to_list, noop_reverse),
        migrations.RemoveField(
            model_name='backtest',
            name='position_mode',
        ),
        migrations.AlterField(
            model_name='backtest',
            name='position_modes',
            field=models.JSONField(
                default=_default_position_modes,
                help_text="Which directions to simulate: include 'long' and/or 'short' (at least one)",
            ),
        ),
    ]
