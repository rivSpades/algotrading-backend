from django.apps import AppConfig


class StrategiesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'strategies'

    def ready(self):
        """Initialize strategy definitions on app startup"""
        from . import signals  # noqa

