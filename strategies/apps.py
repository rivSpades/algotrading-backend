from django.apps import AppConfig


class StrategiesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'strategies'

    def ready(self):
        """Initialize strategy definitions on app startup"""
        from . import lifecycle  # noqa: F401 — post_migrate hook
        from . import signals  # noqa: F401 — register signal handlers package

