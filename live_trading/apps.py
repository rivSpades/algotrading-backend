from django.apps import AppConfig


class LiveTradingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'live_trading'

    def ready(self) -> None:
        # Import the engines package so each engine module's
        # `@register_live_engine` decorator runs and the registry is populated
        # before any task / view tries to look an engine up.
        from . import engines  # noqa: F401
        from .engines import gap_up_gap_down  # noqa: F401
