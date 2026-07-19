"""Celery tasks for analytical tools.

The async indicator computation task was removed (no Celery beat or `.delay()`
caller). Indicator computation runs synchronously via
`analytical_tools.services.compute_indicator_sync` (invoked by the views).
"""
