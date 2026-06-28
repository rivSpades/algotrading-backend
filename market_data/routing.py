"""
WebSocket routing for market_data app
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Celery task IDs are UUIDs (e.g. bc76b761-4794-46f8-9ac8-61c48ad6b7bf)
    re_path(r'^ws/tasks/(?P<task_id>[\w-]+)/$', consumers.TaskProgressConsumer.as_asgi()),
]

