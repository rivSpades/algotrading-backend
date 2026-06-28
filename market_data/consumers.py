"""
WebSocket consumers for real-time task progress updates.

Progress is polled from the Celery result backend — no Redis channel layer
required (avoids blocking the ASGI event loop on group_add/receive).
"""

import asyncio
import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer


def _read_task_progress(task_id):
    """Sync Celery poll — run via sync_to_async from the consumer."""
    from algo_trading_backend.celery import app as celery_app
    from celery.result import AsyncResult

    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        if task_result.successful():
            result = task_result.result
            if isinstance(result, dict):
                return {
                    'status': result.get('status', 'completed'),
                    'progress': result.get('progress', 100),
                    'message': result.get('message', 'Task completed'),
                    'done': True,
                }
            return {
                'status': 'completed',
                'progress': 100,
                'message': 'Task completed successfully',
                'done': True,
            }
        error_info = task_result.info
        return {
            'status': 'failed',
            'progress': 0,
            'message': str(error_info) if error_info else 'Task failed',
            'done': True,
        }

    progress = 0
    message = 'Processing...'
    status = 'running'

    try:
        backend = task_result.backend
        if backend:
            meta = backend.get_task_meta(task_result.id)
            if meta:
                result = meta.get('result')
                if isinstance(result, dict):
                    progress = result.get('progress', 0)
                    message = result.get('message', 'Processing...')
                    status = result.get('status', 'running')
                elif meta.get('meta') and isinstance(meta['meta'], dict):
                    progress = meta['meta'].get('progress', 0)
                    message = meta['meta'].get('message', 'Processing...')
                    status = meta['meta'].get('status', 'running')

        if progress == 0 and message == 'Processing...':
            info = task_result.info
            if isinstance(info, dict):
                progress = info.get('progress', 0)
                message = info.get('message', 'Processing...')
                status = info.get('status', 'running')
            elif info:
                message = str(info)
    except Exception:
        pass

    return {
        'status': status,
        'progress': progress,
        'message': message,
        'done': False,
    }


class TaskProgressConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for task progress updates (Celery poll only)."""

    # Disable Redis channel layer — we don't use group_send; avoids
    # channel_layer.receive() blocking and timing out on redis:6379.
    channel_layer_alias = '__no_channel_layer__'

    async def connect(self):
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        self._poll_task = None
        await self.accept()
        self._poll_task = asyncio.create_task(self.poll_task_status())

    async def disconnect(self, close_code):
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def receive(self, text_data):
        pass

    async def poll_task_status(self):
        read_progress = sync_to_async(_read_task_progress, thread_sensitive=False)

        while True:
            try:
                payload = await read_progress(self.task_id)
                await self.send(text_data=json.dumps({
                    'status': payload['status'],
                    'progress': payload['progress'],
                    'message': payload['message'],
                }))
                if payload.get('done'):
                    break
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}',
                }))
                break
