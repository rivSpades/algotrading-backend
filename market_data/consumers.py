"""
WebSocket consumers for real-time task progress updates
"""

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from celery.result import AsyncResult
import asyncio


class TaskProgressConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for task progress updates"""

    async def connect(self):
        self.task_id = self.scope['url_route']['kwargs']['task_id']
        self.room_group_name = f'task_{self.task_id}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # Start polling task status
        asyncio.create_task(self.poll_task_status())

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        """Handle messages from WebSocket"""
        pass

    async def task_progress(self, event):
        """Send task progress to WebSocket"""
        await self.send(text_data=json.dumps(event))

    async def poll_task_status(self):
        """Poll Celery task status and send updates"""
        from algo_trading_backend.celery import app as celery_app
        
        while True:
            try:
                task_result = AsyncResult(self.task_id, app=celery_app)
                
                if task_result.ready():
                    # Task completed
                    if task_result.successful():
                        result = task_result.result
                        if isinstance(result, dict):
                            await self.send(text_data=json.dumps({
                                'status': result.get('status', 'completed'),
                                'progress': result.get('progress', 100),
                                'message': result.get('message', 'Task completed'),
                            }))
                        else:
                            await self.send(text_data=json.dumps({
                                'status': 'completed',
                                'progress': 100,
                                'message': 'Task completed successfully',
                            }))
                    else:
                        error_info = task_result.info
                        await self.send(text_data=json.dumps({
                            'status': 'failed',
                            'progress': 0,
                            'message': str(error_info) if error_info else 'Task failed',
                        }))
                    break
                else:
                    # Task still running - get state metadata
                    # Celery stores progress in backend metadata when update_state is called
                    progress = 0
                    message = 'Processing...'
                    status = 'running'
                    
                    try:
                        # First try to get from backend metadata (most reliable)
                        backend = task_result.backend
                        if backend:
                            meta = backend.get_task_meta(task_result.id)
                            if meta:
                                # Metadata is stored in meta['result'] when update_state is called
                                result = meta.get('result')
                                if isinstance(result, dict):
                                    progress = result.get('progress', 0)
                                    message = result.get('message', 'Processing...')
                                    status = result.get('status', 'running')
                                # Also check meta['meta'] for some backends
                                elif meta.get('meta') and isinstance(meta['meta'], dict):
                                    progress = meta['meta'].get('progress', 0)
                                    message = meta['meta'].get('message', 'Processing...')
                                    status = meta['meta'].get('status', 'running')
                        
                        # Fallback to info property
                        if progress == 0 and message == 'Processing...':
                            info = task_result.info
                            if isinstance(info, dict):
                                progress = info.get('progress', 0)
                                message = info.get('message', 'Processing...')
                                status = info.get('status', 'running')
                            elif info:
                                message = str(info)
                    except Exception as e:
                        # If all else fails, use defaults
                        pass
                    
                    await self.send(text_data=json.dumps({
                        'status': status,
                        'progress': progress,
                        'message': message,
                    }))
                
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}',
                }))
                break

