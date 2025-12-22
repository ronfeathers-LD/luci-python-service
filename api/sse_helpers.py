"""
SSE (Server-Sent Events) helper utilities
Provides utilities for sending SSE formatted messages
"""

import json
from typing import Any, Dict, Optional


def send_sse_message(wfile, data: Dict[str, Any]) -> None:
    """
    Send a single SSE message

    Args:
        wfile: The output file stream to write to
        data: Dictionary to send as JSON in the SSE data field
    """
    try:
        message = f"data: {json.dumps(data)}\n\n"
        wfile.write(message.encode('utf-8'))
        wfile.flush()
    except Exception as e:
        print(f'Error sending SSE message: {e}')


def send_progress(wfile, step: str, message: str, agent: str = '') -> None:
    """
    Send a progress update SSE message

    Args:
        wfile: The output file stream to write to
        step: The step name/identifier
        message: Progress message to display
        agent: The agent name performing this step
    """
    data = {
        'type': 'progress',
        'step': step,
        'message': message,
        'agent': agent
    }
    send_sse_message(wfile, data)


def send_result(wfile, result: str, provider: str = 'openai', model: str = 'gpt-4o-mini') -> None:
    """
    Send a final result SSE message

    Args:
        wfile: The output file stream to write to
        result: The analysis result text
        provider: AI provider name
        model: Model name used
    """
    data = {
        'type': 'result',
        'result': {
            'result': result,
            'provider': provider,
            'model': model
        }
    }
    send_sse_message(wfile, data)


def send_error(wfile, message: str) -> None:
    """
    Send an error SSE message

    Args:
        wfile: The output file stream to write to
        message: Error message
    """
    data = {
        'type': 'error',
        'message': message
    }
    send_sse_message(wfile, data)


def start_sse_response(handler) -> None:
    """
    Initialize SSE response headers

    Args:
        handler: The HTTP request handler
    """
    handler.send_response(200)
    handler.send_header('Content-Type', 'text/event-stream')
    handler.send_header('Cache-Control', 'no-cache')
    handler.send_header('Connection', 'keep-alive')
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.end_headers()
