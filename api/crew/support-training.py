"""
Hyphenated alias for support_training
This file exposes the same `handler` as `support_training.py` so requests
to `/api/crew/support-training` are handled identically.
"""
from support_training import handler as handler
