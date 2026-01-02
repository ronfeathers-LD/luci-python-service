"""
Hyphenated alias for support_resolution
This file exposes the same `handler` as `support_resolution.py` so requests
to `/api/crew/support-resolution` are handled identically.
"""
from support_resolution import handler as handler
