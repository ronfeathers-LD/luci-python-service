"""
Hyphenated alias for sales_pipeline
This file exposes the same `handler` as `sales_pipeline.py` so requests
to `/api/crew/sales-pipeline` are handled identically.
"""
from sales_pipeline import handler as handler
