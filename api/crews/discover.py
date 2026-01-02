"""
Discovery endpoint for available CrewAI crews.
Returns a JSON list of available crews discovered from `api/crew/` files.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Only allow exact discovery path
        path = self.path.split('?')[0]
        if path != '/api/crews/discover':
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            return

        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            crew_dir = os.path.join(repo_root, 'crew')

            crews = []
            if os.path.isdir(crew_dir):
                for fn in sorted(os.listdir(crew_dir)):
                    if not fn.endswith('.py'):
                        continue
                    if fn.startswith('__'):
                        continue
                    name = fn[:-3]
                    crews.append({
                        'id': name,
                        'route': f'/api/crew/{name}',
                        'file': f'api/crew/{fn}'
                    })

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'crews': crews}).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
