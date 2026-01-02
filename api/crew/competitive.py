"""
Vercel Python Serverless Function for Competitive Crew
Ports LUCI's Next.js route behavior into the Python service.
Handles SSE streaming, runs CrewAI analysis, and saves results to `competitive_history`.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error, send_sse_message

# Shared helpers
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import get_supabase_client, fetch_crew_config


def save_competitive_result(supabase, user_id: str, companies: List[Dict[str, Any]], analysis: Any, analysis_type: str) -> Dict[str, Any]:
    """Save competitive analysis to `competitive_history` table similar to LUCI JS route."""
    try:
        # Create a brief summary
        exec_summary = None
        if isinstance(analysis, dict):
            exec_summary = analysis.get('executive_summary')

        if exec_summary:
            summary = exec_summary[:200] + ('...' if len(exec_summary) > 200 else '')
        else:
            summary = f'Competitive analysis of {len(companies)} companies'

        # Minify companies
        minified = []
        for c in companies:
            minified.append({
                'id': c.get('id'),
                'name': (c.get('properties') or {}).get('name') or c.get('name') or 'Unknown',
                'domain': (c.get('properties') or {}).get('domain') or c.get('domain')
            })

        insert_data = {
            'user_id': user_id,
            'companies': minified,
            'analysis': analysis,
            'analysis_type': analysis_type,
            'summary': summary,
            'analyzed_at': __import__('datetime').datetime.utcnow().isoformat()
        }

        result = supabase.table('competitive_history').insert(insert_data).execute()
        if hasattr(result, 'error') and result.error:
            return {'saved': False, 'error': str(result.error)}

        return {'saved': True, 'analysis': result.data[0] if getattr(result, 'data', None) else None}

    except Exception as e:
        return {'saved': False, 'error': str(e)}


class handler(BaseHTTPRequestHandler):
    """Handler for Competitive Crew - implements streaming analysis and saving."""

    def do_GET(self):
        path = self.path.split('?')[0]
        if path != '/api/crew/competitive':
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'service': 'CrewAI Competitive Analysis'
        }).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        # Start SSE response immediately
        start_sse_response(self)

        try:
            # Read body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')
            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                send_error(self.wfile, 'Invalid JSON in request body')
                return

            user_id = body.get('userId')
            companies = body.get('companies', [])
            analysis_type = body.get('analysisType')
            force_refresh = body.get('forceRefresh', False)

            # Validate
            if not user_id:
                send_error(self.wfile, 'Missing required parameter: userId')
                return

            if not companies or not isinstance(companies, list):
                send_error(self.wfile, 'Missing required parameter: companies (non-empty array)')
                return

            if len(companies) > 10:
                send_error(self.wfile, 'Too many companies: maximum 10 companies per analysis')
                return

            analysis_type_to_use = analysis_type or ('single' if len(companies) == 1 else 'comparative')

            send_progress(self.wfile, 'Initialization', 'Initializing AI model...', 'System')

            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            crew_config = fetch_crew_config('competitive')
            full_system_prompt = None
            if crew_config:
                # build_system_prompt isn't exported here; build simple system prompt
                full_system_prompt = crew_config.get('system_prompt')

            # Prepare CrewAI
            crewai = get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Create a single agent optimized for speed
            analyst = Agent(
                role='Competitive Analyst',
                goal='Analyze provided company data and produce comparative competitive intelligence',
                backstory='Expert competitive intelligence analyst. Provide structured JSON output with executive summary and key differentiators.',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Build companies summary for prompt
            companies_summary = ''
            for i, c in enumerate(companies, 1):
                name = (c.get('properties') or {}).get('name') or c.get('name') or 'Unknown'
                domain = (c.get('properties') or {}).get('domain') or c.get('domain') or ''
                companies_summary += f"{i}. {name} ({domain})\n"

            # Build task description
            task_description = f"""{full_system_prompt or ''}

Analyze the following companies and produce:
1) An executive_summary (short)
2) Key differentiators per company
3) Strategic opportunities and risks
4) A suggested go-to-market positioning

COMPANIES:\n{companies_summary}

Return a JSON object with keys: executive_summary, companies (array with id,name,analysis), analysis_type
"""

            task = Task(
                description=task_description,
                expected_output='JSON object with competitive analysis',
                agent=analyst
            )

            # Send a quick progress update
            send_progress(self.wfile, 'Setup', f'Running competitive analysis for {len(companies)} companies', 'System')

            crew = Crew(agents=[analyst], tasks=[task], verbose=False)
            result = crew.kickoff()
            result_text = str(result)

            # Try to parse JSON analysis from result_text
            parsed = None
            try:
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    parsed = json.loads(result_text[start_idx:end_idx])
            except Exception:
                parsed = None

            # Save to competitive_history if we have parsed analysis
            supabase = get_supabase_client()
            save_result = {'saved': False}
            if supabase and parsed:
                try:
                    save_result = save_competitive_result(supabase, user_id, companies, parsed, analysis_type_to_use)
                except Exception as e:
                    save_result = {'saved': False, 'error': str(e)}

            # Send final result via SSE
            model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
            # Compose result payload similar to other crews
            final_payload = {
                'analysis': parsed or result_text,
                'companies': companies,
                'analysisType': analysis_type_to_use,
                'saved': bool(save_result.get('saved')),
                'saveError': save_result.get('error')
            }

            send_result(self.wfile, json.dumps(final_payload), provider='openai', model=model_name)

        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"
            send_error(self.wfile, error_message)
