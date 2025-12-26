"""
Vercel Python Serverless Function for Implementation CrewAI Analysis
OpenAI-only version - optimized for size
Fetches project data from Supabase and runs analysis
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error

# Import shared helpers (includes warning suppression and lazy loading)
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import fetch_crew_config, save_analysis_to_database, build_system_prompt

def fetch_project_data(salesforce_project_id: Optional[str] = None):
    """Fetch project data and transcriptions from Supabase"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set')
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch project
        if not salesforce_project_id:
            raise ValueError('salesforceProjectId is required')
        
        result = supabase.table('implementation_projects').select('*').eq('salesforce_project_id', salesforce_project_id).limit(1).execute()
        
        if not result.data or len(result.data) == 0:
            raise ValueError('Project not found')
        
        project = result.data[0]
        
        # Get salesforce account ID for transcriptions
        salesforce_account_id = project.get('salesforce_account_id') or project.get('account_id')
        
        # Fetch transcriptions for this account
        transcriptions = []
        if salesforce_account_id:
            try:
                trans_result = supabase.table('transcriptions').select('*').eq('salesforce_account_id', salesforce_account_id).order('meeting_date', desc=True).limit(20).execute()
                if trans_result.data:
                    transcriptions = [{
                        'transcription': t.get('transcription_text', ''),
                        'meeting': {
                            'subject': t.get('meeting_subject', ''),
                            'meeting_date': t.get('meeting_date', ''),
                        }
                    } for t in trans_result.data]
            except Exception as e:
                print(f'Error fetching transcriptions: {e}')
        
        return {
            'name': project.get('name', 'Unknown'),
            'status': project.get('status', 'Unknown'),
            'account_name': project.get('account_name', 'Unknown'),
            'transcriptions': transcriptions
        }
    except Exception as e:
        print(f'Error fetching project data: {e}')
        raise

class handler(BaseHTTPRequestHandler):
    """Main handler for Implementation CrewAI requests - Vercel format"""
    
    def do_GET(self):
        """Handle GET requests - only for health checks on exact path"""
        # Only handle GET for the exact crew implementation path
        path = self.path.split('?')[0]  # Remove query string
        if path != '/api/crew/implementation':
            # For non-crew paths, don't respond - let Vercel route to Next.js
            # However, BaseHTTPRequestHandler requires a response
            # So we'll send a minimal response that might allow fallthrough
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            # Don't write body - empty response might help
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'service': 'CrewAI Implementation Analysis',
            'provider': 'openai'
        }).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - SSE streaming"""
        # Start SSE response immediately
        start_sse_response(self)

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')

            # Parse body
            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                body = {}

            send_progress(self.wfile, 'Initialization', 'Initializing AI model...', 'System')

            # Get projectData from body (if frontend sends it) or fetch it
            project_data = body.get('projectData')
            transcriptions = body.get('transcriptions', [])

            if not project_data:
                send_progress(self.wfile, 'Data Fetching', 'Fetching project data from database...', 'System')
                # Try to fetch from salesforceProjectId
                salesforce_project_id = body.get('salesforceProjectId')
                if salesforce_project_id:
                    fetched_data = fetch_project_data(salesforce_project_id)
                    project_data = {
                        'name': fetched_data.get('name', 'Unknown'),
                        'status': fetched_data.get('status', 'Unknown'),
                        'account_name': fetched_data.get('account_name', 'Unknown'),
                    }
                    if not transcriptions:
                        transcriptions = fetched_data.get('transcriptions', [])
                else:
                    raise ValueError('projectData or salesforceProjectId is required')

            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            # Fetch crew configuration from database
            crew_config = fetch_crew_config('implementation')
            
            # Build comprehensive project context
            project_context = f"""Project: {project_data.get('name', 'Unknown')}
Status: {project_data.get('status', 'Unknown')}
Account: {project_data.get('account_name', 'Unknown')}
Transcripts: {len(transcriptions)}"""
            
            # Add transcript summaries if available
            if transcriptions:
                transcript_summaries = []
                for i, trans in enumerate(transcriptions[:5], 1):  # Limit to first 5
                    meeting = trans.get('meeting', {})
                    transcript_text = trans.get('transcription', '')
                    if transcript_text:
                        # Truncate long transcripts
                        preview = transcript_text[:500] + '...' if len(transcript_text) > 500 else transcript_text
                        transcript_summaries.append(f"Meeting {i}: {meeting.get('subject', 'No subject')} ({meeting.get('meeting_date', 'Unknown date')}) - {preview}")
                
                if transcript_summaries:
                    project_context += f"\n\nRecent Meeting Transcripts:\n" + "\n".join(transcript_summaries)
            
            # Build system prompt using shared helper
            full_system_prompt = build_system_prompt(crew_config)

            send_progress(self.wfile, 'Setup', 'Preparing analysis agents...', 'System')

            # Get lazy-loaded crewai classes
            crewai = get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Create agents from database config or fallback to defaults
            agent_configs = crew_config.get('agent_configs', []) if crew_config else []

            if not agent_configs:
                # Fallback to hardcoded defaults
                agent_configs = [
                    {
                        'role': 'Implementation PM Analyst',
                        'goal': 'Analyze project management effectiveness and communication patterns',
                        'backstory': 'You are an expert in implementation project management and customer communication.'
                    },
                    {
                        'role': 'Implementation Coach',
                        'goal': 'Provide specific, actionable coaching recommendations',
                        'backstory': 'You are an expert implementation coach specializing in customer communication and project management.'
                    }
                ]

            agents = []
            agent_map = {}  # Map role names to Agent objects for task context
            
            for agent_config in agent_configs:
                agent = Agent(
                    role=agent_config.get('role', 'Unknown'),
                    goal=agent_config.get('goal', ''),
                    backstory=agent_config.get('backstory', ''),
                    llm=llm,
                    verbose=True
                )
                agents.append(agent)
                agent_map[agent_config.get('role')] = agent
            
            # Create tasks from database config or fallback to defaults
            task_configs = crew_config.get('task_configs', []) if crew_config else []
            
            if not task_configs:
                # Fallback to hardcoded defaults
                task_configs = [
                    {
                        'description': f'''Analyze project management and communication. {project_context}
                        
                        Provide:
                        1. PM effectiveness assessment
                        2. Communication pattern analysis
                        3. Key strengths and areas for improvement
                        4. Risk factors
                        5. Timeline and milestone assessment''',
                        'expected_output': 'Comprehensive PM and communication analysis',
                        'agent': 'Implementation PM Analyst'
                    },
                    {
                        'description': 'Based on the analysis, provide specific coaching recommendations with examples.',
                        'expected_output': 'Specific, actionable coaching recommendations with examples',
                        'agent': 'Implementation Coach',
                        'context': ['pm_analysis_task']
                    }
                ]
            
            tasks = []
            task_map = {}  # Map task identifiers to Task objects for context references
            
            for i, task_config in enumerate(task_configs):
                # Build task description with project context and system prompt
                task_description = task_config.get('description', '')
                
                # Replace {project_context} placeholder if present
                if '{project_context}' in task_description:
                    task_description = task_description.replace('{project_context}', project_context)
                elif project_context and i == 0:  # Add context to first task if not already included
                    task_description = f"{task_description}\n\n{project_context}"
                
                # Add system prompt to first task if available
                if full_system_prompt and i == 0:
                    task_description = f"{full_system_prompt}\n\n{task_description}"
                
                # Get agent by role name
                agent_role = task_config.get('agent', '')
                agent = agent_map.get(agent_role)
                if not agent:
                    print(f'Warning: Agent "{agent_role}" not found, using first agent')
                    agent = agents[0] if agents else None
                
                if not agent:
                    raise ValueError('No agents available for tasks')
                
                # Build task context (references to other tasks)
                task_context = []
                context_refs = task_config.get('context', [])
                for ref in context_refs:
                    if ref in task_map:
                        task_context.append(task_map[ref])
                
                task = Task(
                    description=task_description,
                    expected_output=task_config.get('expected_output', ''),
                    agent=agent,
                    context=task_context if task_context else None
                )
                
                tasks.append(task)
                # Store task by identifier (use index or description hash)
                task_id = task_config.get('id') or f'task_{i}'
                task_map[task_id] = task
                # Also store by common names for backward compatibility
                if 'pm_analysis' in task_config.get('description', '').lower() or i == 0:
                    task_map['pm_analysis_task'] = task
                if 'coaching' in task_config.get('description', '').lower() or i == 1:
                    task_map['coaching_task'] = task

            # Send progress updates for each agent that will work
            for i, agent_config in enumerate(agent_configs):
                agent_name = agent_config.get('role', f'Agent {i+1}')
                send_progress(self.wfile, f'Step {i+1}', f'Analyzing project data...', agent_name)

            crew = Crew(agents=agents, tasks=tasks, verbose=True)
            result = crew.kickoff()
            result_text = str(result)

            # Save to database using shared helper
            save_analysis_to_database(
                crew_type='implementation',
                result=result_text,
                user_id=body.get('userId'),
                salesforce_project_id=body.get('salesforceProjectId')
            )

            # Send final result via SSE
            model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
            send_result(self.wfile, result_text, provider='openai', model=model_name)

        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"

            # Send error via SSE
            send_error(self.wfile, error_message)
