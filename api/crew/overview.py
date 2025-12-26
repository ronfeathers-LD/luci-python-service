"""
Vercel Python Serverless Function for Overview CrewAI Analysis
OpenAI-only version - optimized for size
Fetches accounts from Supabase and runs portfolio analysis
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error

# Import shared helpers (includes warning suppression and lazy loading)
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import fetch_crew_config, save_analysis_to_database, build_system_prompt

def fetch_user_accounts(user_id: Optional[str] = None):
    """Fetch user's accounts from Supabase via user_accounts relationship"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set')
        
        if not user_id:
            raise ValueError('user_id is required to fetch user accounts')
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # First verify user exists
        user_result = supabase.table('users').select('id').eq('id', user_id).limit(1).execute()
        if not user_result.data or len(user_result.data) == 0:
            raise ValueError(f'User not found: {user_id}')
        
        # Fetch accounts via user_accounts relationship table
        # This ensures we only get accounts assigned to this specific user
        result = supabase.table('user_accounts').select(
            'account_id, accounts(id, name, account_tier, contract_value, industry, salesforce_id)'
        ).eq('user_id', user_id).execute()
        
        if not result.data:
            return []
        
        # Format accounts for the crew
        accounts = []
        for user_account in result.data:
            account = user_account.get('accounts')
            if account:
                accounts.append({
                    'name': account.get('name', 'Unknown'),
                    'account_tier': account.get('account_tier', 'Unknown'),
                    'contract_value': account.get('contract_value', 0),
                    'industry': account.get('industry', 'Unknown'),
                    'salesforce_id': account.get('salesforce_id', ''),
                })
        
        return accounts
    except Exception as e:
        print(f'Error fetching user accounts: {e}')
        raise

class handler(BaseHTTPRequestHandler):
    """Main handler for Overview CrewAI requests - Vercel format"""
    
    def do_GET(self):
        """Handle GET requests - only for health checks on exact path"""
        path = self.path.split('?')[0]  # Remove query string
        if path != '/api/crew/overview':
            # Don't write body - empty 404 might help Vercel fall through
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
            'service': 'CrewAI Overview Analysis',
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

            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            # Get accountsData from body (if frontend sends it) or fetch it
            accounts_data = body.get('accountsData', [])

            if not accounts_data:
                send_progress(self.wfile, 'Data Fetching', 'Fetching account portfolio from database...', 'System')
                # Try to fetch from userId
                user_id = body.get('userId')
                if user_id:
                    accounts_data = fetch_user_accounts(user_id)
                else:
                    raise ValueError('accountsData or userId is required')

            if not accounts_data or len(accounts_data) == 0:
                raise ValueError('No accounts found to analyze')

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            # Fetch crew configuration from database
            crew_config = fetch_crew_config('overview')
            
            # Build comprehensive accounts summary with actual account names and data
            accounts_summary = f"Analyzing {len(accounts_data)} accounts:\n\n"
            for i, account in enumerate(accounts_data[:20], 1):  # Limit to first 20 for context
                name = account.get('name', 'Unknown')
                tier = account.get('account_tier', 'Unknown')
                value = account.get('contract_value', 0)
                industry = account.get('industry', 'Unknown')
                accounts_summary += f"{i}. {name}\n"
                accounts_summary += f"   - Tier: {tier}\n"
                accounts_summary += f"   - Contract Value: {value}\n"
                accounts_summary += f"   - Industry: {industry}\n\n"
            
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
                        'role': 'Portfolio Analyst',
                        'goal': 'Analyze portfolio of accounts to identify trends and patterns',
                        'backstory': 'You are an expert portfolio analyst specializing in B2B account management.'
                    },
                    {
                        'role': 'Account Prioritization Specialist',
                        'goal': 'Prioritize accounts and recommend focus areas',
                        'backstory': 'You are an expert in account prioritization and strategic portfolio management.'
                    }
                ]

            agents = []
            agent_map = {}
            
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
                        'description': f'''Analyze portfolio of accounts. {accounts_summary}
                        
                        IMPORTANT: Use the actual account names provided above (e.g., "{accounts_data[0].get('name', 'Unknown')}" if available). Do NOT use generic placeholders like "Account A" or "Account B".
                        
                        Identify:
                        1. Overall portfolio health trends
                        2. Common risk factors
                        3. Accounts with positive momentum (use actual account names)
                        4. Accounts requiring attention (use actual account names)
                        5. Patterns and insights
                        
                        For each account mentioned, use its actual name from the list above.''',
                        'expected_output': 'Comprehensive portfolio analysis with trends and insights, using actual account names',
                        'agent': 'Portfolio Analyst'
                    },
                    {
                        'description': 'Based on the analysis, prioritize accounts and recommend focus areas. Use actual account names from the provided list - do NOT use generic placeholders like "Account A" or "Account B".',
                        'expected_output': 'Prioritized account list with focus recommendations, using actual account names',
                        'agent': 'Account Prioritization Specialist',
                        'context': ['analysis_task']
                    }
                ]
            
            tasks = []
            task_map = {}
            
            for i, task_config in enumerate(task_configs):
                task_description = task_config.get('description', '')
                
                # Replace {accounts_summary} placeholder if present
                if '{accounts_summary}' in task_description:
                    task_description = task_description.replace('{accounts_summary}', accounts_summary)
                elif accounts_summary and i == 0:
                    task_description = f"{task_description}\n\n{accounts_summary}"
                
                # Add system prompt to first task if available
                if full_system_prompt and i == 0:
                    task_description = f"{full_system_prompt}\n\n{task_description}"
                
                agent_role = task_config.get('agent', '')
                agent = agent_map.get(agent_role)
                if not agent:
                    print(f'Warning: Agent "{agent_role}" not found, using first agent')
                    agent = agents[0] if agents else None
                
                if not agent:
                    raise ValueError('No agents available for tasks')
                
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
                task_id = task_config.get('id') or f'task_{i}'
                task_map[task_id] = task
                if 'analysis' in task_config.get('description', '').lower() or i == 0:
                    task_map['analysis_task'] = task
                if 'prioritization' in task_config.get('description', '').lower() or i == 1:
                    task_map['prioritization_task'] = task

            # Send progress updates for each agent that will work
            for i, agent_config in enumerate(agent_configs):
                agent_name = agent_config.get('role', f'Agent {i+1}')
                send_progress(self.wfile, f'Step {i+1}', f'Analyzing portfolio data...', agent_name)

            crew = Crew(agents=agents, tasks=tasks, verbose=True)
            result = crew.kickoff()
            result_text = str(result)

            # Save to database using shared helper
            save_analysis_to_database(
                crew_type='overview',
                result=result_text,
                user_id=body.get('userId')
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
