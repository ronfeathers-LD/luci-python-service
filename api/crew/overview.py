"""
Vercel Python Serverless Function for Overview CrewAI Analysis
OpenAI-only version - optimized for size
Fetches accounts from Supabase and runs portfolio analysis
"""

import json
import os
import warnings
from http.server import BaseHTTPRequestHandler
from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict, Any

# Suppress OpenTelemetry TracerProvider warnings
# This happens in serverless environments when tracing is initialized multiple times
warnings.filterwarnings('ignore', message='.*Overriding of current TracerProvider.*')
warnings.filterwarnings('ignore', category=UserWarning, module='opentelemetry')

# Suppress third-party library warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, module='langchain')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Mixing V1 models and V2 models.*')
warnings.filterwarnings('ignore', category=UserWarning, module='crewai')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

def get_llm(openai_api_key: Optional[str] = None):
    """Get OpenAI LLM instance"""
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is required')
    os.environ['OPENAI_API_KEY'] = api_key
    return ChatOpenAI(
        api_key=api_key,
        model=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
        temperature=0.7
    )

def fetch_crew_config(crew_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch crew configuration from Supabase database
    
    Args:
        crew_type: The crew_type identifier (e.g., 'account', 'implementation', 'overview')
    
    Returns:
        Dictionary with crew configuration or None if not found
    """
    try:
        from supabase import create_client, Client
        
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            print('Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set, cannot fetch crew config')
            return None
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch crew config from database
        result = supabase.table('crews').select('*').eq('crew_type', crew_type).eq('enabled', True).limit(1).execute()
        
        if not result.data or len(result.data) == 0:
            print(f'Warning: Crew config not found for crew_type: {crew_type}')
            return None
        
        crew = result.data[0]
        
        # Parse JSONB fields
        config = {
            'name': crew.get('name'),
            'description': crew.get('description'),
            'system_prompt': crew.get('system_prompt'),
            'evaluation_criteria': crew.get('evaluation_criteria'),
            'scoring_rubric': crew.get('scoring_rubric'),
            'output_schema': crew.get('output_schema'),
            'agent_configs': crew.get('agent_configs') or [],
            'task_configs': crew.get('task_configs') or [],
        }
        
        return config
        
    except Exception as e:
        print(f'Error fetching crew config from database: {e}')
        return None

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
        """Handle POST requests"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')
            
            # Parse body
            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                body = {}
            
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)
            
            # Get accountsData from body (if frontend sends it) or fetch it
            accounts_data = body.get('accountsData', [])
            
            if not accounts_data:
                # Try to fetch from userId
                user_id = body.get('userId')
                if user_id:
                    accounts_data = fetch_user_accounts(user_id)
                else:
                    raise ValueError('accountsData or userId is required')
            
            if not accounts_data or len(accounts_data) == 0:
                raise ValueError('No accounts found to analyze')
            
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
            
            # Build system prompt with evaluation criteria and scoring rubric if available
            system_prompt_parts = []
            if crew_config and crew_config.get('system_prompt'):
                system_prompt_parts.append(crew_config['system_prompt'])
            if crew_config and crew_config.get('evaluation_criteria'):
                system_prompt_parts.append('\n\n' + crew_config['evaluation_criteria'])
            if crew_config and crew_config.get('scoring_rubric'):
                system_prompt_parts.append('\n\n' + crew_config['scoring_rubric'])
            
            full_system_prompt = '\n'.join(system_prompt_parts) if system_prompt_parts else None
            
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
            
            crew = Crew(agents=agents, tasks=tasks, verbose=True)
                    result = crew.kickoff()
                    result_text = str(result)
                    
                    # Save to database
                    try:
                        from supabase import create_client, Client
                        
                        supabase_url = os.environ.get('SUPABASE_URL')
                        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
                        
                        if supabase_url and supabase_key:
                            supabase: Client = create_client(supabase_url, supabase_key)
                            
                            # Get user_id from request body
                            user_id = body.get('userId')
                            
                            if user_id:
                                save_data = {
                                    'user_id': user_id,
                                    'crew_type': 'overview',
                                    'result': result_text,
                                    'provider': 'openai',
                                    'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                                }
                                
                                supabase.table('crew_analysis_history').insert(save_data).execute()
                    except Exception as save_error:
                        # Don't fail the request if save fails - just log it
                        print(f'Error saving crew analysis to database: {save_error}')
                    
                    # Send response
                    response_data = {
                        'success': True,
                        'result': result_text,
                        'crewType': 'overview',
                        'provider': 'openai'
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data, default=str).encode('utf-8'))
            
        except Exception as e:
            import traceback
            error_data = {
                'error': 'Failed to execute overview crew',
                'message': str(e),
                'traceback': traceback.format_exc() if os.environ.get('DEBUG') else None
            }
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_data).encode('utf-8'))
