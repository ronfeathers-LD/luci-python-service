"""
Vercel Python Serverless Function for Sales Pipeline CrewAI Analysis
Analyzes opportunities and provides strategic hints for moving deals forward

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time by ~1-2 seconds.
"""

import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error

# Suppress warnings early (before heavy imports)
warnings.filterwarnings('ignore', message='.*Overriding of current TracerProvider.*')
warnings.filterwarnings('ignore', category=UserWarning, module='opentelemetry')
warnings.filterwarnings('ignore', category=SyntaxWarning, module='langchain')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Mixing V1 models and V2 models.*')
warnings.filterwarnings('ignore', category=UserWarning, module='crewai')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Lazy-loaded modules (cached after first import)
_langchain_openai = None
_crewai = None


def _get_langchain_openai():
    """Lazy load langchain_openai module"""
    global _langchain_openai
    if _langchain_openai is None:
        from langchain_openai import ChatOpenAI
        _langchain_openai = ChatOpenAI
    return _langchain_openai


def _get_crewai():
    """Lazy load crewai module"""
    global _crewai
    if _crewai is None:
        from crewai import Crew, Agent, Task
        _crewai = {'Crew': Crew, 'Agent': Agent, 'Task': Task}
    return _crewai


def get_llm(openai_api_key: Optional[str] = None):
    """Get OpenAI LLM instance (lazy loads langchain_openai)"""
    ChatOpenAI = _get_langchain_openai()
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
    """Fetch crew configuration from Supabase database"""
    try:
        from supabase import create_client, Client

        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            print('Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set')
            return None

        supabase: Client = create_client(supabase_url, supabase_key)

        result = supabase.table('crews').select('*').eq('crew_type', crew_type).eq('enabled', True).limit(1).execute()

        if not result.data or len(result.data) == 0:
            print(f'Warning: Crew config not found for crew_type: {crew_type}')
            return None

        crew = result.data[0]

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


def fetch_user_opportunities(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch user's opportunities from Supabase
    Returns opportunities sorted by amount (descending)
    """
    try:
        from supabase import create_client, Client

        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set')

        supabase: Client = create_client(supabase_url, supabase_key)

        # Get user's opportunity IDs from junction table
        user_opps = supabase.table('user_opportunities').select('opportunity_id').eq('user_id', user_id).execute()

        if not user_opps.data:
            return []

        opp_ids = [uo['opportunity_id'] for uo in user_opps.data if uo.get('opportunity_id')]

        if not opp_ids:
            return []

        # Fetch opportunities with account info
        result = supabase.table('opportunities').select('''
            id, salesforce_id, name, account_id, salesforce_account_id,
            amount, stage_name, probability, close_date, type, lead_source,
            next_step, description, is_won, is_closed, owner_id, owner_name,
            fiscal_quarter, fiscal_year, last_synced_at,
            accounts (id, name, account_tier, industry, contract_value)
        ''').in_('id', opp_ids).eq('is_closed', False).order('amount', desc=True).limit(limit).execute()

        return result.data if result.data else []

    except Exception as e:
        print(f'Error fetching opportunities: {e}')
        raise


def fetch_opportunity_context(opportunities: List[Dict], supabase_url: str, supabase_key: str) -> Dict[str, Any]:
    """
    Fetch additional context for opportunities:
    - Recent transcriptions for associated accounts
    - Related contacts
    - Support cases that might affect deals
    """
    try:
        from supabase import create_client, Client
        from concurrent.futures import ThreadPoolExecutor, as_completed

        supabase: Client = create_client(supabase_url, supabase_key)

        # Get unique account IDs
        account_ids = list(set([
            opp.get('salesforce_account_id')
            for opp in opportunities
            if opp.get('salesforce_account_id')
        ]))

        if not account_ids:
            return {'transcriptions': [], 'contacts': [], 'cases': []}

        context = {'transcriptions': [], 'contacts': [], 'cases': []}

        def fetch_transcriptions():
            try:
                client = create_client(supabase_url, supabase_key)
                result = client.table('transcriptions').select('''
                    salesforce_account_id, meeting_subject, meeting_date,
                    transcription_text
                ''').in_('salesforce_account_id', account_ids).order('meeting_date', desc=True).limit(10).execute()
                return result.data if result.data else []
            except Exception as e:
                print(f'Error fetching transcriptions: {e}')
                return []

        def fetch_contacts():
            try:
                client = create_client(supabase_url, supabase_key)
                result = client.table('contacts').select('''
                    salesforce_account_id, name, email, title, department
                ''').in_('salesforce_account_id', account_ids).limit(30).execute()
                return result.data if result.data else []
            except Exception as e:
                print(f'Error fetching contacts: {e}')
                return []

        def fetch_cases():
            try:
                client = create_client(supabase_url, supabase_key)
                result = client.table('cases').select('''
                    salesforce_account_id, case_number, subject, status,
                    priority, created_date
                ''').in_('salesforce_account_id', account_ids).order('created_date', desc=True).limit(10).execute()
                return result.data if result.data else []
            except Exception as e:
                print(f'Error fetching cases: {e}')
                return []

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fetch_transcriptions): 'transcriptions',
                executor.submit(fetch_contacts): 'contacts',
                executor.submit(fetch_cases): 'cases'
            }

            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    result = future.result()
                    context[data_type] = result
                except Exception as e:
                    print(f'Error in parallel fetch for {data_type}: {e}')

        return context

    except Exception as e:
        print(f'Error fetching opportunity context: {e}')
        return {'transcriptions': [], 'contacts': [], 'cases': []}


def calculate_days_until_close(close_date_str: Optional[str]) -> Optional[int]:
    """Calculate days until opportunity closes"""
    if not close_date_str:
        return None
    try:
        close_date = datetime.fromisoformat(close_date_str.replace('Z', '+00:00'))
        if close_date.tzinfo is None:
            close_date = close_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (close_date - now).days
    except Exception:
        return None


def format_currency(amount: Optional[float]) -> str:
    """Format amount as currency string"""
    if amount is None:
        return 'Unknown'
    if amount >= 1_000_000:
        return f'${amount / 1_000_000:.1f}M'
    if amount >= 1_000:
        return f'${amount / 1_000:.0f}K'
    return f'${amount:.0f}'


def build_pipeline_context(opportunities: List[Dict], context: Dict) -> str:
    """Build the context string for the AI analysis"""
    now = datetime.now(timezone.utc)

    # Pipeline summary
    total_amount = sum(opp.get('amount') or 0 for opp in opportunities)
    avg_probability = (
        sum(opp.get('probability') or 0 for opp in opportunities) / len(opportunities)
        if opportunities else 0
    )

    pipeline_context = f"""=== SALES PIPELINE OVERVIEW ===
Total Open Opportunities: {len(opportunities)}
Total Pipeline Value: {format_currency(total_amount)}
Average Probability: {avg_probability:.0f}%

"""

    # Group by stage
    stages = {}
    for opp in opportunities:
        stage = opp.get('stage_name') or 'Unknown'
        if stage not in stages:
            stages[stage] = {'count': 0, 'amount': 0, 'opps': []}
        stages[stage]['count'] += 1
        stages[stage]['amount'] += opp.get('amount') or 0
        stages[stage]['opps'].append(opp)

    pipeline_context += "=== PIPELINE BY STAGE ===\n"
    for stage, data in sorted(stages.items(), key=lambda x: x[1]['amount'], reverse=True):
        pipeline_context += f"- {stage}: {data['count']} deals, {format_currency(data['amount'])}\n"

    pipeline_context += "\n=== TOP OPPORTUNITIES (by value) ===\n"

    # Add individual opportunity details
    for i, opp in enumerate(opportunities[:10], 1):  # Top 10
        days_to_close = calculate_days_until_close(opp.get('close_date'))
        account_name = opp.get('accounts', {}).get('name', 'Unknown Account') if opp.get('accounts') else 'Unknown Account'
        account_tier = opp.get('accounts', {}).get('account_tier', '') if opp.get('accounts') else ''

        pipeline_context += f"""
--- Opportunity {i}: {opp.get('name', 'Unnamed')} ---
Account: {account_name} {f'({account_tier})' if account_tier else ''}
Amount: {format_currency(opp.get('amount'))}
Stage: {opp.get('stage_name', 'Unknown')}
Probability: {opp.get('probability', 0)}%
Close Date: {opp.get('close_date', 'Unknown')} ({days_to_close} days away)
Next Step: {opp.get('next_step') or 'Not defined'}
Type: {opp.get('type') or 'Not specified'}
Lead Source: {opp.get('lead_source') or 'Unknown'}
"""

    # Add related context
    transcriptions = context.get('transcriptions', [])
    if transcriptions:
        pipeline_context += "\n=== RECENT MEETING CONTEXT ===\n"
        for trans in transcriptions[:5]:
            meeting_date = trans.get('meeting_date', 'Unknown date')
            subject = trans.get('meeting_subject', 'No subject')
            text = trans.get('transcription_text', '')[:1500]  # Limit text
            pipeline_context += f"\n[{meeting_date}] {subject}\n{text}\n"

    contacts = context.get('contacts', [])
    if contacts:
        pipeline_context += "\n=== KEY CONTACTS ===\n"
        # Group contacts by account
        contacts_by_account = {}
        for contact in contacts:
            acc_id = contact.get('salesforce_account_id', 'unknown')
            if acc_id not in contacts_by_account:
                contacts_by_account[acc_id] = []
            contacts_by_account[acc_id].append(contact)

        for acc_id, acc_contacts in contacts_by_account.items():
            for c in acc_contacts[:3]:  # Top 3 per account
                title = c.get('title', 'No title')
                dept = c.get('department', '')
                pipeline_context += f"- {c.get('name', 'Unknown')}: {title} {f'({dept})' if dept else ''}\n"

    cases = context.get('cases', [])
    if cases:
        pipeline_context += "\n=== OPEN SUPPORT CASES (may affect deals) ===\n"
        for case in cases[:5]:
            pipeline_context += f"- Case #{case.get('case_number', 'N/A')}: {case.get('subject', 'No subject')} (Status: {case.get('status', 'Unknown')}, Priority: {case.get('priority', 'Unknown')})\n"

    return pipeline_context


class handler(BaseHTTPRequestHandler):
    """Main handler for Sales Pipeline CrewAI requests - Vercel format"""

    def do_GET(self):
        """Handle GET requests - only for health checks"""
        path = self.path.split('?')[0]
        if path != '/api/crew/sales_pipeline':
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
            'service': 'CrewAI Sales Pipeline Analysis',
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
        start_sse_response(self)

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')

            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                body = {}

            send_progress(self.wfile, 'Initialization', 'Initializing AI model...', 'System')

            # Lazy load crewai classes
            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            user_id = body.get('userId')
            if not user_id:
                raise ValueError('userId is required')

            send_progress(self.wfile, 'Data Fetching', 'Fetching your opportunities...', 'System')

            # Fetch opportunities
            opportunities = fetch_user_opportunities(user_id, limit=20)

            if not opportunities:
                send_result(
                    self.wfile,
                    "No open opportunities found in your pipeline. To get started:\n\n"
                    "1. Make sure your Salesforce opportunities are synced\n"
                    "2. Check that you have open (not closed) opportunities assigned to you\n"
                    "3. Refresh your opportunities from the Sales Dashboard",
                    provider='openai',
                    model=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
                )
                return

            send_progress(self.wfile, 'Context', f'Found {len(opportunities)} opportunities, gathering context...', 'System')

            # Fetch additional context
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            context = fetch_opportunity_context(opportunities, supabase_url, supabase_key)

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            # Fetch crew configuration
            crew_config = fetch_crew_config('sales_pipeline')

            # Build pipeline context
            pipeline_context = build_pipeline_context(opportunities, context)

            send_progress(self.wfile, 'Setup', 'Preparing analysis...', 'System')

            # Build system prompt
            system_prompt_parts = []
            if crew_config and crew_config.get('system_prompt'):
                system_prompt_parts.append(crew_config['system_prompt'])
            if crew_config and crew_config.get('evaluation_criteria'):
                system_prompt_parts.append('\n\n' + crew_config['evaluation_criteria'])

            full_system_prompt = '\n'.join(system_prompt_parts) if system_prompt_parts else None

            # Single optimized agent
            pipeline_analyst = Agent(
                role='Sales Pipeline Strategist',
                goal='Analyze the sales pipeline and provide actionable hints to move deals forward',
                backstory='Expert sales strategist who analyzes opportunities and provides specific, actionable recommendations based on deal stage, timing, and engagement patterns.',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Task description
            task_description = f'''Analyze the sales pipeline and provide strategic recommendations for each major opportunity.

{pipeline_context}

=== ANALYSIS REQUIREMENTS ===

For each of the TOP 5 opportunities by value, provide:

1. **Deal Health Assessment** (1-10 score)
   - Is the close date realistic based on the stage?
   - Is the probability appropriate for the current stage?
   - Are there red flags from support cases or engagement patterns?

2. **Next Best Action**
   - What specific action should the rep take TODAY?
   - Who should they contact and why?
   - What should they discuss or propose?

3. **Risk Factors**
   - What could derail this deal?
   - Are there competitors mentioned in meetings?
   - Is there stakeholder turnover?

4. **Timeline Recommendations**
   - Should the close date be pushed out or pulled in?
   - What milestones need to happen before close?

=== OUTPUT FORMAT ===

Start with a PIPELINE SUMMARY:
- Total pipeline health score (1-10)
- Top 3 deals to focus on this week
- Deals at risk that need immediate attention

Then for each top opportunity, provide a structured analysis with:
- Deal name and value
- Health score and reasoning
- Specific next actions with who/what/when
- Key risks to monitor

End with STRATEGIC RECOMMENDATIONS:
- Patterns you see across the pipeline
- Suggestions for pipeline balance
- Deals to potentially push to next quarter vs. accelerate'''

            if full_system_prompt:
                task_description = f"{full_system_prompt}\n\n{task_description}"

            analysis_task = Task(
                description=task_description,
                expected_output='Comprehensive pipeline analysis with deal health scores, specific next actions, and strategic recommendations.',
                agent=pipeline_analyst
            )

            send_progress(self.wfile, 'Analysis', 'Running AI analysis...', 'Pipeline Strategist')

            crew = Crew(agents=[pipeline_analyst], tasks=[analysis_task], verbose=False)
            result = crew.kickoff()
            result_text = str(result)

            # Save to database
            try:
                from supabase import create_client, Client

                if supabase_url and supabase_key:
                    supabase: Client = create_client(supabase_url, supabase_key)

                    save_data = {
                        'user_id': user_id,
                        'crew_type': 'sales_pipeline',
                        'result': result_text,
                        'provider': 'openai',
                        'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                    }

                    supabase.table('crew_analysis_history').insert(save_data).execute()
            except Exception as save_error:
                print(f'Error saving crew analysis to database: {save_error}')

            # Send final result
            model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
            send_result(self.wfile, result_text, provider='openai', model=model_name)

        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"

            send_error(self.wfile, error_message)
