"""
Vercel Python Serverless Function for Account CrewAI Analysis
OpenAI-only version - optimized for size
Fetches account data from Supabase and runs analysis

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time by ~1-2 seconds.
"""

import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Optional, Dict, Any
from datetime import datetime, timezone

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

def summarize_transcription(text: str, llm, max_length: int = 1000) -> str:
    """
    Summarize a transcription to key points using LLM
    Returns a concise summary focusing on sentiment, concerns, action items, and key quotes
    """
    if not text or len(text) <= max_length:
        return text
    
    try:
        # Use LLM to create a focused summary
        prompt = f"""Summarize this meeting transcription focusing on:
1. Customer sentiment and tone (positive, negative, neutral, frustrated, etc.)
2. Key concerns or issues raised
3. Action items and commitments mentioned
4. Important quotes that indicate relationship health
5. Overall meeting outcome and next steps

Keep the summary to approximately {max_length} characters. Be specific and include actual quotes when relevant.

Transcription:
{text[:20000]}"""  # Limit input to first 20k chars for summarization
        
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        # Fallback if summary is still too long
        if len(summary) > max_length * 1.5:
            summary = summary[:max_length] + "..."
        
        return summary
    except Exception as e:
        print(f'Error summarizing transcription: {e}')
        # Fallback: truncate intelligently (first 30% + last 70%)
        if len(text) > max_length:
            first_part = text[:max_length // 3]
            last_part = text[-(max_length * 2 // 3):]
            return f"{first_part}\n\n... [middle section truncated] ...\n\n{last_part}"
        return text

def process_transcription_for_crew(transcription_data: dict, llm, meeting_date_str: Optional[str] = None) -> dict:
    """
    Process a transcription based on recency:
    - Recent (0-30 days): Full text up to 15k chars
    - Moderate (30-90 days): Summarized to ~1000 chars
    - Historical (90+ days): Key points only ~500 chars
    """
    text = transcription_data.get('transcription', '')
    if not text:
        return transcription_data
    
    # Calculate recency
    days_ago = None
    if meeting_date_str:
        try:
            meeting_date = datetime.fromisoformat(meeting_date_str.replace('Z', '+00:00'))
            if meeting_date.tzinfo is None:
                meeting_date = meeting_date.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_ago = (now - meeting_date).days
        except Exception as e:
            print(f'Error parsing meeting date {meeting_date_str}: {e}')
    
    # Process based on recency
    if days_ago is None or days_ago <= 30:
        # Recent: Full text but limit to 15k chars to manage context
        MAX_RECENT_LENGTH = 15000
        if len(text) > MAX_RECENT_LENGTH:
            # Keep first 20% and last 80% for very long recent calls
            first_part = text[:MAX_RECENT_LENGTH // 5]
            last_part = text[-(MAX_RECENT_LENGTH * 4 // 5):]
            processed_text = f"{first_part}\n\n... [middle section truncated for length] ...\n\n{last_part}"
        else:
            processed_text = text
    elif days_ago <= 90:
        # Moderate: Summarize to ~1000 chars
        processed_text = summarize_transcription(text, llm, max_length=1000)
    else:
        # Historical: Key points only ~500 chars
        processed_text = summarize_transcription(text, llm, max_length=500)
    
    # Return processed transcription
    result = transcription_data.copy()
    result['transcription'] = processed_text
    result['processing_note'] = (
        'full_text' if days_ago is None or days_ago <= 30 else
        'summarized' if days_ago <= 90 else
        'key_points_only'
    )
    return result

def fetch_account_data(account_id: Optional[str] = None, salesforce_account_id: Optional[str] = None, llm=None):
    """Fetch account data from Supabase with parallel data fetching for performance"""
    try:
        from supabase import create_client, Client
        from concurrent.futures import ThreadPoolExecutor, as_completed

        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set')

        supabase: Client = create_client(supabase_url, supabase_key)

        # Fetch account first (needed to get sf_account_id)
        if account_id:
            result = supabase.table('accounts').select('*').eq('id', account_id).limit(1).execute()
        elif salesforce_account_id:
            result = supabase.table('accounts').select('*').eq('salesforce_id', salesforce_account_id).limit(1).execute()
        else:
            raise ValueError('account_id or salesforce_account_id is required')

        if not result.data or len(result.data) == 0:
            raise ValueError('Account not found')

        account = result.data[0]
        sf_account_id = account.get('salesforce_id')

        # Define fetch functions for parallel execution
        def fetch_contacts():
            try:
                # Create new client for thread safety
                client = create_client(supabase_url, supabase_key)
                contacts_result = client.table('contacts').select('*').eq('salesforce_account_id', sf_account_id).limit(20).execute()
                return contacts_result.data if contacts_result.data else []
            except Exception as e:
                print(f'Error fetching contacts: {e}')
                return []

        def fetch_cases():
            try:
                client = create_client(supabase_url, supabase_key)
                cases_result = client.table('cases').select('*').eq('salesforce_account_id', sf_account_id).order('created_date', desc=True).limit(10).execute()
                return cases_result.data if cases_result.data else []
            except Exception as e:
                print(f'Error fetching cases: {e}')
                return []

        def fetch_transcriptions():
            try:
                client = create_client(supabase_url, supabase_key)
                trans_result = client.table('transcriptions').select('*').eq('salesforce_account_id', sf_account_id).order('meeting_date', desc=True).limit(10).execute()
                return trans_result.data if trans_result.data else []
            except Exception as e:
                print(f'Error fetching transcriptions: {e}')
                return []

        # Execute all fetches in parallel
        contacts = []
        cases = []
        transcriptions = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(fetch_contacts): 'contacts',
                executor.submit(fetch_cases): 'cases',
                executor.submit(fetch_transcriptions): 'transcriptions'
            }

            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    result = future.result()
                    if data_type == 'contacts':
                        contacts = result
                    elif data_type == 'cases':
                        cases = result
                    elif data_type == 'transcriptions':
                        transcriptions = result
                except Exception as e:
                    print(f'Error in parallel fetch for {data_type}: {e}')
        
        # Process transcriptions based on recency (if LLM is available)
        processed_transcriptions = []
        for t in transcriptions:
            trans_data = {
                'transcription': t.get('transcription_text', ''),
                'meeting': {
                    'subject': t.get('meeting_subject', ''),
                    'meeting_date': t.get('meeting_date', ''),
                }
            }
            
            # Process transcription if LLM is available
            if llm:
                processed = process_transcription_for_crew(
                    trans_data,
                    llm,
                    meeting_date_str=t.get('meeting_date')
                )
                processed_transcriptions.append(processed)
            else:
                # Fallback: simple truncation if no LLM
                text = trans_data['transcription']
                if len(text) > 15000:
                    text = text[:15000] + "... [truncated]"
                trans_data['transcription'] = text
                processed_transcriptions.append(trans_data)
        
        return {
            'name': account.get('name', 'Unknown'),
            'account_tier': account.get('account_tier', 'Unknown'),
            'contract_value': account.get('contract_value', 'Unknown'),
            'industry': account.get('industry', 'Unknown'),
            'contacts': contacts,
            'cases': cases,
            'transcriptions': processed_transcriptions
        }
    except Exception as e:
        print(f'Error fetching account data: {e}')
        raise

class handler(BaseHTTPRequestHandler):
    """Main handler for Account CrewAI requests - Vercel format"""
    
    def do_GET(self):
        """Handle GET requests - only for health checks on exact path"""
        path = self.path.split('?')[0]  # Remove query string
        if path != '/api/crew/account':
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
            'service': 'CrewAI Account Analysis',
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

            # Lazy load crewai classes (reduces cold start time)
            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)
            
            # Get account identifiers from body (always extract these for RAG lookup)
            account_id = body.get('accountId')
            salesforce_account_id = body.get('salesforceAccountId')
            context = body.get('context', '')
            use_rag = body.get('useRag', True)  # Default to using RAG for performance

            # Get accountData from body (if frontend sends it) or fetch it
            account_data = body.get('accountData')
            if not account_data:
                send_progress(self.wfile, 'Data Fetching', 'Fetching account data from database...', 'System')
                if account_id or salesforce_account_id:
                    # Pass LLM to fetch_account_data for summarization
                    account_data = fetch_account_data(account_id, salesforce_account_id, llm=llm)
                else:
                    raise ValueError('accountData, accountId, or salesforceAccountId is required')

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            # Fetch crew configuration from database
            crew_config = fetch_crew_config('account')

            # Try to use RAG for context if enabled and account ID is available
            rag_context = None
            avoma_context = None
            context_source = 'raw_data'  # Track where context came from

            print(f'RAG check: use_rag={use_rag}, account_id={account_id}, salesforce_account_id={salesforce_account_id}')

            if use_rag and (account_id or salesforce_account_id):
                # Step 1: Try RAG (fastest - uses pre-computed embeddings)
                try:
                    from rag_helpers import get_relevant_context, get_analysis_query

                    send_progress(self.wfile, 'RAG Search', 'Retrieving relevant context from vector database...', 'System')

                    # Get the account UUID for RAG search
                    # Check if account_id is already a UUID (contains dashes and is 36 chars)
                    is_uuid = account_id and '-' in account_id and len(account_id) == 36
                    rag_account_id = account_id if is_uuid else None

                    # If not a UUID, resolve from salesforce_id
                    if not rag_account_id:
                        from supabase import create_client
                        supabase_url = os.environ.get('SUPABASE_URL')
                        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
                        if supabase_url and supabase_key:
                            client = create_client(supabase_url, supabase_key)
                            lookup_id = salesforce_account_id or account_id
                            if lookup_id:
                                result = client.table('accounts').select('id').eq('salesforce_id', lookup_id).limit(1).execute()
                                if result.data:
                                    rag_account_id = result.data[0]['id']
                                    print(f'Resolved account UUID: {rag_account_id} from salesforce_id: {lookup_id}')
                                else:
                                    print(f'Could not resolve salesforce_id {lookup_id} to UUID')

                    print(f'RAG account_id for search: {rag_account_id}')

                    if rag_account_id:
                        # Generate analysis-specific query for embedding search
                        analysis_query = get_analysis_query('account')
                        print(f'Calling get_relevant_context with account_id={rag_account_id}')
                        rag_result = get_relevant_context(
                            account_id=rag_account_id,
                            query=analysis_query,
                            match_count=20,  # Get more chunks for comprehensive analysis
                            match_threshold=0.4  # Lower threshold for broader coverage
                        )
                        print(f'RAG result: {len(rag_result.get("chunks", []))} chunks, context length: {len(rag_result.get("context", ""))}')
                        if rag_result.get('context'):
                            rag_context = rag_result['context']
                            context_source = 'rag'
                            data_type_counts = rag_result.get('data_type_counts', {})
                            print(f'RAG retrieved context with {len(rag_result.get("chunks", []))} chunks: {data_type_counts}')
                            send_progress(self.wfile, 'RAG Success', f'Found {len(rag_result.get("chunks", []))} relevant chunks', 'System')
                        else:
                            print('RAG returned empty context - falling back to Avoma/raw data')
                            send_progress(self.wfile, 'RAG Empty', 'No embeddings found, trying Avoma...', 'System')
                    else:
                        print('No valid account UUID for RAG search')
                        send_progress(self.wfile, 'RAG Skipped', 'Could not resolve account UUID', 'System')
                except Exception as rag_error:
                    print(f'RAG context retrieval failed: {rag_error}')
                    import traceback
                    traceback.print_exc()
                    send_progress(self.wfile, 'RAG Error', f'RAG failed: {str(rag_error)[:50]}', 'System')
                    rag_context = None

                # Step 2: If RAG failed or returned no context, try Avoma API (real-time fetch)
                if not rag_context:
                    try:
                        from avoma_helpers import get_avoma_context_for_account

                        send_progress(self.wfile, 'Avoma Fetch', 'Fetching meeting data from Avoma...', 'System')

                        avoma_result = get_avoma_context_for_account(
                            salesforce_account_id=salesforce_account_id,
                            customer_name=account_data.get('name'),
                            account_id=account_id,
                            max_meetings=5,
                            max_transcript_chars=10000
                        )

                        if avoma_result.get('success') and avoma_result.get('context'):
                            avoma_context = avoma_result['context']
                            context_source = 'avoma'
                            print(f'Avoma retrieved context: {len(avoma_result.get("meetings", []))} meetings, {len(avoma_context)} chars')
                        else:
                            print(f'Avoma fetch failed or returned no data: {avoma_result.get("error", "Unknown error")}')
                    except Exception as avoma_error:
                        print(f'Avoma context retrieval failed: {avoma_error}')
                        avoma_context = None

            # Build account context - use RAG > Avoma > raw data (in order of preference)
            if rag_context:
                # Use RAG context (fastest - pre-computed embeddings, relevant chunks only)
                send_progress(self.wfile, 'RAG Context', f'Using {len(rag_context)} chars of relevant context from vector DB', 'System')
                account_context = f"""Account: {account_data.get('name', 'Unknown')}
Tier: {account_data.get('account_tier', 'Unknown')}
Value: {account_data.get('contract_value', 'Unknown')}
Industry: {account_data.get('industry', 'Unknown')}
Contacts: {len(account_data.get('contacts', []))}
Cases: {len(account_data.get('cases', []))}
Transcripts: {len(account_data.get('transcriptions', []))}

=== RELEVANT CONTEXT (from vector search) ===
{rag_context}"""

                if context:
                    account_context += f"\n\nAdditional Context: {context}"
            elif avoma_context:
                # Use Avoma context (real-time fetch from Avoma API)
                send_progress(self.wfile, 'Avoma Context', f'Using {len(avoma_context)} chars of meeting data from Avoma', 'System')
                account_context = f"""Account: {account_data.get('name', 'Unknown')}
Tier: {account_data.get('account_tier', 'Unknown')}
Value: {account_data.get('contract_value', 'Unknown')}
Industry: {account_data.get('industry', 'Unknown')}
Contacts: {len(account_data.get('contacts', []))}
Cases: {len(account_data.get('cases', []))}

=== MEETING DATA (from Avoma) ===
{avoma_context}"""

                # Add cases from Supabase since Avoma doesn't have them
                cases = account_data.get('cases', [])
                if cases:
                    account_context += f"\n\n=== SUPPORT CASES (from database) ==="
                    for case in cases[:5]:
                        case_number = case.get('case_number') or case.get('id') or 'N/A'
                        account_context += f"\n- Case #{case_number}: {case.get('subject', 'No subject')} (Status: {case.get('status', 'Unknown')}, Priority: {case.get('priority', 'Unknown')})"

                if context:
                    account_context += f"\n\nAdditional Context: {context}"
            else:
                # Fallback: Build context from raw data (traditional method)
                account_context = f"""Account: {account_data.get('name', 'Unknown')}
Tier: {account_data.get('account_tier', 'Unknown')}
Value: {account_data.get('contract_value', 'Unknown')}
Industry: {account_data.get('industry', 'Unknown')}"""

                # Add contacts details
                contacts = account_data.get('contacts', [])
                if contacts:
                    account_context += f"\n\nContacts ({len(contacts)}):"
                    for contact in contacts[:5]:  # Limit to first 5
                        account_context += f"\n- {contact.get('name', 'Unknown')} ({contact.get('email', 'No email')})"

                # Add cases details with case numbers
                cases = account_data.get('cases', [])
                if cases:
                    account_context += f"\n\nSupport Cases ({len(cases)}):"
                    for case in cases[:5]:  # Limit to first 5
                        case_number = case.get('case_number') or case.get('id') or 'N/A'
                        account_context += f"\n- Case #{case_number}: {case.get('subject', 'No subject')} (Status: {case.get('status', 'Unknown')}, Priority: {case.get('priority', 'Unknown')}, Created: {case.get('created_date', 'Unknown')})"
                        if case.get('description'):
                            desc = case.get('description', '')[:300]
                            account_context += f"\n  Description: {desc}{'...' if len(case.get('description', '')) > 300 else ''}"

                # Add transcriptions with processed text
                transcriptions = account_data.get('transcriptions', [])
                if transcriptions:
                    account_context += f"\n\nMeeting Transcripts ({len(transcriptions)}):"
                    for i, trans in enumerate(transcriptions[:5], 1):  # Limit to first 5
                        meeting = trans.get('meeting', {})
                        subject = meeting.get('subject', 'Unknown Meeting')
                        date = meeting.get('meeting_date', 'Unknown Date')
                        text = trans.get('transcription', '')
                        processing_note = trans.get('processing_note', '')

                        note_text = f" [{processing_note}]" if processing_note and processing_note != 'full_text' else ""
                        account_context += f"\n\n--- Meeting {i}: {subject} ({date}){note_text} ---\n{text}"

                if context:
                    account_context += f"\n\nAdditional Context: {context}"
            
            # Build system prompt with evaluation criteria and scoring rubric if available
            system_prompt_parts = []
            if crew_config and crew_config.get('system_prompt'):
                system_prompt_parts.append(crew_config['system_prompt'])
            if crew_config and crew_config.get('evaluation_criteria'):
                system_prompt_parts.append('\n\n' + crew_config['evaluation_criteria'])
            if crew_config and crew_config.get('scoring_rubric'):
                system_prompt_parts.append('\n\n' + crew_config['scoring_rubric'])
            
            full_system_prompt = '\n'.join(system_prompt_parts) if system_prompt_parts else None
            
            send_progress(self.wfile, 'Setup', 'Preparing analysis agents...', 'System')

            # Create agents from database config or fallback to defaults
            agent_configs = crew_config.get('agent_configs', []) if crew_config else []

            if not agent_configs:
                # Fallback to hardcoded defaults
                agent_configs = [
                    {
                        'role': 'Account Health Analyst',
                        'goal': 'Analyze account health, customer sentiment, and relationship status',
                        'backstory': 'You are an expert B2B account analyst specializing in customer relationship management.'
                    },
                    {
                        'role': 'Account Strategy Advisor',
                        'goal': 'Develop actionable strategies to improve account health and reduce churn risk',
                        'backstory': 'You are a strategic advisor with deep expertise in B2B account management.'
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
                        'description': f'''Analyze account health and sentiment using the following account data. CRITICAL: You must reference SPECIFIC data points, quotes, dates, and names from the provided information. Do NOT use generic placeholders.

{account_context}

CRITICAL REQUIREMENTS - You MUST:
1. Quote actual customer statements from meeting transcripts (include speaker names and meeting dates)
2. Reference specific case numbers, subjects, and statuses by name
3. Name actual contacts and their roles when discussing stakeholders
4. Cite specific meeting dates and subjects when discussing sentiment trends
5. Use actual account details (tier, value, industry) in your analysis

Provide:
1. Account health score (1-10) with detailed reasoning that cites specific evidence
2. Key risk factors or positive indicators with specific evidence
3. Sentiment analysis using actual quotes and meeting details
4. Relationship trajectory with specific evidence
5. Key stakeholders by name''',
                        'expected_output': 'Comprehensive account health analysis that explicitly quotes customer statements, references specific case numbers and dates, and names actual contacts. Every claim must be backed by specific evidence from the provided data.',
                        'agent': 'Account Health Analyst'
                    },
                    {
                        'description': f'''Based on the detailed analysis provided, develop actionable strategies and recommendations. CRITICAL: Your recommendations MUST directly address the specific issues, quotes, and data points identified in the analysis.

Requirements:
- Reference specific customer quotes and concerns from the analysis
- Address actual case numbers and issues mentioned
- Name specific contacts who need to be involved
- Reference specific meeting dates and subjects when proposing follow-ups
- Use actual account details (tier, value) to prioritize recommendations
- Make recommendations specific to the actual problems identified, not generic advice''',
                        'expected_output': 'Highly specific, actionable strategy recommendations that directly reference customer quotes, case numbers, contact names, and meeting dates from the analysis. Each recommendation must be tied to specific evidence.',
                        'agent': 'Account Strategy Advisor',
                        'context': ['analysis_task']
                    }
                ]
            
            tasks = []
            task_map = {}
            
            for i, task_config in enumerate(task_configs):
                task_description = task_config.get('description', '')
                
                # Replace {account_context} placeholder if present
                if '{account_context}' in task_description:
                    task_description = task_description.replace('{account_context}', account_context)
                elif account_context and i == 0:
                    task_description = f"{task_description}\n\n{account_context}"
                
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
                if 'strategy' in task_config.get('description', '').lower() or i == 1:
                    task_map['strategy_task'] = task
            
            # Send progress updates for each agent that will work
            for i, agent_config in enumerate(agent_configs):
                agent_name = agent_config.get('role', f'Agent {i+1}')
                send_progress(self.wfile, f'Step {i+1}', f'Analyzing account data...', agent_name)

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
                    account_id = body.get('accountId')
                    salesforce_account_id = body.get('salesforceAccountId')
                    
                    if user_id:
                        save_data = {
                            'user_id': user_id,
                            'crew_type': 'account',
                            'result': result_text,
                            'provider': 'openai',
                            'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                        }
                        
                        if account_id:
                            save_data['account_id'] = account_id
                        if salesforce_account_id:
                            save_data['salesforce_account_id'] = salesforce_account_id
                        
                        result_insert = supabase.table('crew_analysis_history').insert(save_data).execute()
                        if hasattr(result_insert, 'error') and result_insert.error:
                            print(f'Error saving crew analysis to database: {result_insert.error}')
                    else:
                        print(f'Warning: No userId provided in request body, skipping save')
                else:
                    print('Warning: Supabase URL or Key not set, skipping save')
            except Exception as save_error:
                # Don't fail the request if save fails - just log it
                print(f'Error saving crew analysis to database: {save_error}')
            
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
