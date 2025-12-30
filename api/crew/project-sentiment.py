"""
Vercel Python Serverless Function for Project Sentiment Analysis Crew
Analyzes implementation project sentiment from meeting transcripts

Provides:
- PM effectiveness scoring and coaching
- Customer sentiment analysis
- Timeline and deliverables extraction
- Risk identification

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time.
"""

import json
import os
import sys
import hashlib
from http.server import BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error, send_sse_message

# Import shared helpers
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import get_supabase_client


def fetch_project_and_transcriptions(
    salesforce_project_id: str,
    salesforce_account_id: str,
    transcription_ids: Optional[List[str]] = None,
    days_back: int = 60,
    max_transcriptions: int = 10
) -> Dict[str, Any]:
    """
    Fetch project details and transcriptions from Supabase.

    Args:
        salesforce_project_id: The Salesforce project ID
        salesforce_account_id: The Salesforce account ID
        transcription_ids: Optional list of specific transcription IDs to fetch
        days_back: Number of days to look back for recent transcriptions
        max_transcriptions: Maximum number of transcriptions to return

    Returns:
        Dict with project details and transcriptions
    """
    supabase = get_supabase_client()
    if not supabase:
        raise ValueError('Supabase client not available')

    # Fetch project
    project_result = supabase.table('implementation_projects').select('*').eq(
        'salesforce_project_id', salesforce_project_id
    ).limit(1).execute()

    if not project_result.data:
        raise ValueError(f'Project not found: {salesforce_project_id}')

    project = project_result.data[0]

    # Fetch transcriptions
    transcriptions = []

    if transcription_ids and len(transcription_ids) > 0:
        # Fetch specific transcriptions
        trans_result = supabase.table('transcriptions').select(
            'id, transcription_text, meeting_subject, meeting_date, avoma_meeting_id'
        ).in_('id', transcription_ids).execute()

        if trans_result.data:
            transcriptions = trans_result.data
    else:
        # Fetch recent transcriptions for the account
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        trans_result = supabase.table('transcriptions').select(
            'id, transcription_text, meeting_subject, meeting_date, avoma_meeting_id'
        ).eq(
            'salesforce_account_id', salesforce_account_id
        ).gte(
            'meeting_date', cutoff_date
        ).order(
            'meeting_date', desc=True
        ).limit(max_transcriptions).execute()

        if trans_result.data:
            transcriptions = trans_result.data

    return {
        'project': project,
        'transcriptions': transcriptions
    }


def compute_input_hash(
    salesforce_project_id: str,
    transcription_ids: List[str],
    transcription_text: str
) -> str:
    """Compute a hash of the inputs for caching purposes."""
    # Create fingerprint from transcription content
    text_len = len(transcription_text)
    text_sample = transcription_text[:500] + transcription_text[-200:] if text_len > 700 else transcription_text

    hash_input = json.dumps({
        'project_id': salesforce_project_id,
        'transcription_ids': sorted(transcription_ids),
        'text_fingerprint': text_sample,
        'text_length': text_len
    }, sort_keys=True)

    return hashlib.sha256(hash_input.encode()).hexdigest()


def save_project_sentiment_result(
    supabase,
    user_id: str,
    project: Dict,
    salesforce_account_id: str,
    salesforce_project_id: str,
    result: Dict,
    input_hash: str,
    transcription_count: int,
    transcription_length: int,
    transcription_ids: List[str],
    provider: str = 'openai',
    model: str = None
) -> bool:
    """Save project sentiment analysis result to database."""
    try:
        analysis_row = {
            'user_id': user_id,
            'account_id': project.get('account_id'),
            'salesforce_account_id': salesforce_account_id,
            'salesforce_project_id': salesforce_project_id,
            'project_name': project.get('project_name') or project.get('name'),
            'analyzed_at': datetime.now().isoformat(),
            'score': result.get('score'),
            'pm_effectiveness_score': result.get('pm_effectiveness_score'),
            'summary': result.get('summary'),
            'pm_summary': result.get('pm_summary'),
            'customer_sentiment_summary': result.get('customer_sentiment_summary'),
            'timeline_deliverables_summary': result.get('timeline_deliverables_summary'),
            'risks_summary': result.get('risks_summary'),
            'deliverables': result.get('deliverables', []),
            'timeline_dates': result.get('timeline_dates', []),
            'participants_sentiment': result.get('participants_sentiment', {}),
            'key_quotes': result.get('key_quotes', []),
            'input_hash': input_hash,
            'transcription_count': transcription_count,
            'transcription_length': transcription_length,
            'transcription_ids': transcription_ids,
            'raw_response': {'provider': provider, 'model': model}
        }

        result = supabase.table('project_sentiment_history').insert(analysis_row).execute()
        return True
    except Exception as e:
        print(f'Error saving project sentiment result: {e}')
        return False


class handler(BaseHTTPRequestHandler):
    """Main handler for Project Sentiment Analysis Crew requests"""

    def do_GET(self):
        """Handle GET requests - health check"""
        path = self.path.split('?')[0]
        if path != '/api/crew/project-sentiment':
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
            'service': 'CrewAI Project Sentiment Analysis',
            'provider': 'openai'
        }).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle POST requests - SSE streaming"""
        start_sse_response(self)

        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')

            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                send_error(self.wfile, 'Invalid JSON in request body')
                return

            # Extract parameters
            user_id = body.get('userId')
            salesforce_account_id = body.get('salesforceAccountId')
            salesforce_project_id = body.get('salesforceProjectId')
            transcription_ids = body.get('transcriptionIds', [])
            force_refresh = body.get('forceRefresh', False)

            # Validate required parameters
            if not salesforce_project_id:
                send_error(self.wfile, 'Missing required parameter: salesforceProjectId')
                return

            if not salesforce_account_id:
                send_error(self.wfile, 'Missing required parameter: salesforceAccountId')
                return

            send_progress(self.wfile, 'Initialization', 'Starting project sentiment analysis...', 'System')

            # Get Supabase client
            supabase = get_supabase_client()
            if not supabase:
                send_error(self.wfile, 'Database connection not available')
                return

            # Fetch project and transcriptions
            send_progress(self.wfile, 'Data Fetching', 'Loading project and meeting data...', 'System')

            try:
                data = fetch_project_and_transcriptions(
                    salesforce_project_id=salesforce_project_id,
                    salesforce_account_id=salesforce_account_id,
                    transcription_ids=transcription_ids if transcription_ids else None
                )
                project = data['project']
                transcriptions = data['transcriptions']
            except Exception as e:
                send_error(self.wfile, f'Error fetching data: {str(e)}')
                return

            if not transcriptions:
                send_error(self.wfile, 'No transcriptions found for analysis. Please select meetings with transcripts.')
                return

            # Build transcription context
            transcription_texts = []
            transcription_id_list = []
            total_length = 0

            for t in transcriptions:
                text = t.get('transcription_text', '')
                if text:
                    # Truncate very long transcriptions
                    if len(text) > 8000:
                        text = text[:8000] + '\n[...truncated...]'

                    meeting_info = f"\n\n=== MEETING: {t.get('meeting_subject', 'Unknown')} ({t.get('meeting_date', 'Unknown date')}) ===\n"
                    transcription_texts.append(meeting_info + text)
                    transcription_id_list.append(t.get('id'))
                    total_length += len(text)

            combined_transcriptions = '\n'.join(transcription_texts)

            # Compute input hash for caching
            input_hash = compute_input_hash(
                salesforce_project_id,
                transcription_id_list,
                combined_transcriptions
            )

            send_progress(self.wfile, 'Setup', f'Analyzing {len(transcriptions)} meetings ({total_length:,} characters)...', 'System')

            # Initialize CrewAI
            crewai = get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            openai_api_key = os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            # Build project context
            project_context = f"""
PROJECT INFORMATION:
- Project Name: {project.get('project_name') or project.get('name', 'Unknown')}
- Account: {project.get('account_name', 'Unknown')}
- Status: {project.get('project_status', 'Unknown')}
- Implementation Consultant: {project.get('project_manager_name', 'Unknown')}
- Start Date: {project.get('start_date', 'Unknown')}
- Target Go-Live: {project.get('target_go_live_date', 'Unknown')}
- % Complete: {project.get('completion_percentage', 'Unknown')}%
"""

            # Create agents
            send_progress(self.wfile, 'Agents', 'Creating analysis agents...', 'System')

            # Single comprehensive agent for efficiency
            project_analyst = Agent(
                role='Implementation Project Analyst',
                goal='Analyze project sentiment, PM effectiveness, customer reception, and extract deliverables',
                backstory='''You are an expert Implementation Program Reviewer specializing in B2B SaaS deployment projects.
You evaluate both the Implementation Consultant (PM) effectiveness AND customer sentiment/reception.
You understand that effective PMs set clear timelines, assign owners, confirm understanding, and drive alignment.
You look for specific behaviors and identify both strengths and coaching opportunities.
You apply RECENCY WEIGHTING: calls from the last 30 days are PRIMARY indicators (80-90% weight).''',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Create comprehensive analysis task
            send_progress(self.wfile, 'Analysis', 'Running comprehensive project analysis...', 'System')

            analysis_task = Task(
                description=f'''Analyze this implementation project comprehensively.

{project_context}

=== MEETING TRANSCRIPTS ===
{combined_transcriptions}

=== ANALYSIS REQUIREMENTS ===

Apply RECENCY WEIGHTING throughout:
- Last 30 days: PRIMARY indicators (80-90% weight)
- 30-60 days: SECONDARY (moderate weight)
- 60+ days: HISTORICAL context only

Analyze these dimensions:

1. PM EFFECTIVENESS (score 1-10):
   - Timeline setting and concrete date commitments
   - Owner and deliverable assignment clarity
   - Understanding confirmation and recap/next steps
   - Scope change and dependency management
   - Meeting flow (agenda, decisions, parking lot, follow-ups)
   - Handling customer pushback/confusion

2. CUSTOMER SENTIMENT (score 1-10):
   - Tone: collaborative vs combative
   - Confidence: confident vs anxious/hesitant
   - Engagement: engaged vs disengaged
   - Ownership clarity vs confusion
   - Frustration, skepticism, or misalignment signals

3. TIMELINE & DELIVERABLES:
   - Extract concrete dates mentioned with context
   - Identify deliverables with owners and status (committed/tentative/unclear)
   - Note contradictions or ambiguity

4. RISKS:
   - Identify project risks and blockers
   - Customer relationship risks
   - Timeline risks

=== OUTPUT FORMAT ===

Return a JSON object with this EXACT structure:
{{
  "score": <integer 1-10, overall project health>,
  "pm_effectiveness_score": <integer 1-10>,
  "summary": "<6-10 sentence executive summary>",
  "pm_summary": "<10-16 sentences covering: PM strengths, PM improvement areas, 6-10 coaching recommendations as directives, 2-4 example phrases PM could use>",
  "customer_sentiment_summary": "<6-10 sentences about customer tone and engagement>",
  "timeline_deliverables_summary": "<6-12 sentences about dates, deliverables, and status>",
  "risks_summary": "<6-10 sentences about risks and concerns>",
  "deliverables": [
    {{"deliverable": "", "owner": "", "due_date": "", "status": "committed|tentative|unclear", "evidence": "<quote>", "call": "<meeting subject>", "call_date": ""}}
  ],
  "timeline_dates": [
    {{"date": "", "context": "", "call": "", "call_date": ""}}
  ],
  "participants_sentiment": {{
    "customer": ["<tone descriptor>", "<engagement level>"],
    "pm": ["<communication style>", "<effectiveness indicator>"],
    "notes": "<2-4 coaching notes about interaction dynamics>"
  }},
  "key_quotes": [
    {{"speaker": "", "quote": "", "call": "", "call_date": ""}}
  ]
}}

SCORING RUBRIC:
- 9-10: Clear plan, owners, dates; customer aligned/positive; risks actively managed
- 7-8: Mostly strong; minor gaps but good momentum
- 5-6: Mixed; recurring ambiguity, weak follow-up, or customer hesitation
- 3-4: Significant risk; unclear ownership; customer frustration; PM not driving decisions
- 1-2: Derailing; high conflict/confusion; no credible plan''',
                agent=project_analyst,
                expected_output='JSON object with scores, summaries, deliverables, timeline, and key quotes'
            )

            # Create and run crew
            crew = Crew(
                agents=[project_analyst],
                tasks=[analysis_task],
                verbose=False
            )

            send_progress(self.wfile, 'Processing', 'AI analysis in progress...', 'AI Crew')

            result = crew.kickoff()
            result_text = str(result.raw if hasattr(result, 'raw') else result)

            send_progress(self.wfile, 'Parsing', 'Processing results...', 'System')

            # Parse JSON from result
            parsed_result = None
            try:
                # Handle markdown code blocks
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0]
                elif '```' in result_text:
                    parts = result_text.split('```')
                    if len(parts) >= 2:
                        result_text = parts[1]

                # Find JSON object
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result_text[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = json.loads(result_text.strip())

            except json.JSONDecodeError as e:
                print(f'JSON parse error: {e}')
                # Fallback structure
                parsed_result = {
                    'score': 5,
                    'pm_effectiveness_score': 5,
                    'summary': result_text[:1000],
                    'pm_summary': '',
                    'customer_sentiment_summary': '',
                    'timeline_deliverables_summary': '',
                    'risks_summary': '',
                    'deliverables': [],
                    'timeline_dates': [],
                    'participants_sentiment': {},
                    'key_quotes': []
                }

            # Validate and normalize scores
            score = parsed_result.get('score', 5)
            if not isinstance(score, int) or score < 1 or score > 10:
                score = 5
            parsed_result['score'] = score

            pm_score = parsed_result.get('pm_effectiveness_score', 5)
            if not isinstance(pm_score, int) or pm_score < 1 or pm_score > 10:
                pm_score = 5
            parsed_result['pm_effectiveness_score'] = pm_score

            # Ensure arrays exist
            parsed_result['deliverables'] = parsed_result.get('deliverables', [])
            parsed_result['timeline_dates'] = parsed_result.get('timeline_dates', [])
            parsed_result['key_quotes'] = parsed_result.get('key_quotes', [])
            parsed_result['participants_sentiment'] = parsed_result.get('participants_sentiment', {})

            # Save to database
            send_progress(self.wfile, 'Saving', 'Saving analysis results...', 'System')

            save_project_sentiment_result(
                supabase=supabase,
                user_id=user_id,
                project=project,
                salesforce_account_id=salesforce_account_id,
                salesforce_project_id=salesforce_project_id,
                result=parsed_result,
                input_hash=input_hash,
                transcription_count=len(transcriptions),
                transcription_length=total_length,
                transcription_ids=transcription_id_list,
                provider='openai'
            )

            # Send final result
            result_data = {
                'type': 'result',
                'result': parsed_result,
                'input_hash': input_hash,
                'transcription_count': len(transcriptions),
                'transcription_length': total_length,
                'transcription_ids': transcription_id_list,
                'provider': 'openai',
                'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
            }

            send_sse_message(self.wfile, result_data)

        except Exception as e:
            import traceback
            error_message = str(e)
            print(f'Error in project sentiment analysis: {error_message}')
            traceback.print_exc()

            if os.environ.get('DEBUG'):
                error_message = f'{error_message}\n{traceback.format_exc()}'

            send_error(self.wfile, error_message)
