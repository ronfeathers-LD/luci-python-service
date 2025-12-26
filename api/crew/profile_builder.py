"""
Vercel Python Serverless Function for Support Agent Profile Builder Crew
Analyzes top performer patterns and builds reusable profiles for training.

Runs weekly via cron job to pre-compute profiles that the real-time
suggestion crew will use.

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time.
"""

import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings early
warnings.filterwarnings('ignore', message='.*Overriding of current TracerProvider.*')
warnings.filterwarnings('ignore', category=UserWarning, module='opentelemetry')
warnings.filterwarnings('ignore', category=SyntaxWarning, module='langchain')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Mixing V1 models and V2 models.*')
warnings.filterwarnings('ignore', category=UserWarning, module='crewai')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# Import shared helpers
from crew.database_helpers import get_supabase_client

# Lazy-loaded modules
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
    """Get OpenAI LLM instance"""
    ChatOpenAI = _get_langchain_openai()
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is required')
    os.environ['OPENAI_API_KEY'] = api_key
    return ChatOpenAI(
        api_key=api_key,
        model=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
        temperature=0.3  # Lower temperature for more consistent analysis
    )


def fetch_agent_metrics(supabase, days_back: int = 90) -> List[Dict[str, Any]]:
    """
    Fetch performance metrics for all support agents with closed cases.
    Returns list of agents with their metrics.
    """
    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

    # Get all closed cases with owner info
    result = supabase.table('cases').select(
        'owner_id, owner_name, created_date, closed_date, case_number, subject, type'
    ).not_(
        'closed_date', 'is', None
    ).not_(
        'owner_id', 'is', None
    ).gte(
        'closed_date', cutoff_date
    ).execute()

    if not result.data:
        return []

    # Group by owner and calculate metrics
    agent_cases = {}
    for case in result.data:
        owner_id = case['owner_id']
        if owner_id not in agent_cases:
            agent_cases[owner_id] = {
                'owner_id': owner_id,
                'owner_name': case.get('owner_name', 'Unknown'),
                'cases': [],
                'resolution_times': [],
            }

        # Calculate resolution time in hours
        if case.get('created_date') and case.get('closed_date'):
            try:
                created = datetime.fromisoformat(case['created_date'].replace('Z', '+00:00'))
                closed = datetime.fromisoformat(case['closed_date'].replace('Z', '+00:00'))
                resolution_hours = (closed - created).total_seconds() / 3600
                agent_cases[owner_id]['resolution_times'].append(resolution_hours)
            except Exception:
                pass

        agent_cases[owner_id]['cases'].append(case)

    # Calculate aggregate metrics
    agents = []
    for owner_id, data in agent_cases.items():
        case_count = len(data['cases'])
        resolution_times = data['resolution_times']

        avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else None

        agents.append({
            'owner_id': owner_id,
            'owner_name': data['owner_name'],
            'total_cases': case_count,
            'avg_resolution_hours': round(avg_resolution, 2) if avg_resolution else None,
            'cases': data['cases'][:10],  # Keep sample for analysis
        })

    return agents


def identify_top_performers(agents: List[Dict], percentile: int = 80) -> List[Dict]:
    """
    Identify top performers based on case volume and resolution time.
    Returns agents in the top percentile.
    """
    if not agents:
        return []

    # Filter agents with enough data
    qualified = [a for a in agents if a['total_cases'] >= 5 and a.get('avg_resolution_hours')]

    if not qualified:
        return []

    # Sort by resolution time (lower is better) and case volume (higher is better)
    # Composite score: normalize both metrics
    max_cases = max(a['total_cases'] for a in qualified)
    max_resolution = max(a['avg_resolution_hours'] for a in qualified)

    for agent in qualified:
        # Higher score = better performer
        volume_score = agent['total_cases'] / max_cases if max_cases > 0 else 0
        speed_score = 1 - (agent['avg_resolution_hours'] / max_resolution) if max_resolution > 0 else 0
        agent['performance_score'] = (volume_score * 0.4) + (speed_score * 0.6)

    # Sort by performance score
    qualified.sort(key=lambda x: x['performance_score'], reverse=True)

    # Get top percentile
    cutoff_index = max(1, int(len(qualified) * (100 - percentile) / 100))
    top_performers = qualified[:cutoff_index]

    # Assign percentile ranks
    for i, agent in enumerate(qualified):
        agent['performance_percentile'] = int(100 - (i / len(qualified) * 100))

    return top_performers


def fetch_agent_responses(supabase, owner_id: str, limit: int = 50) -> Dict[str, List]:
    """
    Fetch case comments and emails for a specific agent.
    """
    responses = {'comments': [], 'emails': []}

    # Fetch comments authored by this agent
    try:
        comments_result = supabase.table('case_comments').select(
            'comment_body, is_published, created_date, salesforce_case_id'
        ).eq(
            'author_id', owner_id
        ).order(
            'created_date', desc=True
        ).limit(limit).execute()

        if comments_result.data:
            responses['comments'] = comments_result.data
    except Exception as e:
        print(f'Error fetching comments for {owner_id}: {e}')

    # Fetch outgoing emails (agent responses)
    try:
        # Get cases owned by this agent
        cases_result = supabase.table('cases').select('salesforce_id').eq('owner_id', owner_id).limit(100).execute()

        if cases_result.data:
            case_ids = [c['salesforce_id'] for c in cases_result.data if c.get('salesforce_id')]

            if case_ids:
                emails_result = supabase.table('case_emails').select(
                    'subject, text_body, from_address, message_date, salesforce_case_id'
                ).in_(
                    'salesforce_case_id', case_ids
                ).eq(
                    'incoming', False  # Outgoing emails from agent
                ).order(
                    'message_date', desc=True
                ).limit(limit).execute()

                if emails_result.data:
                    responses['emails'] = emails_result.data
    except Exception as e:
        print(f'Error fetching emails for {owner_id}: {e}')

    return responses


def analyze_agent_patterns(llm, agent: Dict, responses: Dict) -> Dict[str, Any]:
    """
    Use CrewAI to analyze an agent's response patterns.
    Returns structured profile data.
    """
    crewai = _get_crewai()
    Agent = crewai['Agent']
    Task = crewai['Task']
    Crew = crewai['Crew']

    # Build sample responses text
    sample_texts = []

    for comment in responses.get('comments', [])[:20]:
        if comment.get('comment_body'):
            sample_texts.append(f"[Case Comment]\n{comment['comment_body'][:500]}")

    for email in responses.get('emails', [])[:20]:
        if email.get('text_body'):
            sample_texts.append(f"[Email Response]\nSubject: {email.get('subject', 'N/A')}\n{email['text_body'][:500]}")

    if not sample_texts:
        return None

    responses_text = "\n\n---\n\n".join(sample_texts[:15])

    # Create analysis agent
    pattern_analyst = Agent(
        role='Support Response Analyst',
        goal='Analyze support agent response patterns to identify communication style, expertise areas, and successful techniques',
        backstory='Expert in analyzing customer support interactions to extract patterns that lead to successful case resolutions. Focuses on identifying replicable best practices.',
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )

    # Create analysis task
    analysis_task = Task(
        description=f'''Analyze these support agent responses to identify patterns and best practices.

=== AGENT INFO ===
Name: {agent.get('owner_name', 'Unknown')}
Total Cases Resolved: {agent.get('total_cases', 0)}
Avg Resolution Time: {agent.get('avg_resolution_hours', 'N/A')} hours

=== SAMPLE RESPONSES ===
{responses_text}

=== ANALYSIS REQUIREMENTS ===

Analyze the responses and extract:

1. COMMUNICATION PATTERNS
   - Tone (formal/casual/mixed)
   - Technical depth (how they explain technical concepts)
   - Empathy markers (phrases showing understanding)
   - Average response structure

2. EXPERTISE AREAS
   - What topics do they handle confidently?
   - Infer from terminology and detail level

3. SUCCESSFUL PATTERNS
   - Common opening phrases
   - Diagnostic questions they ask
   - Resolution approaches they use
   - Closing/follow-up patterns

4. SAMPLE TEMPLATES
   - Extract 2-3 reusable response snippets

=== OUTPUT FORMAT ===

Return a JSON object:
{{
  "communication_patterns": {{
    "tone": "<description>",
    "technical_depth": "<description>",
    "empathy_markers": ["<phrase1>", "<phrase2>"],
    "avg_response_length": <estimated_words>
  }},
  "expertise_areas": [
    {{"category": "<topic>", "confidence": <0.0-1.0>}}
  ],
  "successful_patterns": {{
    "opening_phrases": ["<phrase1>", "<phrase2>"],
    "diagnostic_questions": ["<question1>", "<question2>"],
    "resolution_approaches": ["<approach1>", "<approach2>"],
    "closing_patterns": ["<pattern1>"]
  }},
  "sample_responses": [
    {{
      "use_case": "<when to use>",
      "template": "<reusable response template>"
    }}
  ]
}}''',
        agent=pattern_analyst,
        expected_output='JSON object with communication patterns, expertise areas, successful patterns, and sample responses'
    )

    crew = Crew(
        agents=[pattern_analyst],
        tasks=[analysis_task],
        verbose=False
    )

    result = crew.kickoff()
    result_text = str(result)

    # Parse JSON result
    try:
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(result_text[start_idx:end_idx])
    except json.JSONDecodeError:
        print(f'Failed to parse agent analysis result')

    return None


def build_team_best_practices(llm, top_performers: List[Dict], individual_profiles: List[Dict]) -> Dict[str, Any]:
    """
    Synthesize team-wide best practices from top performer profiles.
    """
    if not individual_profiles:
        return None

    crewai = _get_crewai()
    Agent = crewai['Agent']
    Task = crewai['Task']
    Crew = crewai['Crew']

    # Build profiles summary
    profiles_text = ""
    for i, profile in enumerate(individual_profiles[:5]):
        if profile:
            profiles_text += f"\n=== TOP PERFORMER {i+1} ===\n"
            profiles_text += json.dumps(profile, indent=2)[:2000]
            profiles_text += "\n"

    synthesis_agent = Agent(
        role='Best Practices Synthesizer',
        goal='Synthesize common patterns from top performers into team-wide best practices',
        backstory='Expert in identifying and codifying support best practices from high performers to create training materials for the entire team.',
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )

    synthesis_task = Task(
        description=f'''Synthesize team-wide best practices from these top performer profiles.

{profiles_text}

=== REQUIREMENTS ===

Create a unified best practices guide that:
1. Identifies COMMON patterns across top performers
2. Extracts the MOST effective techniques
3. Creates REUSABLE templates

=== OUTPUT FORMAT ===

Return a JSON object:
{{
  "communication_guidelines": {{
    "recommended_tone": "<description>",
    "technical_explanation_style": "<how to explain technical concepts>",
    "empathy_best_practices": ["<practice1>", "<practice2>"]
  }},
  "response_structure": {{
    "opening": "<recommended opening approach>",
    "body": "<recommended body structure>",
    "closing": "<recommended closing approach>"
  }},
  "diagnostic_framework": {{
    "initial_questions": ["<question1>", "<question2>"],
    "follow_up_triggers": ["<when to ask what>"]
  }},
  "resolution_playbook": [
    {{
      "issue_type": "<category>",
      "recommended_approach": "<approach>",
      "template": "<reusable template>"
    }}
  ],
  "common_phrases": {{
    "openings": ["<phrase1>", "<phrase2>"],
    "transitions": ["<phrase1>"],
    "closings": ["<phrase1>", "<phrase2>"]
  }}
}}''',
        agent=synthesis_agent,
        expected_output='JSON object with synthesized team best practices'
    )

    crew = Crew(
        agents=[synthesis_agent],
        tasks=[synthesis_task],
        verbose=False
    )

    result = crew.kickoff()
    result_text = str(result)

    try:
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(result_text[start_idx:end_idx])
    except json.JSONDecodeError:
        print('Failed to parse team best practices result')

    return None


def save_profile(supabase, profile_type: str, profile_data: Dict, agent_info: Optional[Dict] = None) -> bool:
    """Save a profile to the support_agent_profiles table."""
    try:
        save_data = {
            'profile_type': profile_type,
            'profile_data': profile_data,
            'analyzed_at': datetime.now().isoformat(),
        }

        if agent_info:
            save_data['salesforce_user_id'] = agent_info.get('owner_id')
            save_data['total_cases_analyzed'] = agent_info.get('total_cases', 0)
            save_data['avg_resolution_hours'] = agent_info.get('avg_resolution_hours')
            save_data['performance_percentile'] = agent_info.get('performance_percentile')

            # Try to find matching user_id
            if agent_info.get('owner_id'):
                user_result = supabase.table('users').select('id').eq(
                    'salesforce_user_id', agent_info['owner_id']
                ).limit(1).execute()
                if user_result.data:
                    save_data['user_id'] = user_result.data[0]['id']

        # Upsert based on profile type
        if profile_type == 'team_best_practices':
            # Delete existing team profile first (unique constraint)
            supabase.table('support_agent_profiles').delete().eq(
                'profile_type', 'team_best_practices'
            ).execute()

        elif profile_type == 'individual' and agent_info and agent_info.get('owner_id'):
            # Delete existing profile for this agent
            supabase.table('support_agent_profiles').delete().eq(
                'profile_type', 'individual'
            ).eq(
                'salesforce_user_id', agent_info['owner_id']
            ).execute()

        result = supabase.table('support_agent_profiles').insert(save_data).execute()

        if hasattr(result, 'error') and result.error:
            print(f'Error saving profile: {result.error}')
            return False

        print(f'✅ Saved {profile_type} profile')
        return True

    except Exception as e:
        print(f'Error saving profile: {e}')
        return False


class handler(BaseHTTPRequestHandler):
    """Handler for Profile Builder Crew requests"""

    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'service': 'Support Agent Profile Builder',
            'description': 'Builds training profiles from top performer patterns'
        }).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Internal-Cron-Secret')
        self.end_headers()

    def do_POST(self):
        """Build profiles - main entry point"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')
            body = json.loads(body_str) if body_str else {}

            # Verify cron secret for automated calls
            cron_secret = self.headers.get('X-Internal-Cron-Secret')
            expected_secret = os.environ.get('CRON_SECRET')

            # Allow if cron secret matches or if explicit trigger
            is_authorized = (cron_secret and expected_secret and cron_secret == expected_secret) or body.get('manualTrigger')

            if not is_authorized:
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Unauthorized'}).encode('utf-8'))
                return

            days_back = body.get('daysBack', 90)
            top_percentile = body.get('topPercentile', 80)

            print(f'Starting profile builder: {days_back} days back, top {100-top_percentile}%')

            supabase = get_supabase_client()
            if not supabase:
                raise ValueError('Supabase not configured')

            # Step 1: Fetch agent metrics
            print('Step 1: Fetching agent metrics...')
            agents = fetch_agent_metrics(supabase, days_back)
            print(f'Found {len(agents)} agents with closed cases')

            if not agents:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'completed',
                    'message': 'No agents with closed cases found',
                    'results': {'agents': 0, 'profiles': 0}
                }).encode('utf-8'))
                return

            # Step 2: Identify top performers
            print('Step 2: Identifying top performers...')
            top_performers = identify_top_performers(agents, top_percentile)
            print(f'Identified {len(top_performers)} top performers')

            if not top_performers:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'completed',
                    'message': 'Not enough qualified agents for analysis',
                    'results': {'agents': len(agents), 'qualified': 0, 'profiles': 0}
                }).encode('utf-8'))
                return

            # Initialize LLM
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key)

            # Step 3: Analyze each top performer
            print('Step 3: Analyzing top performer patterns...')
            individual_profiles = []
            profiles_saved = 0

            for agent in top_performers[:5]:  # Limit to top 5 for performance
                print(f'  Analyzing {agent["owner_name"]}...')

                # Fetch their responses
                responses = fetch_agent_responses(supabase, agent['owner_id'])
                total_responses = len(responses.get('comments', [])) + len(responses.get('emails', []))

                if total_responses < 5:
                    print(f'    Skipping - only {total_responses} responses')
                    continue

                # Analyze patterns
                profile = analyze_agent_patterns(llm, agent, responses)

                if profile:
                    individual_profiles.append(profile)

                    # Save individual profile
                    if save_profile(supabase, 'individual', profile, agent):
                        profiles_saved += 1

            # Step 4: Build team best practices
            print('Step 4: Building team best practices...')
            if individual_profiles:
                team_practices = build_team_best_practices(llm, top_performers, individual_profiles)

                if team_practices:
                    if save_profile(supabase, 'team_best_practices', team_practices):
                        profiles_saved += 1

            # Return results
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            result = {
                'status': 'completed',
                'message': 'Profile building completed',
                'results': {
                    'total_agents': len(agents),
                    'top_performers_analyzed': len(top_performers),
                    'individual_profiles_created': len(individual_profiles),
                    'profiles_saved': profiles_saved,
                    'team_best_practices_created': bool(individual_profiles),
                }
            }

            self.wfile.write(json.dumps(result).encode('utf-8'))
            print(f'Profile building completed: {profiles_saved} profiles saved')

        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f'Error in profile builder: {error_msg}')
            print(traceback.format_exc())

            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'error': error_msg
            }).encode('utf-8'))
