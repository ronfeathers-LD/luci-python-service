"""
Vercel Python Serverless Function for Support Training Crew
Provides real-time personalized suggestions for open cases using pre-computed profiles.

This crew is designed to be FAST (<10 seconds) by:
1. Using pre-computed profiles from support_agent_profiles table
2. Single LLM call with focused context
3. Streaming response via SSE

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded.
"""

import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error, send_sse_message

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
    """Get OpenAI LLM instance optimized for speed"""
    ChatOpenAI = _get_langchain_openai()
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is required')
    os.environ['OPENAI_API_KEY'] = api_key
    return ChatOpenAI(
        api_key=api_key,
        model=os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
        temperature=0.5  # Balanced for creativity and consistency
    )


def fetch_team_best_practices(supabase) -> Optional[Dict]:
    """Fetch the team best practices profile."""
    try:
        result = supabase.table('support_agent_profiles').select(
            'profile_data, analyzed_at'
        ).eq(
            'profile_type', 'team_best_practices'
        ).limit(1).execute()

        if result.data and len(result.data) > 0:
            return result.data[0].get('profile_data')
    except Exception as e:
        print(f'Error fetching team best practices: {e}')
    return None


def fetch_user_profile(supabase, user_id: Optional[str] = None, salesforce_user_id: Optional[str] = None) -> Optional[Dict]:
    """Fetch the individual user's profile."""
    try:
        query = supabase.table('support_agent_profiles').select(
            'profile_data, performance_percentile, total_cases_analyzed, avg_resolution_hours, analyzed_at'
        ).eq('profile_type', 'individual')

        if user_id:
            query = query.eq('user_id', user_id)
        elif salesforce_user_id:
            query = query.eq('salesforce_user_id', salesforce_user_id)
        else:
            return None

        result = query.limit(1).execute()

        if result.data and len(result.data) > 0:
            data = result.data[0]
            return {
                'profile_data': data.get('profile_data'),
                'performance_percentile': data.get('performance_percentile'),
                'total_cases_analyzed': data.get('total_cases_analyzed'),
                'avg_resolution_hours': data.get('avg_resolution_hours'),
            }
    except Exception as e:
        print(f'Error fetching user profile: {e}')
    return None


def fetch_similar_resolved_cases(supabase, case_type: str, case_subject: str, owner_id: str, limit: int = 3) -> List[Dict]:
    """Find similar resolved cases from this agent's history."""
    try:
        # Get closed cases with similar type or subject keywords
        result = supabase.table('cases').select(
            'case_number, subject, type, closed_date'
        ).eq(
            'owner_id', owner_id
        ).not_.is_(
            'closed_date', 'null'
        ).order(
            'closed_date', desc=True
        ).limit(50).execute()

        if not result.data:
            return []

        # Simple keyword matching for similarity
        subject_words = set(case_subject.lower().split())
        similar = []

        for case in result.data:
            case_subject_words = set((case.get('subject') or '').lower().split())
            overlap = len(subject_words & case_subject_words)

            # Match on type or significant word overlap
            if case.get('type') == case_type or overlap >= 2:
                similar.append({
                    'case_number': case['case_number'],
                    'subject': case['subject'],
                    'type': case.get('type'),
                })

        return similar[:limit]

    except Exception as e:
        print(f'Error fetching similar cases: {e}')
        return []


def build_suggestion_context(
    team_practices: Optional[Dict],
    user_profile: Optional[Dict],
    similar_cases: List[Dict]
) -> str:
    """Build the context string from pre-computed profiles."""
    context_parts = []

    # Team best practices
    if team_practices:
        context_parts.append("=== TEAM BEST PRACTICES ===")

        if team_practices.get('communication_guidelines'):
            guidelines = team_practices['communication_guidelines']
            context_parts.append(f"Recommended Tone: {guidelines.get('recommended_tone', 'Professional and helpful')}")
            context_parts.append(f"Technical Style: {guidelines.get('technical_explanation_style', 'Clear and accessible')}")

        if team_practices.get('common_phrases'):
            phrases = team_practices['common_phrases']
            if phrases.get('openings'):
                context_parts.append(f"Effective Openings: {', '.join(phrases['openings'][:3])}")

        if team_practices.get('diagnostic_framework'):
            framework = team_practices['diagnostic_framework']
            if framework.get('initial_questions'):
                context_parts.append(f"Key Diagnostic Questions: {', '.join(framework['initial_questions'][:3])}")

    # User's personal style
    if user_profile and user_profile.get('profile_data'):
        profile = user_profile['profile_data']
        context_parts.append("\n=== YOUR PERSONAL STYLE ===")

        if profile.get('communication_patterns'):
            patterns = profile['communication_patterns']
            context_parts.append(f"Your Tone: {patterns.get('tone', 'Not analyzed yet')}")
            if patterns.get('empathy_markers'):
                context_parts.append(f"Your Empathy Phrases: {', '.join(patterns['empathy_markers'][:3])}")

        if profile.get('expertise_areas'):
            areas = profile['expertise_areas'][:3]
            expertise_str = ', '.join([f"{a['category']} ({int(a['confidence']*100)}%)" for a in areas])
            context_parts.append(f"Your Expertise: {expertise_str}")

        if profile.get('successful_patterns'):
            patterns = profile['successful_patterns']
            if patterns.get('opening_phrases'):
                context_parts.append(f"Your Typical Opening: {patterns['opening_phrases'][0]}")

        # Performance context
        if user_profile.get('performance_percentile'):
            context_parts.append(f"Your Performance: Top {100 - user_profile['performance_percentile']}% of team")

    # Similar resolved cases
    if similar_cases:
        context_parts.append("\n=== SIMILAR CASES YOU RESOLVED ===")
        for case in similar_cases:
            context_parts.append(f"- {case['case_number']}: {case['subject']}")

    return '\n'.join(context_parts) if context_parts else "(No pre-computed profiles available yet)"


class handler(BaseHTTPRequestHandler):
    """Handler for Support Training Crew - real-time suggestions"""

    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'service': 'Support Training Crew',
            'description': 'Real-time personalized case suggestions'
        }).encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Generate suggestions for a case - SSE streaming"""
        start_sse_response(self)

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body_str = self.rfile.read(content_length).decode('utf-8')

            try:
                body = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                send_error(self.wfile, 'Invalid JSON in request body')
                return

            # Extract case details
            case_number = body.get('caseNumber', '')
            case_subject = body.get('caseSubject', '')
            case_description = body.get('caseDescription', '')
            case_type = body.get('caseType', '')
            case_priority = body.get('casePriority', '')
            account_name = body.get('accountName', 'Unknown')
            contact_name = body.get('contactName', '')

            # User context
            user_id = body.get('userId')
            salesforce_user_id = body.get('salesforceUserId')

            if not case_subject and not case_description:
                send_error(self.wfile, 'Missing required: caseSubject or caseDescription')
                return

            send_progress(self.wfile, 'Loading', 'Loading your profile and best practices...', 'System')

            # Get Supabase client
            supabase = get_supabase_client()
            if not supabase:
                send_error(self.wfile, 'Database not configured')
                return

            # Fetch profiles in parallel-ish (Python sequential but fast DB calls)
            team_practices = fetch_team_best_practices(supabase)
            user_profile = fetch_user_profile(supabase, user_id, salesforce_user_id)

            # Fetch similar cases this user resolved
            similar_cases = []
            if salesforce_user_id:
                similar_cases = fetch_similar_resolved_cases(
                    supabase, case_type, case_subject, salesforce_user_id
                )

            send_progress(self.wfile, 'Analyzing', 'Generating personalized suggestions...', 'AI Agent')

            # Build context from profiles
            profile_context = build_suggestion_context(team_practices, user_profile, similar_cases)

            # Check if we have any useful context
            has_profiles = bool(team_practices or (user_profile and user_profile.get('profile_data')))

            # Initialize LLM and CrewAI
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key)

            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Create suggestion agent
            coach_agent = Agent(
                role='Support Response Coach',
                goal='Generate personalized, actionable suggestions for handling support cases based on team best practices and individual agent style',
                backstory='Expert support coach who combines team knowledge with individual agent strengths to provide tailored guidance. Focuses on practical, immediately usable suggestions.',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Build the task
            suggestion_task = Task(
                description=f'''Generate personalized suggestions for handling this support case.

=== CASE DETAILS ===
Case Number: {case_number}
Subject: {case_subject}
Type: {case_type}
Priority: {case_priority}
Account: {account_name}
Contact: {contact_name}

Description:
{case_description[:1500] if case_description else "(No description provided)"}

{profile_context}

=== GENERATE SUGGESTIONS ===

Based on the case details and the agent's profile/team practices, provide:

1. RECOMMENDED APPROACH
   - How should this case be approached given the agent's strengths?
   - What's the likely issue category?

2. SUGGESTED OPENING
   - A personalized opening message matching the agent's style
   - Should acknowledge the customer and show understanding

3. KEY DIAGNOSTIC QUESTIONS
   - 2-3 questions to ask the customer to diagnose the issue
   - Prioritize based on case type

4. RESOLUTION PATH
   - Likely resolution steps based on case type
   - Reference similar resolved cases if available

5. CONFIDENCE SCORE
   - How well does this case match the agent's expertise?

=== OUTPUT FORMAT ===

Return a JSON object:
{{
  "recommendedApproach": {{
    "summary": "<1-2 sentence approach recommendation>",
    "issueCategory": "<routing|matching|integration|permissions|data|other>",
    "estimatedComplexity": "<simple|moderate|complex>",
    "matchesExpertise": <true|false>
  }},
  "suggestedOpening": "<personalized opening message ready to use>",
  "diagnosticQuestions": [
    "<question 1>",
    "<question 2>",
    "<question 3>"
  ],
  "resolutionSteps": [
    "<step 1>",
    "<step 2>",
    "<step 3>"
  ],
  "similarCasesReference": "<mention of similar cases if available>",
  "confidenceScore": <0.0-1.0>,
  "tips": ["<tip based on team best practices>"]
}}''',
                agent=coach_agent,
                expected_output='JSON object with personalized case handling suggestions'
            )

            # Run crew
            crew = Crew(
                agents=[coach_agent],
                tasks=[suggestion_task],
                verbose=False
            )

            result = crew.kickoff()
            result_text = str(result)

            send_progress(self.wfile, 'Complete', 'Suggestions ready!', 'System')

            # Parse result
            parsed_result = None
            try:
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    parsed_result = json.loads(result_text[start_idx:end_idx])
            except json.JSONDecodeError:
                pass

            if not parsed_result:
                # Fallback structure
                parsed_result = {
                    "recommendedApproach": {
                        "summary": "Review case details and gather more information",
                        "issueCategory": "other",
                        "estimatedComplexity": "moderate",
                        "matchesExpertise": False
                    },
                    "suggestedOpening": f"Hi {contact_name or 'there'}, thank you for reaching out about {case_subject}. I'm reviewing your case and will provide an update shortly.",
                    "diagnosticQuestions": [
                        "Can you provide more details about when this issue started?",
                        "Have there been any recent changes to your configuration?",
                        "Can you share a screenshot or example of the issue?"
                    ],
                    "resolutionSteps": [
                        "Review case details and history",
                        "Gather additional information from customer",
                        "Investigate root cause"
                    ],
                    "confidenceScore": 0.5,
                    "tips": ["Check similar resolved cases for patterns"],
                    "rawResponse": result_text[:500]
                }

            # Add metadata
            parsed_result['metadata'] = {
                'hasTeamPractices': bool(team_practices),
                'hasUserProfile': bool(user_profile and user_profile.get('profile_data')),
                'userPerformancePercentile': user_profile.get('performance_percentile') if user_profile else None,
                'similarCasesFound': len(similar_cases),
            }

            # Send result
            send_sse_message(self.wfile, {
                'type': 'result',
                'result': parsed_result
            })

        except Exception as e:
            import traceback
            error_message = str(e)
            print(f'Error in support training crew: {error_message}')
            print(traceback.format_exc())
            send_error(self.wfile, error_message)
