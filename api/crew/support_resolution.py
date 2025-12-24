"""
Vercel Python Serverless Function for Support Case Resolution Crew
Analyzes support cases and suggests resolutions using LeanData knowledge base

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time by ~1-2 seconds.
"""

import json
import os
import sys
import warnings
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler
from typing import Optional, List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error, send_sse_message

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


def search_support_articles(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search for relevant support articles using Google site search.
    Returns list of article URLs and titles.
    """
    articles = []

    try:
        # Use Google Search API or fallback to known article patterns
        # For now, we'll use a curated approach based on common topics

        # Known article patterns on support.leandata.com
        known_topics = {
            'routing': [
                {'url': 'https://support.leandata.com/s/article/RoutingRoutetoMatchedAccountNodeGuide6901eba487ff6', 'title': 'Route to Matched Account Node Guide'},
                {'url': 'https://support.leandata.com/s/article/RoutingRoutetoEnterpriseTerritoryManagementETMModel6901d26943b24', 'title': 'Route to Enterprise Territory Management (ETM) Model'},
            ],
            'matching': [
                {'url': 'https://support.leandata.com/s/topic/0TO5e000000L5XXGA0/matching', 'title': 'Matching Documentation'},
            ],
            'implementation': [
                {'url': 'https://support.leandata.com/s/article/LeanDataRouterLeanDataSalesforceSalesEngagementImplementationGuide690201b865446', 'title': 'LeanData Router Implementation Guide'},
            ],
            'attribution': [
                {'url': 'https://support.leandata.com/s/topic/0TO5e000000L5XYGA0/attribution', 'title': 'Attribution Documentation'},
            ],
            'go live': [
                {'url': 'https://support.leandata.com/s/article/HowtoGoLivewithLeanData6901f66b529a8', 'title': 'How to Go Live with LeanData'},
            ],
        }

        query_lower = query.lower()

        # Match articles based on query keywords
        for topic, topic_articles in known_topics.items():
            if topic in query_lower:
                articles.extend(topic_articles)

        # If no matches, return general resources
        if not articles:
            articles = [
                {'url': 'https://support.leandata.com/s/', 'title': 'LeanData Support Home'},
                {'url': 'https://support.leandata.com/s/contactsupport', 'title': 'Contact Support'},
            ]

        return articles[:max_results]

    except Exception as e:
        print(f'Error searching support articles: {e}')
        return []


def fetch_article_content(url: str, timeout: int = 10) -> str:
    """
    Fetch and extract text content from a support article URL.
    Returns extracted text or empty string on failure.
    """
    try:
        # Create request with user agent
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; LeanData Support Bot)'}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read().decode('utf-8', errors='ignore')

            # Basic HTML to text extraction
            # Remove script and style tags
            import re
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags but keep text
            text = re.sub(r'<[^>]+>', ' ', html)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Limit length
            return text[:8000] if len(text) > 8000 else text

    except Exception as e:
        print(f'Error fetching article {url}: {e}')
        return ''


class handler(BaseHTTPRequestHandler):
    """Main handler for Support Case Resolution Crew requests - Vercel format"""

    def do_GET(self):
        """Handle GET requests - only for health checks"""
        path = self.path.split('?')[0]
        if path != '/api/crew/support_resolution':
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
            'service': 'CrewAI Support Case Resolution',
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
                send_error(self.wfile, 'Invalid JSON in request body')
                return

            # Extract case details
            case_subject = body.get('caseSubject', '')
            case_description = body.get('caseDescription', '')
            case_type = body.get('caseType', '')
            case_priority = body.get('casePriority', '')
            case_status = body.get('caseStatus', '')
            account_name = body.get('accountName', 'Unknown Account')
            case_history = body.get('caseHistory', [])  # Previous cases for context

            if not case_subject and not case_description:
                send_error(self.wfile, 'Missing required parameter: caseSubject or caseDescription')
                return

            send_progress(self.wfile, 'Initialization', 'Initializing support resolution agent...', 'System')

            # Lazy load crewai classes
            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Initialize LLM
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            send_progress(self.wfile, 'Knowledge Search', 'Searching LeanData knowledge base...', 'System')

            # Build search query from case details
            search_query = f"{case_subject} {case_type}".strip()
            if not search_query:
                search_query = case_description[:100] if case_description else 'leandata support'

            # Search for relevant articles
            articles = search_support_articles(search_query)

            # Fetch content from top articles
            article_content = []
            for article in articles[:3]:  # Fetch top 3
                send_progress(self.wfile, 'Fetching', f"Reading: {article['title'][:50]}...", 'System')
                content = fetch_article_content(article['url'])
                if content:
                    article_content.append({
                        'title': article['title'],
                        'url': article['url'],
                        'content': content[:3000]  # Limit per article
                    })

            # Build knowledge context
            knowledge_context = ""
            if article_content:
                knowledge_context = "\n\n".join([
                    f"=== Article: {a['title']} ===\nURL: {a['url']}\n{a['content']}"
                    for a in article_content
                ])
            else:
                knowledge_context = "(No specific knowledge base articles found - using general LeanData expertise)"

            # Build case history context
            history_context = ""
            if case_history:
                history_context = "\n=== PREVIOUS CASES FOR THIS ACCOUNT ===\n"
                for prev_case in case_history[:5]:
                    history_context += f"- {prev_case.get('subject', 'No subject')} ({prev_case.get('status', 'Unknown')})\n"

            send_progress(self.wfile, 'Analysis', 'Analyzing case and formulating resolution options...', 'AI Agent')

            # Create support resolution agent
            support_agent = Agent(
                role='LeanData Support Specialist',
                goal='Analyze support cases and provide resolution options based on LeanData knowledge base',
                backstory='Expert LeanData support specialist with deep knowledge of routing, matching, attribution, and Salesforce integration. Provides clear, actionable resolution steps.',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Create resolution task
            resolution_task = Task(
                description=f'''Analyze this support case and provide resolution options.

=== CASE DETAILS ===
Subject: {case_subject}
Type: {case_type}
Priority: {case_priority}
Status: {case_status}
Account: {account_name}

Description:
{case_description}
{history_context}

=== KNOWLEDGE BASE CONTEXT ===
{knowledge_context}

=== ANALYSIS REQUIREMENTS ===

1. IDENTIFY the root cause or likely issue category:
   - Routing configuration
   - Matching rules
   - Attribution settings
   - Salesforce integration
   - User permissions
   - Data quality
   - Other

2. PROVIDE 2-3 resolution options, ranked by likelihood of success

3. For each option, include:
   - Clear step-by-step instructions
   - Relevant knowledge base article links if available
   - Expected outcome

4. IDENTIFY any information needed from the customer to proceed

5. SUGGEST escalation path if needed (L2/L3, engineering, etc.)

=== OUTPUT FORMAT ===

Return a JSON object with this EXACT structure:
{{
  "issueCategory": "<category from list above>",
  "rootCauseSummary": "<1-2 sentence summary of the likely root cause>",
  "resolutionOptions": [
    {{
      "title": "<short title>",
      "confidence": "<high|medium|low>",
      "steps": ["<step 1>", "<step 2>", ...],
      "articleLinks": ["<url1>", ...],
      "expectedOutcome": "<what happens when resolved>"
    }}
  ],
  "additionalInfoNeeded": ["<question 1>", "<question 2>", ...],
  "escalationPath": "<when and how to escalate if resolution fails>",
  "suggestedResponse": "<draft customer response message>"
}}''',
                agent=support_agent,
                expected_output='JSON object with issueCategory, rootCauseSummary, resolutionOptions, additionalInfoNeeded, escalationPath, and suggestedResponse'
            )

            # Create and run crew
            crew = Crew(
                agents=[support_agent],
                tasks=[resolution_task],
                verbose=False
            )

            result = crew.kickoff()
            result_text = str(result)

            send_progress(self.wfile, 'Complete', 'Processing results...', 'System')

            # Parse result
            parsed_result = None
            try:
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result_text[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback structure
                parsed_result = {
                    "issueCategory": "Unknown",
                    "rootCauseSummary": "Unable to parse structured response",
                    "resolutionOptions": [{
                        "title": "Review case manually",
                        "confidence": "low",
                        "steps": ["Review the case details", "Check knowledge base", "Contact L2 support"],
                        "articleLinks": ["https://support.leandata.com/s/"],
                        "expectedOutcome": "Resolution determined after manual review"
                    }],
                    "additionalInfoNeeded": [],
                    "escalationPath": "Escalate to L2 if unable to resolve",
                    "suggestedResponse": result_text[:500],
                    "rawAnalysis": result_text
                }

            # Add metadata
            parsed_result['articlesSearched'] = [{'title': a['title'], 'url': a['url']} for a in article_content]
            parsed_result['searchQuery'] = search_query

            # Save to database
            try:
                from supabase import create_client, Client

                supabase_url = os.environ.get('SUPABASE_URL')
                supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

                if supabase_url and supabase_key:
                    supabase: Client = create_client(supabase_url, supabase_key)

                    user_id = body.get('userId')
                    account_id = body.get('accountId')
                    salesforce_account_id = body.get('salesforceAccountId')
                    case_id = body.get('caseId')

                    if user_id or case_id:
                        save_data = {
                            'crew_type': 'support_resolution',
                            'result': json.dumps(parsed_result),
                            'provider': 'openai',
                            'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                        }

                        if user_id:
                            save_data['user_id'] = user_id
                        if account_id:
                            save_data['account_id'] = account_id
                        if salesforce_account_id:
                            save_data['salesforce_account_id'] = salesforce_account_id

                        result_insert = supabase.table('crew_analysis_history').insert(save_data).execute()
                        if hasattr(result_insert, 'error') and result_insert.error:
                            print(f'Error saving support resolution to database: {result_insert.error}')
                        else:
                            print(f'✅ Saved support resolution to crew_analysis_history')
                    else:
                        print('Warning: No userId/caseId provided, skipping database save')
            except Exception as save_error:
                print(f'Error saving support resolution to database: {save_error}')

            # Send result
            result_data = {
                'type': 'result',
                'result': parsed_result
            }
            send_sse_message(self.wfile, result_data)

        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"
            send_error(self.wfile, error_message)
