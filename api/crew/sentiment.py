"""
Vercel Python Serverless Function for Sentiment Analysis Crew
Analyzes customer sentiment from conversation transcriptions and Salesforce context

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time by ~1-2 seconds.
"""

import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Optional

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

class handler(BaseHTTPRequestHandler):
    """Main handler for Sentiment Analysis Crew requests - Vercel format"""
    
    def do_GET(self):
        """Handle GET requests - only for health checks"""
        path = self.path.split('?')[0]
        if path != '/api/crew/sentiment':
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
            'service': 'CrewAI Sentiment Analysis',
            'provider': 'openai'
        }).encode('utf-8'))
    
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
            
            # Validate required parameters
            transcription = body.get('transcription', '')
            salesforce_context = body.get('salesforceContext', {})
            
            if transcription is None:
                send_error(self.wfile, 'Missing required parameter: transcription')
                return
            
            if not salesforce_context:
                send_error(self.wfile, 'Missing required parameter: salesforceContext')
                return
            
            send_progress(self.wfile, 'Initialization', 'Initializing AI model...', 'System')

            # Lazy load crewai classes (reduces cold start time)
            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Initialize LLM
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)
            
            # Prepare context for analysis
            transcription_text = transcription or '(No transcription available)'
            customer_identifier = body.get('customerIdentifier', 'Unknown Account')
            
            # Build context string with recency indicators
            context_parts = []
            context_parts.append(f"Account: {customer_identifier}")
            context_parts.append(f"Account Tier: {salesforce_context.get('account_tier', 'Unknown')}")
            context_parts.append(f"Contract Value: {salesforce_context.get('contract_value', 'Unknown')}")
            context_parts.append(f"Industry: {salesforce_context.get('industry', 'Unknown')}")
            context_parts.append(f"Account Manager: {salesforce_context.get('account_manager', 'Unknown')}")
            context_parts.append(f"Total Cases: {salesforce_context.get('total_cases_count', 0)}")
            context_parts.append(f"Total Avoma Calls: {salesforce_context.get('total_avoma_calls', 0)}")
            context_parts.append(f"Ready Transcripts: {salesforce_context.get('ready_avoma_calls', 0)}")
            
            # Add contact level involvement (critical indicator)
            contact_levels = salesforce_context.get('contact_levels', {})
            if contact_levels:
                context_parts.append(f"\n⚠️ CONTACT LEVEL INVOLVEMENT IN CASES (CRITICAL):")
                c_level_in_cases = contact_levels.get('c_level_in_cases', 0)
                sr_level_in_cases = contact_levels.get('sr_level_in_cases', 0)
                c_level_count = contact_levels.get('c_level_count', 0)
                sr_level_count = contact_levels.get('sr_level_count', 0)
                other_in_cases = contact_levels.get('other_in_cases', 0)
                other_count = contact_levels.get('other_count', 0)
                
                if c_level_in_cases > 0:
                    context_parts.append(f"  - C-Level contacts in cases: {c_level_in_cases} of {c_level_count} total (MAJOR RED FLAG - indicates escalated problems)")
                if sr_level_in_cases > 0:
                    context_parts.append(f"  - Sr. Level contacts in cases: {sr_level_in_cases} of {sr_level_count} total (SIGNIFICANT CONCERN)")
                if other_in_cases > 0:
                    context_parts.append(f"  - Other contacts in cases: {other_in_cases} of {other_count} total")
            
            # Add recent cases with recency labels
            recent_tickets = salesforce_context.get('recent_tickets', [])
            if recent_tickets:
                context_parts.append(f"\nRECENT SUPPORT CASES (with recency weighting):")
                for ticket in recent_tickets[:20]:  # Limit to 20 most recent
                    recency_label = ticket.get('recency_label', 'HISTORICAL')
                    context_parts.append(f"  [{recency_label}] {ticket.get('subject', 'No subject')} - Status: {ticket.get('status')}, Priority: {ticket.get('priority')}")
                    if ticket.get('description'):
                        desc = ticket.get('description', '')[:200]  # Truncate long descriptions
                        context_parts.append(f"    Description: {desc}...")
            
            # Add LinkedIn data if available
            linkedin_data = salesforce_context.get('linkedin_data', {})
            if linkedin_data and linkedin_data.get('contacts'):
                context_parts.append(f"\nCONTACT INTELLIGENCE:")
                context_parts.append(f"  Total contacts: {linkedin_data.get('total_contacts', 0)}")
                context_parts.append(f"  Contacts with enriched data: {linkedin_data.get('contacts_with_enriched_data', 0)}")
                if linkedin_data.get('contact_level_counts'):
                    counts = linkedin_data['contact_level_counts']
                    context_parts.append(f"  Contact breakdown: C-Level: {counts.get('C-Level', 0)}, Sr. Level: {counts.get('Sr. Level', 0)}, Other: {counts.get('Other', 0)}")
            
            context_string = "\n".join(context_parts)
            
            send_progress(self.wfile, 'Setup', 'Preparing sentiment analysis agents...', 'System')

            # Create step callback for real-time progress updates
            wfile_ref = self.wfile  # Capture for closure
            step_count = [0]  # Mutable container for step counter

            def agent_step_callback(step_output):
                """Send progress update after each agent step"""
                try:
                    step_count[0] += 1
                    # Extract meaningful info from step output
                    step_str = str(step_output)[:150] if step_output else 'Processing...'
                    # Clean up the output for display
                    if 'Action:' in step_str:
                        step_str = step_str.split('Action:')[0].strip()[:100]
                    elif len(step_str) > 100:
                        step_str = step_str[:100] + '...'
                    send_progress(wfile_ref, f'Step {step_count[0]}', f'Agent working: {step_str}', 'AI Analysis')
                except Exception as e:
                    print(f'Step callback error: {e}')

            # Create agents with step_callback for progress monitoring
            conversation_analyst = Agent(
                role='Conversation Sentiment Analyst',
                goal='Extract emotional tone, language patterns, and satisfaction signals from customer conversations, applying recency weighting where recent conversations (0-30 days) are PRIMARY indicators (80-90% weight) and historical conversations (90+ days) provide context only (5-10% weight)',
                backstory='''You are an expert at analyzing B2B customer conversations. You understand that:
- Recent conversations (last 30 days) are the PRIMARY indicators of current sentiment (80-90% weight)
- Historical conversations (90+ days old) provide context only (5-10% weight)
- Language patterns, emotional indicators, urgency, and frustration levels reveal true sentiment
- Resolution quality and how concerns were addressed are critical indicators
- Final outcome and customer satisfaction level should be assessed from most recent meetings''',
                llm=llm,
                verbose=True,
                step_callback=agent_step_callback
            )
            
            support_analyst = Agent(
                role='Support Context Analyst',
                goal='Evaluate support case patterns, priorities, contact level involvement, and apply recency weighting where recent cases (last 30 days) are PRIMARY indicators (80-90% weight) and historical cases (90+ days) provide trend context only (5-10% weight)',
                backstory='''You specialize in understanding how support case patterns reveal underlying customer issues. You know that:
- Recent cases (last 30 days) are PRIMARY indicators of current sentiment (80-90% weight)
- Historical cases (90+ days) are for trend context only (5-10% weight)
- C-Level or Sr. Level involvement in cases is a MAJOR RED FLAG indicating escalated problems
- Case priorities, statuses, and descriptions reveal customer frustration levels
- Resolution timelines and patterns indicate relationship health''',
                llm=llm,
                verbose=True,
                step_callback=agent_step_callback
            )

            synthesizer = Agent(
                role='Relationship Health Synthesizer',
                goal='Synthesize all analysis into comprehensive sentiment assessment with executive summary (150 words max) and detailed analysis (500-800 words), providing sentiment score (1-10) with detailed reasoning',
                backstory='''You are an expert at synthesizing complex customer relationship data into clear, actionable insights. You understand:
- Account tiers, contract values, and industry context affect expectations
- Engagement metrics (Avoma calls, transcripts) indicate relationship strength
- Relationship trajectory (improving, declining, stable) must be assessed with evidence
- Risk factors and opportunities must be clearly identified
- Recommendations must be specific and actionable
- Executive summary must be concise for C-level executives
- Comprehensive analysis must be detailed for CSMs and Account Managers''',
                llm=llm,
                verbose=True,
                step_callback=agent_step_callback
            )
            
            # Create tasks
            send_progress(self.wfile, 'Preparing', 'Setting up conversation analysis task...', 'System')
            
            task1 = Task(
                description=f'''Analyze the conversation transcription for sentiment indicators.

CONVERSATION TRANSCRIPTION:
{transcription_text}

CRITICAL RECENCY WEIGHTING RULES:
- If transcription is marked with recency labels ([MOST_RECENT], [RECENT], [HISTORICAL]), apply weighting:
  * [MOST_RECENT] or [RECENT] (0-30 days): 80-90% weight - PRIMARY indicators
  * [HISTORICAL] (90+ days): 5-10% weight - context only
- Focus on most recent conversations for current sentiment assessment
- Historical conversations provide trend context but should NOT drive the score

Analyze:
1. Initial customer tone and emotional state (focus on most recent)
2. Language patterns (positive/negative indicators, urgency, frustration) - recent patterns matter most
3. Resolution quality and how concerns were addressed - recent resolutions are most relevant
4. Final outcome and customer satisfaction level - prioritize most recent meetings

Provide detailed analysis of conversation sentiment with emphasis on recent interactions.''',
                agent=conversation_analyst,
                expected_output='Detailed analysis of conversation sentiment with recency weighting applied, including tone, language patterns, resolution quality, and satisfaction indicators'
            )
            
            send_progress(self.wfile, 'Preparing', 'Setting up support case analysis task...', 'System')
            
            task2 = Task(
                description=f'''Analyze support case context and contact involvement.

SALESFORCE ACCOUNT CONTEXT:
{context_string}

CRITICAL RECENCY WEIGHTING RULES:
- Cases marked [MOST_RECENT] or [RECENT] (0-30 days): 80-90% weight - PRIMARY indicators
- Cases marked [HISTORICAL] (90+ days): 5-10% weight - trend context only
- Focus on cases from last 30 days as they indicate CURRENT issues

CRITICAL CONTACT LEVEL ANALYSIS:
- C-Level involvement in cases = MAJOR RED FLAG (indicates escalated problems)
- Sr. Level involvement in cases = SIGNIFICANT CONCERN (indicates frustration)
- Weight case involvement by contact level heavily

Analyze:
1. Support case patterns and trends (recent cases are PRIMARY indicators)
2. Case priorities and statuses (high priority or unresolved = issues)
3. Case descriptions for customer feedback and issue details
4. Contact level involvement (C-Level/Sr. Level = major concern)
5. Resolution timelines and patterns

Provide detailed analysis of support case context with recency weighting and contact level assessment.''',
                agent=support_analyst,
                expected_output='Detailed analysis of support case patterns, contact involvement, and recency-weighted case assessment'
            )
            
            send_progress(self.wfile, 'Preparing', 'Setting up synthesis task...', 'System')
            
            task3 = Task(
                description=f'''Synthesize all analysis into a comprehensive sentiment assessment.

ACCOUNT PROFILE:
- Account: {customer_identifier}
- Tier: {salesforce_context.get('account_tier', 'Unknown')}
- Contract Value: {salesforce_context.get('contract_value', 'Unknown')}
- Industry: {salesforce_context.get('industry', 'Unknown')}
- Account Manager: {salesforce_context.get('account_manager', 'Unknown')}

ENGAGEMENT METRICS:
- Total Avoma Calls: {salesforce_context.get('total_avoma_calls', 0)}
- Ready Transcripts: {salesforce_context.get('ready_avoma_calls', 0)}

You will receive:
1. Conversation sentiment analysis from Conversation Sentiment Analyst
2. Support case context analysis from Support Context Analyst

Your task:
1. Synthesize both analyses into a single comprehensive assessment
2. Consider account profile (tier, contract value, industry) - higher value accounts may have different expectations
3. Evaluate relationship trajectory (improving, declining, stable) with evidence
4. Identify risk factors and opportunities
5. Generate TWO outputs:

   A. EXECUTIVE SUMMARY (150 words max):
      - Overall sentiment score (1-10) and key takeaway
      - Top 2-3 critical factors
      - Immediate action required (if any)
      - Relationship health status
      - Suitable for C-level executives

   B. COMPREHENSIVE ANALYSIS (500-800 words):
      - Detailed breakdown of all factors influencing the score
      - Specific concerns or positive signals with examples
      - Relationship trajectory analysis with evidence
      - Contact level involvement analysis (C-Level/Sr. Level in cases is major red flag)
      - Support case patterns and implications
      - Engagement metrics and their meaning
      - Risk factors and opportunities
      - Detailed actionable recommendations
      - Account-specific context and nuances
      - Comparison to account tier expectations

CRITICAL: Apply recency weighting throughout:
- Recent data (0-30 days): 80-90% weight - PRIMARY indicators
- Historical data (90+ days): 5-10% weight - context only
- The sentiment score must reflect CURRENT customer sentiment based primarily on recent interactions

Return your response as a JSON object with this exact structure:
{{
  "score": <integer 1-10>,
  "summary": "<executive summary, 150 words max>",
  "comprehensiveAnalysis": "<comprehensive analysis, 500-800 words>"
}}''',
                agent=synthesizer,
                expected_output='JSON object with score (1-10), summary (150 words max), and comprehensiveAnalysis (500-800 words) fields'
            )
            
            # Create crew
            crew = Crew(
                agents=[conversation_analyst, support_analyst, synthesizer],
                tasks=[task1, task2, task3],
                verbose=True
            )
            
            send_progress(self.wfile, 'Analysis', 'Running AI analysis (this may take 1-3 minutes)...', 'AI Crew')

            # Execute crew - this is the main analysis step and takes the longest
            result = crew.kickoff()
            result_text = str(result)

            send_progress(self.wfile, 'Complete', 'Processing and saving results...', 'System')
            
            # Parse result - CrewAI returns a string, we need to extract JSON
            parsed_result = None
            
            # Try to find JSON in the result
            try:
                # Look for JSON object in the result
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result_text[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                else:
                    # If no JSON found, try parsing the whole result
                    parsed_result = json.loads(result_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response from the text
                # This is a fallback - ideally the crew should return proper JSON
                parsed_result = {
                    "score": 5,  # Default neutral score
                    "summary": result_text[:500] if len(result_text) > 500 else result_text,
                    "comprehensiveAnalysis": result_text
                }
            
            # Validate and structure the response
            score = parsed_result.get('score', 5)
            if not isinstance(score, int) or score < 1 or score > 10:
                # Try to extract score from text if not valid
                score = 5  # Default to neutral
            
            summary = parsed_result.get('summary', parsed_result.get('executiveSummary', ''))
            comprehensive_analysis = parsed_result.get('comprehensiveAnalysis', parsed_result.get('comprehensive_analysis', parsed_result.get('analysis', result_text)))
            
            # Ensure summary is within word limit
            if summary:
                words = summary.split()
                if len(words) > 150:
                    summary = ' '.join(words[:150]) + '...'
            
            # Format final result to match expected structure
            # The Next.js route and UI expect: {score, summary, comprehensiveAnalysis}
            final_result = {
                "score": score,
                "summary": summary or "Analysis completed. See comprehensive analysis for details.",
                "comprehensiveAnalysis": comprehensive_analysis or result_text
            }

            # Save to database (same pattern as account.py)
            try:
                from supabase import create_client, Client

                supabase_url = os.environ.get('SUPABASE_URL')
                supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

                if supabase_url and supabase_key:
                    supabase: Client = create_client(supabase_url, supabase_key)

                    # Get IDs from request body
                    user_id = body.get('userId')
                    account_id = body.get('accountId')
                    salesforce_account_id = body.get('salesforceAccountId')

                    if user_id or account_id or salesforce_account_id:
                        # Resolve account_id from salesforce_account_id if needed
                        resolved_account_id = account_id
                        if not resolved_account_id and salesforce_account_id:
                            try:
                                acc_result = supabase.table('accounts').select('id').eq('salesforce_id', salesforce_account_id).limit(1).execute()
                                if acc_result.data and len(acc_result.data) > 0:
                                    resolved_account_id = acc_result.data[0].get('id')
                            except Exception as acc_err:
                                print(f'Warning: Could not resolve account_id from salesforce_account_id: {acc_err}')

                        # Save to crew_analysis_history
                        save_data = {
                            'crew_type': 'sentiment',
                            'result': json.dumps(final_result),
                            'provider': 'openai',
                            'model': os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                        }

                        if user_id:
                            save_data['user_id'] = user_id
                        if resolved_account_id:
                            save_data['account_id'] = resolved_account_id
                        if salesforce_account_id:
                            save_data['salesforce_account_id'] = salesforce_account_id

                        result_insert = supabase.table('crew_analysis_history').insert(save_data).execute()
                        if hasattr(result_insert, 'error') and result_insert.error:
                            print(f'Error saving sentiment analysis to database: {result_insert.error}')
                        else:
                            print(f'✅ Saved sentiment analysis to crew_analysis_history')
                    else:
                        print('Warning: No userId/accountId/salesforceAccountId provided, skipping database save')
                else:
                    print('Warning: Supabase URL or Key not set, skipping database save')
            except Exception as save_error:
                # Don't fail the request if save fails - just log it
                print(f'Error saving sentiment analysis to database: {save_error}')

            # Send final result via SSE
            # The Next.js route does: finalResult = data.result || data
            # So we send: {type: 'result', result: {score, summary, comprehensiveAnalysis}}
            result_data = {
                'type': 'result',
                'result': final_result  # This becomes finalResult in Next.js, which has score/summary/comprehensiveAnalysis
            }

            # Send via SSE using the helper
            send_sse_message(self.wfile, result_data)
            
        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"
            
            # Send error via SSE
            send_error(self.wfile, error_message)

