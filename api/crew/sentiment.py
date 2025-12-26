"""
Vercel Python Serverless Function for Sentiment Analysis Crew
Analyzes customer sentiment from conversation transcriptions and Salesforce context

PERFORMANCE OPTIMIZATION: Heavy imports (crewai, langchain) are lazy-loaded
inside functions to reduce cold start time by ~1-2 seconds.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error, send_sse_message

# Import shared helpers (includes warning suppression and lazy loading)
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import save_analysis_to_database

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

            # Get lazy-loaded crewai classes
            crewai = get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Initialize LLM
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)
            
            # Prepare context for analysis
            transcription_text = transcription or '(No transcription available)'
            customer_identifier = body.get('customerIdentifier', 'Unknown Account')
            use_rag = body.get('useRag', True)  # Default to using RAG for performance

            # Get account ID for RAG lookup
            account_id = body.get('accountId')
            salesforce_account_id = body.get('salesforceAccountId')

            # Try to use RAG for context if enabled and account ID is available
            rag_context = None
            avoma_context = None
            context_source = 'traditional'  # Track where context came from

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
                        # Generate sentiment-specific query for embedding search
                        analysis_query = get_analysis_query('sentiment')
                        print(f'Calling get_relevant_context with account_id={rag_account_id}')
                        rag_result = get_relevant_context(
                            account_id=rag_account_id,
                            query=analysis_query,
                            match_count=15,  # Get relevant chunks for sentiment
                            match_threshold=0.4  # Lower threshold for broader coverage
                        )
                        print(f'RAG result: {len(rag_result.get("chunks", []))} chunks, context length: {len(rag_result.get("context", ""))}')
                        if rag_result.get('context'):
                            rag_context = rag_result['context']
                            context_source = 'rag'
                            data_type_counts = rag_result.get('data_type_counts', {})
                            print(f'RAG retrieved context for sentiment with {len(rag_result.get("chunks", []))} chunks: {data_type_counts}')
                            send_progress(self.wfile, 'RAG Context', f'Using {len(rag_context)} chars of relevant context', 'System')
                except Exception as rag_error:
                    print(f'RAG context retrieval failed for sentiment: {rag_error}')
                    rag_context = None

                # Step 2: If RAG failed or returned no context, try Avoma API (real-time fetch)
                if not rag_context:
                    try:
                        from avoma_helpers import get_avoma_context_for_account

                        send_progress(self.wfile, 'Avoma Fetch', 'Fetching meeting data from Avoma...', 'System')

                        avoma_result = get_avoma_context_for_account(
                            salesforce_account_id=salesforce_account_id,
                            customer_name=customer_identifier,
                            account_id=account_id,
                            max_meetings=5,
                            max_transcript_chars=10000
                        )

                        if avoma_result.get('success') and avoma_result.get('context'):
                            avoma_context = avoma_result['context']
                            context_source = 'avoma'
                            print(f'Avoma retrieved context for sentiment: {len(avoma_result.get("meetings", []))} meetings, {len(avoma_context)} chars')
                            send_progress(self.wfile, 'Avoma Context', f'Using {len(avoma_context)} chars from Avoma', 'System')
                        else:
                            print(f'Avoma fetch failed or returned no data: {avoma_result.get("error", "Unknown error")}')
                    except Exception as avoma_error:
                        print(f'Avoma context retrieval failed for sentiment: {avoma_error}')
                        avoma_context = None

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
            
            send_progress(self.wfile, 'Setup', 'Preparing sentiment analysis...', 'System')

            # Create step callback for real-time progress updates
            wfile_ref = self.wfile  # Capture for closure
            step_count = [0]  # Mutable container for step counter

            def agent_step_callback(step_output):
                """Send progress update after each agent step"""
                try:
                    step_count[0] += 1
                    step_str = str(step_output)[:150] if step_output else 'Processing...'
                    if 'Action:' in step_str:
                        step_str = step_str.split('Action:')[0].strip()[:100]
                    elif len(step_str) > 100:
                        step_str = step_str[:100] + '...'
                    send_progress(wfile_ref, f'Step {step_count[0]}', f'Analyzing: {step_str}', 'AI Analysis')
                except Exception as e:
                    print(f'Step callback error: {e}')

            # OPTIMIZED: Single agent with minimal iterations to reduce LLM calls
            # - max_iter=1: Single pass, no reasoning loop
            # - allow_delegation=False: No delegation overhead
            # - verbose=False: Reduces internal processing
            sentiment_analyst = Agent(
                role='Sentiment Analyst',
                goal='Analyze sentiment and return JSON with score, summary, and analysis',
                backstory='Expert sentiment analyst. Respond directly with the requested JSON output.',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )
            
            # OPTIMIZED: Single comprehensive task instead of 3 separate tasks
            # This reduces LLM calls from 3 to 1, significantly improving performance
            send_progress(self.wfile, 'Preparing', 'Setting up analysis task...', 'System')

            # Build context based on source: RAG > Avoma > traditional
            if rag_context:
                data_context = f'''=== RELEVANT CONTEXT (from vector database) ===
{rag_context}

=== ACCOUNT INFO ===
{context_string}'''
            elif avoma_context:
                data_context = f'''=== MEETING DATA (from Avoma) ===
{avoma_context}

=== ACCOUNT INFO ===
{context_string}'''
            else:
                data_context = f'''=== CONVERSATION DATA ===
{transcription_text}

=== ACCOUNT INFO ===
{context_string}'''

            # Single comprehensive task
            analysis_task = Task(
                description=f'''Analyze customer sentiment and produce a comprehensive assessment.

=== ACCOUNT PROFILE ===
Account: {customer_identifier}
Tier: {salesforce_context.get('account_tier', 'Unknown')}
Contract Value: {salesforce_context.get('contract_value', 'Unknown')}
Industry: {salesforce_context.get('industry', 'Unknown')}
Account Manager: {salesforce_context.get('account_manager', 'Unknown')}
Avoma Calls: {salesforce_context.get('total_avoma_calls', 0)} total, {salesforce_context.get('ready_avoma_calls', 0)} with transcripts

{data_context}

=== ANALYSIS REQUIREMENTS ===

Apply RECENCY WEIGHTING throughout:
- Recent data (0-30 days): 80-90% weight - PRIMARY indicators
- Historical data (90+ days): 5-10% weight - context only

Watch for RED FLAGS:
- C-Level involvement in support cases = MAJOR escalation concern
- Sr. Level involvement in support cases = significant frustration
- High priority/unresolved cases = active issues

Analyze these factors:
1. Conversation sentiment: tone, language patterns, satisfaction signals
2. Support patterns: case trends, priorities, resolution quality
3. Relationship trajectory: improving, declining, or stable
4. Risk factors and opportunities

=== OUTPUT FORMAT ===

Return a JSON object with this EXACT structure:
{{
  "score": <integer 1-10, where 1=very negative, 5=neutral, 10=very positive>,
  "summary": "<executive summary, 150 words max - key takeaways for executives>",
  "comprehensiveAnalysis": "<detailed analysis, 400-600 words - breakdown of factors, evidence, recommendations>"
}}''',
                agent=sentiment_analyst,
                expected_output='JSON object with score (1-10), summary (150 words), and comprehensiveAnalysis (400-600 words)'
            )

            # Create crew with single agent and task - verbose=False for performance
            crew = Crew(
                agents=[sentiment_analyst],
                tasks=[analysis_task],
                verbose=False
            )

            send_progress(self.wfile, 'Analysis', 'Running AI analysis...', 'AI Crew')

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

            # Save to database using shared helper
            save_analysis_to_database(
                crew_type='sentiment',
                result=json.dumps(final_result),
                user_id=body.get('userId'),
                account_id=body.get('accountId'),
                salesforce_account_id=body.get('salesforceAccountId')
            )

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

