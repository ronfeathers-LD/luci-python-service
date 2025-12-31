# Crew Development Guide

## Overview

This guide covers how to create, register, and deploy new CrewAI agents in the LUCI system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LUCI Frontend                            │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ CrewAnalysis    │    │ /api/crews/     │                     │
│  │ Component       │───>│ available       │ (discovers crews)   │
│  └─────────────────┘    └─────────────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ /api/crew/      │    │ Supabase DB     │                     │
│  │ [crewType]      │    │ - crews table   │                     │
│  └─────────────────┘    │ - crew_role_    │                     │
│           │             │   assignments   │                     │
└───────────│─────────────└─────────────────┘─────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python Crew Service                           │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ /api/crew/      │    │ crew_registry   │                     │
│  │ [crewType].py   │<───│ (auto-discover) │                     │
│  └─────────────────┘    └─────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ CrewAI Agents   │───>│ OpenAI API      │                     │
│  └─────────────────┘    └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## Creating a New Crew

### Step 1: Create the Python Crew File

Create a new file in `/api/crew/[crew_name].py`:

```python
"""
Vercel Python Serverless Function for [Crew Name] Analysis
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sse_helpers import start_sse_response, send_progress, send_result, send_error

# Import shared helpers (includes warning suppression and lazy loading)
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import fetch_crew_config, save_analysis_to_database, build_system_prompt


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'service': 'CrewAI [Crew Name] Analysis',
            'provider': 'openai'
        }).encode('utf-8'))

    def do_POST(self):
        # Always use SSE streaming
        start_sse_response(self)

        try:
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length > 0 else {}

            send_progress(self.wfile, 'Initialization', 'Initializing AI model...', 'System')

            # Lazy load crewai classes (reduces cold start time)
            crewai = get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']

            # Get LLM instance
            openai_api_key = body.get('openaiApiKey') or os.environ.get('OPENAI_API_KEY')
            llm = get_llm(openai_api_key=openai_api_key)

            send_progress(self.wfile, 'Configuration', 'Loading analysis configuration...', 'System')

            # Fetch crew configuration from database
            crew_config = fetch_crew_config('my_crew')  # Use your crew_type here
            full_system_prompt = build_system_prompt(crew_config)

            # Fetch your data here...
            send_progress(self.wfile, 'Data Fetching', 'Fetching data...', 'System')

            # Build context for analysis
            context = "[Build your context string here]"

            send_progress(self.wfile, 'Setup', 'Preparing analysis...', 'System')

            # OPTIMIZED: Single agent with minimal iterations
            # - max_iter=1: Single pass, no reasoning loop
            # - allow_delegation=False: No delegation overhead
            # - verbose=False: Reduces internal processing
            analyst = Agent(
                role='[Role Name]',
                goal='[Clear, specific goal]',
                backstory='[Brief backstory establishing expertise]',
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=1
            )

            # Build task description
            task_description = f'''[Comprehensive task description]

{context}

=== REQUIREMENTS ===
[Specific requirements for output]

=== OUTPUT FORMAT ===
[Expected output structure]'''

            if full_system_prompt:
                task_description = f"{full_system_prompt}\n\n{task_description}"

            task = Task(
                description=task_description,
                expected_output='[Clear description of expected output]',
                agent=analyst
            )

            send_progress(self.wfile, 'Analysis', 'Running AI analysis...', 'Analyst')

            # Execute with verbose=False for performance
            crew = Crew(agents=[analyst], tasks=[task], verbose=False)
            result = crew.kickoff()
            result_text = str(result)

            # Save to database
            save_analysis_to_database(
                crew_type='my_crew',
                result=result_text,
                user_id=body.get('userId'),
                account_id=body.get('accountId'),
                salesforce_account_id=body.get('salesforceAccountId')
            )

            # Send final result
            model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')
            send_result(self.wfile, result_text, provider='openai', model=model_name)

        except Exception as e:
            import traceback
            error_message = str(e)
            if os.environ.get('DEBUG'):
                error_message = f"{error_message}\n{traceback.format_exc()}"
            send_error(self.wfile, error_message)
```

### Step 2: Register in Database

Add the crew to the `crews` table:

```sql
INSERT INTO crews (
    crew_type,
    name,
    description,
    endpoint,
    category,
    icon,
    required_params,
    enabled
) VALUES (
    'my_crew',
    'My Crew Analysis',
    'Analyzes XYZ for accounts',
    '/api/crew/my_crew',
    'analysis',
    'chart-bar',
    '{"accountId": "optional", "salesforceAccountId": "optional"}',
    true
);
```

### Step 3: Assign to Roles

Enable the crew for specific roles:

```sql
INSERT INTO crew_role_assignments (role, crew_id, enabled)
SELECT 'csm', id, true FROM crews WHERE crew_type = 'my_crew';

INSERT INTO crew_role_assignments (role, crew_id, enabled)
SELECT 'admin', id, true FROM crews WHERE crew_type = 'my_crew';
```

### Step 4: Create Next.js Route (Optional)

If you need custom logic, create `/src/app/api/crew/my_crew/route.js`:

```javascript
import { NextRequest, NextResponse } from 'next/server';
import { handlePreflight, sendErrorResponse, log, logError } from '../../../../lib/next-api-helpers';
import { validateUserId } from '../../../../lib/auth-helpers';

export async function OPTIONS() {
  return handlePreflight(new NextRequest('http://localhost', { method: 'OPTIONS' }));
}

export async function POST(request) {
  const preflight = await handlePreflight(request);
  if (preflight) return preflight;

  try {
    const body = await request.json();
    const { userId } = body;

    const authValidation = await validateUserId(request, userId);
    if (!authValidation.valid) {
      return sendErrorResponse(new Error('Authentication required'), 401);
    }

    const pythonServiceUrl = process.env.CREWAI_PYTHON_SERVICE_URL;
    const response = await fetch(`${pythonServiceUrl}/api/crew/my_crew?stream=true`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        userId: authValidation.userId,
        ...body
      }),
    });

    // Stream response back to client
    return new NextResponse(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error) {
    logError('Error in /api/crew/my_crew:', error);
    return sendErrorResponse(error, 500);
  }
}
```

## Best Practices

### 1. Single Agent, Single Task (SAST) Pattern

**DO:**
```python
# One agent that does everything with optimized settings
analyst = Agent(
    role='Comprehensive Analyst',
    goal='Analyze data and provide recommendations',
    backstory='Expert analyst...',
    llm=llm,
    verbose=False,           # Reduces internal processing
    allow_delegation=False,  # No delegation overhead
    max_iter=1               # Single pass, no reasoning loop
)

task = Task(
    description='Complete analysis with all requirements...',
    agent=analyst
)

crew = Crew(agents=[analyst], tasks=[task], verbose=False)
```

**DON'T:**
```python
# Multiple agents that each make LLM calls
analyst = Agent(role='Analyst', ...)
strategist = Agent(role='Strategist', ...)
synthesizer = Agent(role='Synthesizer', ...)

# This makes 3 LLM calls instead of 1!
crew = Crew(agents=[analyst, strategist, synthesizer], tasks=[t1, t2, t3])
```

### 2. Use RAG Context with 3-Tier Fallback

The recommended pattern is: RAG (fastest) → Avoma (medium) → Raw data (slowest)

```python
from rag_helpers import get_relevant_context, get_analysis_query
from context_helpers import resolve_account_uuid

# Resolve account UUID for RAG search
rag_account_id = resolve_account_uuid(account_id, salesforce_account_id)

context_source = 'raw_data'
rag_context = None

if rag_account_id:
    # Step 1: Try RAG (fastest - uses pre-computed embeddings)
    rag_result = get_relevant_context(
        account_id=rag_account_id,
        query=get_analysis_query('my_analysis_type'),
        match_count=20,
        match_threshold=0.4
    )
    if rag_result.get('context'):
        rag_context = rag_result['context']
        context_source = 'rag'

if not rag_context:
    # Step 2: Try Avoma API (real-time fetch)
    from avoma_helpers import get_avoma_context_for_account
    avoma_result = get_avoma_context_for_account(
        salesforce_account_id=salesforce_account_id,
        customer_name=account_name
    )
    if avoma_result.get('context'):
        rag_context = avoma_result['context']
        context_source = 'avoma'

# Step 3: Fall back to raw data if both failed
if not rag_context:
    rag_context = build_raw_context(account_data)
```

### 3. Use Shared Helper Modules

**DO:** Use the shared helpers that handle lazy loading automatically:
```python
from crew.llm_helpers import get_llm, get_crewai
from crew.database_helpers import fetch_crew_config, save_analysis_to_database, build_system_prompt

# These are already lazy-loaded and cached
crewai = get_crewai()
llm = get_llm()
```

**DON'T:** Define your own lazy loading:
```python
# Don't do this - use llm_helpers instead
_crewai = None
def _get_crewai():
    global _crewai
    if _crewai is None:
        from crewai import Crew, Agent, Task
        _crewai = {'Crew': Crew, 'Agent': Agent, 'Task': Task}
    return _crewai
```

### 4. Clear Output Formats

Always specify exact output format in the task:

```python
task = Task(
    description='''...

=== OUTPUT FORMAT ===

Return a JSON object with this EXACT structure:
{
  "score": <integer 1-10>,
  "summary": "<150 words max>",
  "analysis": "<detailed analysis>"
}''',
    expected_output='JSON object with score, summary, and analysis fields'
)
```

### 5. Progress Updates

Keep users informed during long-running operations:

```python
send_progress(self.wfile, 'Stage', 'What is happening...', 'Component')

# Examples:
send_progress(self.wfile, 'RAG Search', 'Retrieving context...', 'System')
send_progress(self.wfile, 'Analysis', 'Running AI analysis...', 'Agent')
send_progress(self.wfile, 'Complete', 'Saving results...', 'System')
```

## Crew Categories

- `analysis` - Data analysis crews (account health, sentiment)
- `strategy` - Strategic planning crews
- `competitive` - Competitive intelligence
- `implementation` - Project/implementation focused
- `portfolio` - Multi-account/portfolio views
- `sales` - Sales pipeline and opportunity analysis
- `support` - Support case resolution and agent training

## Available Helper Modules

### `crew/llm_helpers.py` - LLM Management
- `get_crewai()` - Returns dict with lazy-loaded Crew, Agent, Task classes
- `get_llm(openai_api_key, model, temperature)` - Returns configured ChatOpenAI instance
- `create_agent(role, goal, backstory, llm, ...)` - Factory for creating agents
- `create_task(description, expected_output, agent, ...)` - Factory for creating tasks
- `create_crew(agents, tasks, ...)` - Factory for creating crews
- `summarize_text(text, llm, max_length)` - Summarize long text using LLM

### `crew/database_helpers.py` - Supabase Operations
- `get_supabase_client()` - Returns cached Supabase client
- `fetch_crew_config(crew_type)` - Load crew config from `crews` table
- `save_analysis_to_database(crew_type, result, ...)` - Save to `crew_analysis_history`
- `build_system_prompt(crew_config)` - Combine system prompt parts from config

### `crew/context_helpers.py` - Context Resolution
- `resolve_account_uuid(account_id, salesforce_account_id)` - Convert IDs to UUID format

### `crew/mavenlink_helpers.py` - Project Management Analysis
- `calculate_task_metrics(tasks)` - Calculate task completion metrics
- `identify_pm_workflow_issues(tasks, time_entries)` - Detect PM process problems
- `build_pm_workflow_context(project, tasks, time_entries)` - Format Mavenlink data

### `sse_helpers.py` - Server-Sent Events
- `start_sse_response(handler)` - Initialize SSE headers
- `send_progress(wfile, step, message, agent)` - Send progress update
- `send_result(wfile, result, provider, model)` - Send final result
- `send_error(wfile, message)` - Send error message

### `rag_helpers.py` - RAG/Vector Search
- `get_relevant_context(account_id, query, match_count, match_threshold)` - Vector search
- `get_analysis_query(analysis_type)` - Get analysis-specific embedding query
- `get_openai_embedding(text)` - Generate embeddings

### `avoma_helpers.py` - Avoma Meeting Data
- `get_avoma_context_for_account(salesforce_account_id, customer_name, ...)` - Fetch meeting context
- `list_meetings(from_date, to_date)` - List meetings from Avoma API

## Testing

1. Test the Python endpoint directly (all endpoints use SSE streaming):
```bash
curl -X POST https://your-service.vercel.app/api/crew/my_crew \
  -H "Content-Type: application/json" \
  -d '{"accountId": "uuid-here"}'
```

2. Test through Next.js:
```bash
curl -X POST https://your-app.vercel.app/api/crew/my_crew \
  -H "Content-Type: application/json" \
  -H "Cookie: your-auth-cookie" \
  -d '{"userId": "user-uuid", "accountId": "account-uuid"}'
```

3. Health check:
```bash
curl https://your-service.vercel.app/api/crew/my_crew
```

## Deployment

1. Commit Python crew file to `luci-python-service`
2. Push to trigger Vercel deployment
3. Run database migrations for crew registration
4. Assign crew to appropriate roles
5. Frontend automatically discovers new crew via `/api/crews/available`
