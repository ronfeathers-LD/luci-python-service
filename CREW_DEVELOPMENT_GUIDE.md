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

# Lazy imports for performance
_crewai = None

def _get_crewai():
    global _crewai
    if _crewai is None:
        from crewai import Crew, Agent, Task
        _crewai = {'Crew': Crew, 'Agent': Agent, 'Task': Task}
    return _crewai

def get_llm(openai_api_key=None):
    from langchain_openai import ChatOpenAI
    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    return ChatOpenAI(api_key=api_key, model='gpt-4o-mini', temperature=0.7)


# Crew metadata for auto-discovery
CREW_METADATA = {
    'crew_type': 'my_crew',           # Unique identifier
    'name': 'My Crew Analysis',       # Display name
    'description': 'Analyzes XYZ',    # Short description
    'category': 'analysis',           # Category for grouping
    'icon': 'chart-bar',              # Icon name (heroicons)
    'required_params': {              # Parameters this crew needs
        'accountId': 'optional',      # or 'required'
        'salesforceAccountId': 'optional',
    },
    'supports_streaming': True,
}


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            # Check for streaming
            enable_streaming = 'stream=true' in (self.path or '')

            if enable_streaming:
                start_sse_response(self)
                self._run_analysis_streaming(body)
            else:
                self._run_analysis(body)

        except Exception as e:
            self._send_error(str(e))

    def _run_analysis_streaming(self, body):
        try:
            send_progress(self.wfile, 'Starting', 'Initializing analysis...', 'System')

            # Get dependencies
            crewai = _get_crewai()
            Agent = crewai['Agent']
            Task = crewai['Task']
            Crew = crewai['Crew']
            llm = get_llm()

            # Create single optimized agent
            analyst = Agent(
                role='[Role Name]',
                goal='[Clear, specific goal]',
                backstory='[Brief backstory establishing expertise]',
                llm=llm,
                verbose=True
            )

            # Create single comprehensive task
            task = Task(
                description='''[Comprehensive task description with all context]

=== DATA ===
[Include all relevant data here]

=== REQUIREMENTS ===
[Specific requirements for output]

=== OUTPUT FORMAT ===
[Expected output structure]''',
                expected_output='[Clear description of expected output]',
                agent=analyst
            )

            send_progress(self.wfile, 'Analysis', 'Running AI analysis...', 'System')

            # Execute
            crew = Crew(agents=[analyst], tasks=[task], verbose=True)
            result = crew.kickoff()

            # Send result
            send_result(self.wfile, {'result': str(result)})

        except Exception as e:
            send_error(self.wfile, str(e))

    def _send_error(self, message):
        self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': message}).encode())
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
# One agent that does everything
analyst = Agent(
    role='Comprehensive Analyst',
    goal='Analyze data and provide recommendations',
    backstory='Expert analyst...',
    llm=llm
)

task = Task(
    description='Complete analysis with all requirements...',
    agent=analyst
)

crew = Crew(agents=[analyst], tasks=[task])
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

### 2. Use RAG Context When Available

```python
# Try RAG first (fast, uses embeddings)
from rag_helpers import get_relevant_context, get_analysis_query

rag_result = get_relevant_context(
    account_id=account_id,
    query=get_analysis_query('my_analysis_type'),
    match_count=15
)

if rag_result.get('context'):
    context = rag_result['context']  # Use RAG context
else:
    context = raw_data  # Fallback to raw data
```

### 3. Lazy Load Heavy Imports

```python
# Good: Lazy load
_crewai = None

def _get_crewai():
    global _crewai
    if _crewai is None:
        from crewai import Crew, Agent, Task
        _crewai = {'Crew': Crew, 'Agent': Agent, 'Task': Task}
    return _crewai

# Bad: Import at top level (slows cold start)
from crewai import Crew, Agent, Task
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

## Testing

1. Test the Python endpoint directly:
```bash
curl -X POST https://your-service.vercel.app/api/crew/my_crew?stream=true \
  -H "Content-Type: application/json" \
  -d '{"accountId": "uuid-here"}'
```

2. Test through Next.js:
```bash
curl -X POST https://your-app.vercel.app/api/crew/my_crew?stream=true \
  -H "Content-Type: application/json" \
  -H "Cookie: your-auth-cookie" \
  -d '{"userId": "user-uuid", "accountId": "account-uuid"}'
```

## Deployment

1. Commit Python crew file to `luci-python-service`
2. Push to trigger Vercel deployment
3. Run database migrations for crew registration
4. Assign crew to appropriate roles
5. Frontend automatically discovers new crew via `/api/crews/available`
