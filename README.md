# LUCI Python Service

Python CrewAI service for LUCI application. Provides AI-powered analysis for account management, project implementation, sales pipeline, and support operations using OpenAI with RAG (Retrieval-Augmented Generation) for enhanced context.

## Endpoints

### Account & Portfolio Analysis
- `/api/crew/account` - Account health analysis with risk assessment and relationship scoring
- `/api/crew/overview` - Portfolio-level analysis across all user accounts
- `/api/crew/sentiment` - Customer sentiment analysis from conversation transcriptions

### Implementation & Projects
- `/api/crew/implementation` - Project management effectiveness and execution analysis
- `/api/crew/project-sentiment` - Implementation project sentiment with PM coaching insights

### Sales
- `/api/crew/sales_pipeline` - Sales opportunity analysis with deal progression guidance

### Support
- `/api/crew/support_resolution` - Support case analysis with resolution suggestions
- `/api/crew/support_training` - Real-time personalized training suggestions for support agents
- `/api/crew/profile_builder` - Analyzes top performer patterns and builds reusable profiles (runs via cron)

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for CrewAI
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key

Optional:
- `OPENAI_MODEL_NAME` - Model to use (default: gpt-4o-mini)
- `DEBUG` - Enable full error tracebacks

## Tech Stack

- **AI Framework**: CrewAI with LangChain
- **LLM**: OpenAI (gpt-4o-mini default)
- **Database**: Supabase (PostgreSQL)
- **Streaming**: Server-Sent Events (SSE) for real-time progress
- **External APIs**: Avoma (meeting data), Mavenlink (project management)

## Deployment

Deployed on Vercel as Python 3.11 serverless functions with 5-minute max duration.

## Usage

This service is called by the main LUCI Next.js application via API proxy routes. All endpoints stream progress updates via SSE and return structured AI insights with data citations.




