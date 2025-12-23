# LUCI Python Service

Python CrewAI service for LUCI application.

## Endpoints

- `/api/crew/account` - Account health analysis
- `/api/crew/implementation` - Implementation project analysis
- `/api/crew/overview` - Portfolio overview analysis

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for CrewAI
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key

Optional:
- `OPENAI_MODEL_NAME` - Model to use (default: gpt-4o-mini)

## Deployment

This service is deployed on Vercel as Python serverless functions.

## Usage

This service is called by the main LUCI Next.js application via API proxy routes.




