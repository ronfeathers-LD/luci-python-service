"""
Shared database helper functions for crew modules.
Provides centralized Supabase operations to reduce code duplication.
"""

import os
import json
from typing import Optional, Dict, Any

# Cached Supabase client
_supabase_client = None


def get_supabase_client():
    """
    Get or create a cached Supabase client.

    Returns:
        Supabase Client instance or None if credentials not configured
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    from supabase import create_client, Client

    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_key:
        print('Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set')
        return None

    _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client


def fetch_crew_config(crew_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch crew configuration from Supabase database.

    Args:
        crew_type: The crew_type identifier (e.g., 'account', 'implementation', 'overview')

    Returns:
        Dictionary with crew configuration or None if not found
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return None

        # Fetch crew config from database
        result = supabase.table('crews').select('*').eq('crew_type', crew_type).eq('enabled', True).limit(1).execute()

        if not result.data or len(result.data) == 0:
            print(f'Warning: Crew config not found for crew_type: {crew_type}')
            return None

        crew = result.data[0]

        # Parse JSONB fields
        config = {
            'name': crew.get('name'),
            'description': crew.get('description'),
            'system_prompt': crew.get('system_prompt'),
            'evaluation_criteria': crew.get('evaluation_criteria'),
            'scoring_rubric': crew.get('scoring_rubric'),
            'output_schema': crew.get('output_schema'),
            'agent_configs': crew.get('agent_configs') or [],
            'task_configs': crew.get('task_configs') or [],
        }

        return config

    except Exception as e:
        print(f'Error fetching crew config from database: {e}')
        return None


def save_analysis_to_database(
    crew_type: str,
    result: Any,
    user_id: Optional[str] = None,
    account_id: Optional[str] = None,
    salesforce_account_id: Optional[str] = None,
    salesforce_project_id: Optional[str] = None,
    provider: str = 'openai',
    model: Optional[str] = None
) -> bool:
    """
    Save crew analysis result to crew_analysis_history table.

    Args:
        crew_type: Type of crew analysis (e.g., 'sentiment', 'account', 'implementation')
        result: Analysis result (will be JSON serialized if not already a string)
        user_id: Optional user ID
        account_id: Optional account ID (internal)
        salesforce_account_id: Optional Salesforce account ID
        salesforce_project_id: Optional Salesforce project ID
        provider: LLM provider (default: 'openai')
        model: Model name (default: from OPENAI_MODEL_NAME env var)

    Returns:
        True if save successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            print('Warning: Supabase not configured, skipping database save')
            return False

        # Don't save if no identifying information provided
        if not user_id and not account_id and not salesforce_account_id:
            print('Warning: No userId/accountId/salesforceAccountId provided, skipping database save')
            return False

        # Resolve account_id from salesforce_account_id if needed
        resolved_account_id = account_id
        if not resolved_account_id and salesforce_account_id:
            try:
                acc_result = supabase.table('accounts').select('id').eq('salesforce_id', salesforce_account_id).limit(1).execute()
                if acc_result.data and len(acc_result.data) > 0:
                    resolved_account_id = acc_result.data[0].get('id')
            except Exception as acc_err:
                print(f'Warning: Could not resolve account_id from salesforce_account_id: {acc_err}')

        # Prepare save data
        result_str = result if isinstance(result, str) else json.dumps(result)
        save_data = {
            'crew_type': crew_type,
            'result': result_str,
            'provider': provider,
            'model': model or os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
        }

        if user_id:
            save_data['user_id'] = user_id
        if resolved_account_id:
            save_data['account_id'] = resolved_account_id
        if salesforce_account_id:
            save_data['salesforce_account_id'] = salesforce_account_id
        if salesforce_project_id:
            save_data['salesforce_project_id'] = salesforce_project_id

        # Insert into database
        result_insert = supabase.table('crew_analysis_history').insert(save_data).execute()

        if hasattr(result_insert, 'error') and result_insert.error:
            print(f'Error saving {crew_type} analysis to database: {result_insert.error}')
            return False

        print(f'✅ Saved {crew_type} analysis to crew_analysis_history')
        return True

    except Exception as save_error:
        print(f'Error saving {crew_type} analysis to database: {save_error}')
        return False


def build_system_prompt(crew_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Build a complete system prompt from crew config parts.

    Args:
        crew_config: Crew configuration dictionary

    Returns:
        Combined system prompt string or None if no config
    """
    if not crew_config:
        return None

    system_prompt_parts = []

    if crew_config.get('system_prompt'):
        system_prompt_parts.append(crew_config['system_prompt'])
    if crew_config.get('evaluation_criteria'):
        system_prompt_parts.append('\n\n' + crew_config['evaluation_criteria'])
    if crew_config.get('scoring_rubric'):
        system_prompt_parts.append('\n\n' + crew_config['scoring_rubric'])

    return '\n'.join(system_prompt_parts) if system_prompt_parts else None
