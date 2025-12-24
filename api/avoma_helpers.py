"""
Avoma API Helper Functions for CrewAI Analysis

Provides fallback data fetching from Avoma when RAG embeddings are not available.
Uses the same API that the Avoma MCP server wraps.
"""

import os
import json
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


def get_avoma_config(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Fetch Avoma API configuration from Supabase database.

    Returns:
        Dictionary with 'api_key' and 'api_url' or None if not configured
    """
    try:
        from supabase import create_client, Client

        supabase_url = supabase_url or os.environ.get('SUPABASE_URL')
        supabase_key = supabase_key or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')

        if not supabase_url or not supabase_key:
            print('Warning: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set')
            return None

        supabase: Client = create_client(supabase_url, supabase_key)

        result = supabase.table('avoma_configs').select('api_key, api_url').eq('is_active', True).limit(1).execute()

        if not result.data or len(result.data) == 0:
            print('Warning: No active Avoma configuration found')
            return None

        config = result.data[0]
        return {
            'api_key': config.get('api_key'),
            'api_url': config.get('api_url', 'https://api.avoma.com/v1')
        }
    except Exception as e:
        print(f'Error fetching Avoma config: {e}')
        return None


def avoma_request(
    endpoint: str,
    api_key: str,
    api_url: str = 'https://api.avoma.com/v1',
    method: str = 'GET',
    params: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Make authenticated request to Avoma API.
    """
    try:
        url = f"{api_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print(f'Avoma API error: {response.status_code} - {response.text}')
            return None

        return response.json()
    except Exception as e:
        print(f'Error making Avoma request: {e}')
        return None


def list_meetings(
    api_key: str,
    api_url: str = 'https://api.avoma.com/v1',
    salesforce_account_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    limit: int = 10,
    months_back: int = 12
) -> List[Dict]:
    """
    List meetings from Avoma API (equivalent to MCP list_meetings tool).

    Args:
        api_key: Avoma API key
        api_url: Avoma API base URL
        salesforce_account_id: Filter by Salesforce Account ID (preferred)
        customer_name: Filter by customer name (fallback)
        limit: Maximum number of meetings to return
        months_back: How many months back to search

    Returns:
        List of meeting objects
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=months_back * 30)

    params = {
        'page_size': str(limit),
        'from_date': from_date.isoformat(),
        'to_date': to_date.isoformat()
    }

    # Priority: Salesforce Account ID > Customer Name
    if salesforce_account_id:
        params['crm_account_ids'] = salesforce_account_id
    elif customer_name:
        params['customer_name'] = customer_name

    result = avoma_request('/meetings', api_key, api_url, params=params)

    if not result:
        return []

    # Handle paginated response
    meetings = result.get('results', result.get('meetings', []))
    if isinstance(result, list):
        meetings = result

    return meetings


def get_meeting_transcript(
    meeting_uuid: str,
    api_key: str,
    api_url: str = 'https://api.avoma.com/v1'
) -> Optional[Dict]:
    """
    Get transcript for a meeting (equivalent to MCP get_meeting_transcript tool).

    Returns:
        Dictionary with 'text' (formatted transcript) and 'speakers'
    """
    result = avoma_request(f'/transcriptions/{meeting_uuid}/', api_key, api_url)

    if not result:
        return None

    # Format transcript with speaker names
    speakers = result.get('speakers', [])
    speaker_map = {s.get('id'): s.get('name', f"Speaker {s.get('id')}") for s in speakers}

    transcript_text = ''
    transcript_segments = result.get('transcript', [])

    if transcript_segments and isinstance(transcript_segments, list):
        formatted_segments = []
        for segment in transcript_segments:
            speaker_id = segment.get('speaker_id')
            speaker_name = speaker_map.get(speaker_id, f"Speaker {speaker_id}")
            text = segment.get('transcript', '')
            formatted_segments.append(f"{speaker_name}: {text}")
        transcript_text = '\n\n'.join(formatted_segments)
    elif result.get('content'):
        transcript_text = result.get('content')

    return {
        'text': transcript_text,
        'speakers': speakers,
        'raw': result
    }


def get_meeting_notes(
    meeting_uuid: str,
    api_key: str,
    api_url: str = 'https://api.avoma.com/v1'
) -> Optional[str]:
    """
    Get meeting notes/summary (equivalent to MCP get_meeting_notes tool).

    Returns:
        Meeting notes as string
    """
    result = avoma_request(f'/meetings/{meeting_uuid}', api_key, api_url)

    if not result:
        return None

    # Extract notes from meeting details
    notes = result.get('notes', result.get('summary', ''))

    # Also try to get AI-generated notes if available
    ai_notes = result.get('ai_notes', result.get('ai_summary', ''))

    if ai_notes:
        return f"{notes}\n\nAI Summary:\n{ai_notes}" if notes else ai_notes

    return notes


def get_avoma_context_for_account(
    salesforce_account_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    account_id: Optional[str] = None,
    max_meetings: int = 5,
    max_transcript_chars: int = 10000
) -> Dict[str, Any]:
    """
    Fetch and format Avoma meeting data for crew analysis.
    This is the main function to use as a fallback when RAG is not available.

    Args:
        salesforce_account_id: Salesforce Account ID
        customer_name: Customer/Account name
        account_id: Internal account UUID (used to look up salesforce_id if needed)
        max_meetings: Maximum number of meetings to include
        max_transcript_chars: Maximum characters per transcript

    Returns:
        Dictionary with:
        - context: Formatted context string for LLM
        - meetings: List of meeting data
        - success: Boolean indicating if data was fetched
        - error: Error message if failed
    """
    # Get Avoma configuration
    config = get_avoma_config()
    if not config:
        return {
            'context': '',
            'meetings': [],
            'success': False,
            'error': 'Avoma not configured'
        }

    api_key = config['api_key']
    api_url = config['api_url']

    # If we only have account_id, try to look up salesforce_id
    if not salesforce_account_id and account_id:
        try:
            from supabase import create_client
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            if supabase_url and supabase_key:
                client = create_client(supabase_url, supabase_key)
                result = client.table('accounts').select('salesforce_id, name').eq('id', account_id).limit(1).execute()
                if result.data:
                    salesforce_account_id = result.data[0].get('salesforce_id')
                    if not customer_name:
                        customer_name = result.data[0].get('name')
        except Exception as e:
            print(f'Error looking up account: {e}')

    if not salesforce_account_id and not customer_name:
        return {
            'context': '',
            'meetings': [],
            'success': False,
            'error': 'No salesforce_account_id or customer_name provided'
        }

    # Fetch meetings from Avoma
    print(f'Fetching Avoma meetings for {"SF:" + salesforce_account_id if salesforce_account_id else customer_name}')
    meetings = list_meetings(
        api_key=api_key,
        api_url=api_url,
        salesforce_account_id=salesforce_account_id,
        customer_name=customer_name,
        limit=max_meetings
    )

    if not meetings:
        return {
            'context': '',
            'meetings': [],
            'success': False,
            'error': 'No meetings found in Avoma'
        }

    # Fetch transcripts for each meeting
    context_parts = []
    meeting_data = []

    for i, meeting in enumerate(meetings[:max_meetings]):
        meeting_uuid = meeting.get('uuid') or meeting.get('id')
        subject = meeting.get('subject', meeting.get('title', 'Unknown Meeting'))
        meeting_date = meeting.get('scheduled_at', meeting.get('start_time', meeting.get('date', 'Unknown Date')))

        # Try to get transcript
        transcript_data = None
        if meeting_uuid:
            transcript_data = get_meeting_transcript(meeting_uuid, api_key, api_url)

        # Format meeting context
        meeting_context = f"--- Meeting {i+1}: {subject} ({meeting_date}) ---"

        if transcript_data and transcript_data.get('text'):
            transcript_text = transcript_data['text']
            # Truncate if too long
            if len(transcript_text) > max_transcript_chars:
                # Keep first 30% and last 70% for context
                first_part = transcript_text[:max_transcript_chars // 3]
                last_part = transcript_text[-(max_transcript_chars * 2 // 3):]
                transcript_text = f"{first_part}\n\n... [middle section truncated] ...\n\n{last_part}"
            meeting_context += f"\n{transcript_text}"
        else:
            # Try to get notes instead
            notes = get_meeting_notes(meeting_uuid, api_key, api_url) if meeting_uuid else None
            if notes:
                meeting_context += f"\nNotes: {notes}"
            else:
                meeting_context += "\n(No transcript or notes available)"

        context_parts.append(meeting_context)
        meeting_data.append({
            'uuid': meeting_uuid,
            'subject': subject,
            'date': meeting_date,
            'has_transcript': bool(transcript_data and transcript_data.get('text')),
            'attendees': meeting.get('attendees', [])
        })

    context = '\n\n'.join(context_parts)

    print(f'Avoma context retrieved: {len(meetings)} meetings, {len(context)} chars')

    return {
        'context': context,
        'meetings': meeting_data,
        'success': True,
        'error': None
    }
