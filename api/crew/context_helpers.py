"""
Shared context retrieval helpers for crew modules.
Provides unified RAG and Avoma context fetching with fallback logic.
"""

import os
from typing import Optional, Dict, Any, Callable


def resolve_account_uuid(
    account_id: Optional[str] = None,
    salesforce_account_id: Optional[str] = None
) -> Optional[str]:
    """
    Resolve an account identifier to a UUID for RAG lookups.

    Args:
        account_id: Internal account ID (may already be UUID)
        salesforce_account_id: Salesforce account ID

    Returns:
        Account UUID or None if cannot be resolved
    """
    # Check if account_id is already a UUID (contains dashes and is 36 chars)
    if account_id and '-' in account_id and len(account_id) == 36:
        return account_id

    # Try to resolve from Salesforce ID
    lookup_id = salesforce_account_id or account_id
    if not lookup_id:
        return None

    try:
        from database_helpers import get_supabase_client

        supabase = get_supabase_client()
        if not supabase:
            return None

        result = supabase.table('accounts').select('id').eq('salesforce_id', lookup_id).limit(1).execute()
        if result.data:
            uuid = result.data[0]['id']
            print(f'Resolved account UUID: {uuid} from salesforce_id: {lookup_id}')
            return uuid
        else:
            print(f'Could not resolve salesforce_id {lookup_id} to UUID')
            return None

    except Exception as e:
        print(f'Error resolving account UUID: {e}')
        return None


def get_rag_context(
    account_id: Optional[str] = None,
    salesforce_account_id: Optional[str] = None,
    analysis_type: str = 'sentiment',
    match_count: int = 15,
    match_threshold: float = 0.4,
    send_progress: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Retrieve context using RAG (vector similarity search).

    Args:
        account_id: Internal account ID
        salesforce_account_id: Salesforce account ID
        analysis_type: Type of analysis ('sentiment', 'account', etc.)
        match_count: Number of chunks to retrieve
        match_threshold: Similarity threshold for matching
        send_progress: Optional callback for progress updates

    Returns:
        Dictionary with:
        - success: bool
        - context: str (the context text)
        - chunks: list (matched chunks)
        - data_type_counts: dict (counts by data type)
        - error: str (if failed)
    """
    try:
        # Resolve to UUID for RAG lookup
        rag_account_id = resolve_account_uuid(account_id, salesforce_account_id)
        if not rag_account_id:
            return {'success': False, 'error': 'Could not resolve account to UUID'}

        from rag_helpers import get_relevant_context, get_analysis_query

        if send_progress:
            send_progress('RAG Search', 'Retrieving relevant context from vector database...')

        # Generate analysis-specific query for embedding search
        analysis_query = get_analysis_query(analysis_type)
        print(f'Calling get_relevant_context with account_id={rag_account_id}')

        rag_result = get_relevant_context(
            account_id=rag_account_id,
            query=analysis_query,
            match_count=match_count,
            match_threshold=match_threshold
        )

        chunks = rag_result.get('chunks', [])
        context = rag_result.get('context', '')
        data_type_counts = rag_result.get('data_type_counts', {})

        print(f'RAG result: {len(chunks)} chunks, context length: {len(context)}')

        if context:
            if send_progress:
                send_progress('RAG Context', f'Using {len(context)} chars of relevant context')
            return {
                'success': True,
                'context': context,
                'chunks': chunks,
                'data_type_counts': data_type_counts
            }
        else:
            return {'success': False, 'error': 'RAG returned no context'}

    except Exception as e:
        print(f'RAG context retrieval failed: {e}')
        return {'success': False, 'error': str(e)}


def get_avoma_context(
    salesforce_account_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    account_id: Optional[str] = None,
    max_meetings: int = 5,
    max_transcript_chars: int = 10000,
    send_progress: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Retrieve context from Avoma API (real-time fetch).

    Args:
        salesforce_account_id: Salesforce account ID
        customer_name: Customer/company name for matching
        account_id: Internal account ID
        max_meetings: Maximum number of meetings to fetch
        max_transcript_chars: Maximum characters of transcript to include
        send_progress: Optional callback for progress updates

    Returns:
        Dictionary with:
        - success: bool
        - context: str (the context text)
        - meetings: list (meeting data)
        - error: str (if failed)
    """
    try:
        from avoma_helpers import get_avoma_context_for_account

        if send_progress:
            send_progress('Avoma Fetch', 'Fetching meeting data from Avoma...')

        avoma_result = get_avoma_context_for_account(
            salesforce_account_id=salesforce_account_id,
            customer_name=customer_name,
            account_id=account_id,
            max_meetings=max_meetings,
            max_transcript_chars=max_transcript_chars
        )

        if avoma_result.get('success') and avoma_result.get('context'):
            context = avoma_result['context']
            meetings = avoma_result.get('meetings', [])
            print(f'Avoma retrieved context: {len(meetings)} meetings, {len(context)} chars')

            if send_progress:
                send_progress('Avoma Context', f'Using {len(context)} chars from Avoma')

            return {
                'success': True,
                'context': context,
                'meetings': meetings
            }
        else:
            error = avoma_result.get('error', 'Unknown error')
            print(f'Avoma fetch failed or returned no data: {error}')
            return {'success': False, 'error': error}

    except Exception as e:
        print(f'Avoma context retrieval failed: {e}')
        return {'success': False, 'error': str(e)}


def get_context_with_fallback(
    account_id: Optional[str] = None,
    salesforce_account_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    use_rag: bool = True,
    analysis_type: str = 'sentiment',
    send_progress: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Get context using RAG with Avoma fallback.

    Tries RAG first (fastest - uses pre-computed embeddings), then falls back
    to Avoma API if RAG fails or returns no context.

    Args:
        account_id: Internal account ID
        salesforce_account_id: Salesforce account ID
        customer_name: Customer/company name
        use_rag: Whether to try RAG first (default: True)
        analysis_type: Type of analysis for query generation
        send_progress: Optional callback for progress updates

    Returns:
        Dictionary with:
        - success: bool
        - context: str
        - source: str ('rag', 'avoma', or 'none')
        - metadata: dict (source-specific metadata)
    """
    print(f'Context retrieval: use_rag={use_rag}, account_id={account_id}, salesforce_account_id={salesforce_account_id}')

    # Step 1: Try RAG if enabled
    if use_rag and (account_id or salesforce_account_id):
        rag_result = get_rag_context(
            account_id=account_id,
            salesforce_account_id=salesforce_account_id,
            analysis_type=analysis_type,
            send_progress=send_progress
        )

        if rag_result.get('success') and rag_result.get('context'):
            return {
                'success': True,
                'context': rag_result['context'],
                'source': 'rag',
                'metadata': {
                    'chunks': rag_result.get('chunks', []),
                    'data_type_counts': rag_result.get('data_type_counts', {})
                }
            }

    # Step 2: Fall back to Avoma
    avoma_result = get_avoma_context(
        salesforce_account_id=salesforce_account_id,
        customer_name=customer_name,
        account_id=account_id,
        send_progress=send_progress
    )

    if avoma_result.get('success') and avoma_result.get('context'):
        return {
            'success': True,
            'context': avoma_result['context'],
            'source': 'avoma',
            'metadata': {
                'meetings': avoma_result.get('meetings', [])
            }
        }

    # No context available
    return {
        'success': False,
        'context': '',
        'source': 'none',
        'metadata': {}
    }
