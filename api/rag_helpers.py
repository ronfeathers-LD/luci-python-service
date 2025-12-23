"""
RAG (Retrieval-Augmented Generation) Helper Functions for CrewAI Analysis

Uses vector similarity search to retrieve relevant account context
instead of passing full transcripts, significantly reducing token count.
"""

import os
import json
from typing import Optional, List, Dict, Any


def get_openai_embedding(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Generate embedding using OpenAI's text-embedding-3-small model (1536 dimensions)
    """
    import requests

    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is required for embeddings')

    response = requests.post(
        'https://api.openai.com/v1/embeddings',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        json={
            'model': 'text-embedding-3-small',
            'input': text[:8000]  # Limit input to avoid token limits
        }
    )

    if response.status_code != 200:
        raise ValueError(f'OpenAI embedding API error: {response.status_code} - {response.text}')

    data = response.json()
    return data['data'][0]['embedding']


def get_relevant_context(
    account_id: str,
    query: str,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    match_threshold: float = 0.5,
    match_count: int = 15,
    data_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Retrieve relevant context chunks from the vector database using semantic search.

    Args:
        account_id: UUID of the account to search within
        query: The query/question to find relevant context for
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key
        openai_api_key: OpenAI API key for generating query embedding
        match_threshold: Minimum similarity threshold (0-1)
        match_count: Maximum number of chunks to retrieve
        data_types: Optional list of data types to filter by (e.g., ['transcription', 'case'])

    Returns:
        Dictionary with:
        - context: Combined context string
        - chunks: List of individual chunks with metadata
        - data_type_counts: Count of chunks by type
    """
    from supabase import create_client, Client

    supabase_url = supabase_url or os.environ.get('SUPABASE_URL')
    supabase_key = supabase_key or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required')

    # Generate query embedding
    query_embedding = get_openai_embedding(query, openai_api_key)

    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)

    # Call the vector similarity search RPC function
    try:
        result = supabase.rpc('match_account_embeddings', {
            'query_embedding': query_embedding,
            'match_account_id': account_id,
            'match_threshold': match_threshold,
            'match_count': match_count
        }).execute()

        if not result.data:
            print(f'No matching embeddings found for account {account_id}')
            return {
                'context': '',
                'chunks': [],
                'data_type_counts': {}
            }

        chunks = result.data

    except Exception as e:
        print(f'Error calling match_account_embeddings RPC: {e}')
        # Fallback: get diverse embeddings directly
        chunks = []
        fallback_types = data_types or ['transcription', 'case', 'sentiment', 'contact', 'account']

        for data_type in fallback_types:
            try:
                type_result = supabase.table('account_embeddings') \
                    .select('id, content, data_type, metadata') \
                    .eq('account_id', account_id) \
                    .eq('data_type', data_type) \
                    .limit(3) \
                    .execute()

                if type_result.data:
                    for row in type_result.data:
                        chunks.append({
                            'id': row['id'],
                            'content': row['content'],
                            'data_type': row['data_type'],
                            'metadata': row['metadata'],
                            'similarity': 0.5  # Default similarity for fallback
                        })
            except Exception as type_error:
                print(f'Error fetching {data_type} embeddings: {type_error}')

    # Filter by data types if specified
    if data_types:
        chunks = [c for c in chunks if c.get('data_type') in data_types]

    # Sort by similarity (highest first)
    chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # Count chunks by data type
    data_type_counts = {}
    for chunk in chunks:
        dt = chunk.get('data_type', 'unknown')
        data_type_counts[dt] = data_type_counts.get(dt, 0) + 1

    # Build context string
    context_parts = []
    for i, chunk in enumerate(chunks):
        data_type = chunk.get('data_type', 'unknown').upper()
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        similarity = chunk.get('similarity', 0)

        # Add metadata context
        meta_parts = []
        if metadata.get('meetingSubject'):
            meta_parts.append(f"Meeting: {metadata['meetingSubject']}")
        if metadata.get('meetingDate'):
            meta_parts.append(f"Date: {metadata['meetingDate']}")
        if metadata.get('caseNumber'):
            meta_parts.append(f"Case: {metadata['caseNumber']}")
        if metadata.get('status'):
            meta_parts.append(f"Status: {metadata['status']}")
        if metadata.get('priority'):
            meta_parts.append(f"Priority: {metadata['priority']}")
        if metadata.get('score'):
            meta_parts.append(f"Sentiment Score: {metadata['score']}/10")

        meta_str = ' | '.join(meta_parts) if meta_parts else ''
        header = f"[{data_type} {i+1}] (relevance: {similarity:.2f})"
        if meta_str:
            header += f" {meta_str}"

        context_parts.append(f"{header}\n{content}")

    context = '\n\n---\n\n'.join(context_parts)

    print(f'Retrieved {len(chunks)} context chunks: {json.dumps(data_type_counts)}')

    return {
        'context': context,
        'chunks': chunks,
        'data_type_counts': data_type_counts
    }


def get_analysis_query(analysis_type: str) -> str:
    """
    Generate an appropriate query for different analysis types.
    This query is used to find relevant embeddings.
    """
    queries = {
        'account': '''
            Customer sentiment satisfaction concerns issues problems
            relationship health engagement communication
            meeting outcomes action items follow-ups
            support cases escalations complaints resolutions
            key stakeholders decision makers champions
            risks opportunities churn signals expansion potential
        ''',
        'sentiment': '''
            Customer sentiment feelings satisfaction frustration concerns
            emotional tone language patterns urgency indicators
            relationship health positive negative signals
            recent interactions current state mood
            support experience case resolution quality
        ''',
        'implementation': '''
            Project status timeline milestones deliverables
            implementation progress blockers dependencies
            stakeholder alignment communication effectiveness
            project management meeting outcomes decisions
            risks issues escalations action items
        ''',
        'overview': '''
            Portfolio health account priorities attention needed
            risk factors opportunities trends patterns
            sentiment scores relationship trajectories
            key accounts strategic importance
        '''
    }

    return queries.get(analysis_type, queries['account']).strip()
