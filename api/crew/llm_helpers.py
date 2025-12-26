"""
Shared LLM helper functions for crew modules.
Provides lazy-loading of heavy imports (crewai, langchain) to reduce cold start time.

PERFORMANCE OPTIMIZATION: Heavy imports are lazy-loaded inside functions
to reduce cold start time by ~1-2 seconds on serverless platforms.
"""

import os
import warnings
from typing import Optional, Dict, Any

# Suppress common warnings from dependencies
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


def get_langchain_openai():
    """
    Lazy load langchain_openai ChatOpenAI class.

    Returns:
        ChatOpenAI class from langchain_openai
    """
    global _langchain_openai
    if _langchain_openai is None:
        from langchain_openai import ChatOpenAI
        _langchain_openai = ChatOpenAI
    return _langchain_openai


def get_crewai():
    """
    Lazy load crewai module classes.

    Returns:
        Dictionary with Crew, Agent, Task classes
    """
    global _crewai
    if _crewai is None:
        from crewai import Crew, Agent, Task
        _crewai = {'Crew': Crew, 'Agent': Agent, 'Task': Task}
    return _crewai


def get_llm(openai_api_key: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.7):
    """
    Get OpenAI LLM instance with lazy loading.

    Args:
        openai_api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        model: Optional model name (defaults to OPENAI_MODEL_NAME env var or 'gpt-4o-mini')
        temperature: Temperature setting (default: 0.7)

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If no API key is provided or found in environment
    """
    ChatOpenAI = get_langchain_openai()

    api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is required')

    # Set in environment for other components that may need it
    os.environ['OPENAI_API_KEY'] = api_key

    model_name = model or os.environ.get('OPENAI_MODEL_NAME', 'gpt-4o-mini')

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )


def create_agent(
    role: str,
    goal: str,
    backstory: str,
    llm=None,
    verbose: bool = True,
    **kwargs
):
    """
    Create a CrewAI Agent with lazy loading.

    Args:
        role: Agent's role
        goal: Agent's goal
        backstory: Agent's backstory
        llm: Optional LLM instance (will create default if not provided)
        verbose: Enable verbose mode (default: True)
        **kwargs: Additional agent parameters

    Returns:
        Configured CrewAI Agent instance
    """
    crewai = get_crewai()
    Agent = crewai['Agent']

    if llm is None:
        llm = get_llm()

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=verbose,
        **kwargs
    )


def create_task(
    description: str,
    expected_output: str,
    agent,
    **kwargs
):
    """
    Create a CrewAI Task with lazy loading.

    Args:
        description: Task description
        expected_output: Expected output description
        agent: Agent assigned to the task
        **kwargs: Additional task parameters

    Returns:
        Configured CrewAI Task instance
    """
    crewai = get_crewai()
    Task = crewai['Task']

    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        **kwargs
    )


def create_crew(agents: list, tasks: list, verbose: bool = True, **kwargs):
    """
    Create a CrewAI Crew with lazy loading.

    Args:
        agents: List of Agent instances
        tasks: List of Task instances
        verbose: Enable verbose mode (default: True)
        **kwargs: Additional crew parameters

    Returns:
        Configured CrewAI Crew instance
    """
    crewai = get_crewai()
    Crew = crewai['Crew']

    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=verbose,
        **kwargs
    )


def summarize_text(text: str, llm, max_length: int = 1000, prompt_template: Optional[str] = None) -> str:
    """
    Summarize text using LLM if it exceeds max_length.

    Args:
        text: Text to summarize
        llm: LLM instance to use for summarization
        max_length: Maximum length before summarization (default: 1000)
        prompt_template: Optional custom prompt template

    Returns:
        Original text if short enough, otherwise summarized text
    """
    if not text or len(text) <= max_length:
        return text

    default_prompt = """Summarize the following text into key points, focusing on:
1. Main topics and decisions
2. Action items and commitments
3. Sentiment indicators (positive/negative language)
4. Key quotes that reveal customer sentiment

Text to summarize:
{text}

Provide a concise summary (max 500 words):"""

    prompt = (prompt_template or default_prompt).format(text=text)

    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Error summarizing text: {e}")
        # Return truncated text as fallback
        return text[:max_length] + "..."
