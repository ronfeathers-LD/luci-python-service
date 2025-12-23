"""
Crew Configuration Module for Python Service
This module fetches crew configuration from Supabase and builds CrewAI agents and tasks dynamically.

Place this file in: api/crew/crew_config.py in your Python service repository
"""

import os
import json
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI


class CrewConfigLoader:
    """Loads and manages crew configuration from Supabase database"""
    
    def __init__(self):
        """Initialize Supabase client"""
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing required environment variables: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
            )
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def get_crew_config(self, crew_type: str) -> Optional[Dict[str, Any]]:
        """
        Fetch crew configuration from database by crew_type
        
        Args:
            crew_type: The crew_type identifier (e.g., 'account', 'implementation', 'overview')
            
        Returns:
            Dictionary with crew configuration or None if not found
        """
        try:
            response = self.supabase.table('crews').select('*').eq('crew_type', crew_type).eq('enabled', True).single().execute()
            
            if response.data:
                return response.data
            return None
        except Exception as e:
            print(f"Error fetching crew config for {crew_type}: {str(e)}")
            return None
    
    def validate_crew_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate that crew config has required fields

        Returns:
            (is_valid, error_message)
        """
        # Check basic required fields
        if 'crew_type' not in config:
            return False, "Missing required field: crew_type"
        if 'endpoint' not in config:
            return False, "Missing required field: endpoint"

        # Support both new ('agents'/'tasks') and legacy ('agent_configs'/'task_configs') field names
        agents = config.get('agents') or config.get('agent_configs')
        tasks = config.get('tasks') or config.get('task_configs')

        if not agents:
            return False, "Missing required field: agents (or agent_configs)"

        if not tasks:
            return False, "Missing required field: tasks (or task_configs)"

        if not isinstance(agents, list) or len(agents) == 0:
            return False, "agents must be a non-empty list"

        if not isinstance(tasks, list) or len(tasks) == 0:
            return False, "tasks must be a non-empty list"

        return True, None


class CrewBuilder:
    """Builds CrewAI agents, tasks, and crews from database configuration"""
    
    def __init__(self, llm=None):
        """
        Initialize CrewBuilder
        
        Args:
            llm: Optional LangChain LLM instance. If not provided, creates default ChatOpenAI
        """
        if llm is None:
            self.llm = ChatOpenAI(
                model=os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini'),
                temperature=0.7,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            self.llm = llm
    
    def build_agent(self, agent_config: Dict[str, Any]) -> Agent:
        """
        Build a CrewAI Agent from configuration
        
        Args:
            agent_config: Dictionary with agent configuration:
                - role: Agent role (required)
                - goal: Agent goal (required)
                - backstory: Agent backstory (optional)
                - verbose: Whether to enable verbose mode (optional, default: True)
        
        Returns:
            Configured CrewAI Agent
        """
        if 'role' not in agent_config or 'goal' not in agent_config:
            raise ValueError("Agent config must include 'role' and 'goal'")
        
        agent_kwargs = {
            'role': agent_config['role'],
            'goal': agent_config['goal'],
            'llm': self.llm,
            'verbose': agent_config.get('verbose', True)
        }
        
        if 'backstory' in agent_config:
            agent_kwargs['backstory'] = agent_config['backstory']
        
        return Agent(**agent_kwargs)
    
    def build_agents(self, agent_configs: List[Dict[str, Any]]) -> List[Agent]:
        """
        Build multiple agents from configuration list
        
        Args:
            agent_configs: List of agent configuration dictionaries
        
        Returns:
            List of configured CrewAI Agents
        """
        agents = []
        agent_map = {}  # Map agent role to Agent instance for task references
        
        for agent_config in agent_configs:
            agent = self.build_agent(agent_config)
            agents.append(agent)
            agent_map[agent_config['role']] = agent
        
        return agents, agent_map
    
    def build_task(
        self,
        task_config: Dict[str, Any],
        agent_map: Dict[str, Agent],
        context_variables: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Build a CrewAI Task from configuration

        Args:
            task_config: Dictionary with task configuration:
                - description: Task description (required, can include {variables})
                - output: Expected output description (required, also accepts 'expected_output')
                - agent: Agent role name (required) - must match an agent role
                - context: List of task IDs this task depends on (optional)
            agent_map: Dictionary mapping agent roles to Agent instances
            context_variables: Optional dictionary of variables to format into description

        Returns:
            Configured CrewAI Task
        """
        # Support both 'output' (new) and 'expected_output' (legacy) field names
        expected_output = task_config.get('output') or task_config.get('expected_output')
        if 'description' not in task_config or not expected_output:
            raise ValueError("Task config must include 'description' and 'output' (or 'expected_output')")
        
        if 'agent' not in task_config:
            raise ValueError("Task config must include 'agent' (agent role name)")
        
        agent_role = task_config['agent']
        if agent_role not in agent_map:
            raise ValueError(f"Agent role '{agent_role}' not found in agent_map")
        
        # Format description with context variables if provided
        description = task_config['description']
        if context_variables:
            try:
                description = description.format(**context_variables)
            except KeyError as e:
                print(f"Warning: Missing context variable {e} in task description")
        
        task_kwargs = {
            'description': description,
            'expected_output': expected_output,
            'agent': agent_map[agent_role]
        }
        
        # Handle task context/dependencies if specified
        if 'context' in task_config and isinstance(task_config['context'], list):
            # Note: CrewAI task context is handled via task dependencies
            # This would need to be implemented based on your CrewAI version
            pass
        
        return Task(**task_kwargs)
    
    def build_tasks(
        self, 
        task_configs: List[Dict[str, Any]], 
        agent_map: Dict[str, Agent],
        context_variables: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """
        Build multiple tasks from configuration list
        
        Args:
            task_configs: List of task configuration dictionaries
            agent_map: Dictionary mapping agent roles to Agent instances
            context_variables: Optional dictionary of variables to format into descriptions
        
        Returns:
            List of configured CrewAI Tasks
        """
        tasks = []
        for task_config in task_configs:
            task = self.build_task(task_config, agent_map, context_variables)
            tasks.append(task)
        return tasks
    
    def build_crew(
        self,
        crew_config: Dict[str, Any],
        context_variables: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Crew:
        """
        Build a complete CrewAI Crew from database configuration
        
        Args:
            crew_config: Crew configuration dictionary from database
            context_variables: Optional dictionary of variables to format into task descriptions
            verbose: Whether to enable verbose mode (default: True)
        
        Returns:
            Configured CrewAI Crew ready to execute
        """
        # Validate config
        loader = CrewConfigLoader()
        is_valid, error = loader.validate_crew_config(crew_config)
        if not is_valid:
            raise ValueError(f"Invalid crew config: {error}")

        # Build agents (support both 'agents' and legacy 'agent_configs')
        agent_configs = crew_config.get('agents') or crew_config.get('agent_configs', [])
        agents, agent_map = self.build_agents(agent_configs)

        # Build tasks (support both 'tasks' and legacy 'task_configs')
        task_configs = crew_config.get('tasks') or crew_config.get('task_configs', [])
        tasks = self.build_tasks(task_configs, agent_map, context_variables)
        
        # Create crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=verbose
        )
        
        return crew


def get_crew_with_config(
    crew_type: str,
    context_variables: Optional[Dict[str, Any]] = None,
    llm=None,
    verbose: bool = True
) -> tuple[Optional[Crew], Optional[Dict[str, Any]], Optional[str]]:
    """
    Convenience function to get a crew with configuration from database
    
    Args:
        crew_type: The crew_type identifier
        context_variables: Optional variables to format into task descriptions
        llm: Optional LangChain LLM instance
        verbose: Whether to enable verbose mode
    
    Returns:
        (crew, config, error_message)
        - crew: Configured CrewAI Crew or None if error
        - config: Crew configuration dictionary or None if error
        - error_message: Error message or None if successful
    """
    try:
        # Load config
        loader = CrewConfigLoader()
        config = loader.get_crew_config(crew_type)
        
        if not config:
            return None, None, f"Crew '{crew_type}' not found or not enabled"
        
        # Build crew
        builder = CrewBuilder(llm=llm)
        crew = builder.build_crew(config, context_variables, verbose)
        
        return crew, config, None
    
    except Exception as e:
        error_msg = f"Error building crew '{crew_type}': {str(e)}"
        print(error_msg)
        return None, None, error_msg


def format_system_prompt(crew_config: Dict[str, Any], context_variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Format system prompt from crew config with context variables
    
    Args:
        crew_config: Crew configuration dictionary
        context_variables: Optional variables to format into prompt
    
    Returns:
        Formatted system prompt string
    """
    system_prompt = crew_config.get('system_prompt', '')
    
    if context_variables and system_prompt:
        try:
            system_prompt = system_prompt.format(**context_variables)
        except KeyError as e:
            print(f"Warning: Missing context variable {e} in system prompt")
    
    return system_prompt

