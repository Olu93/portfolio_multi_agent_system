"""
Core Agent Factory for LangGraph Supervisor Multi-Agent System
Creates agents based on YAML configuration
"""

import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config_manager import ConfigManager
from .data_models import ResearchResponse, ScrapedData

logger = logging.getLogger(__name__)


# ============================================================================
# Response Format Registry
# ============================================================================

RESPONSE_FORMATS = {
    "ResearchResponse": ResearchResponse,
    "ScrapedData": ScrapedData,
}


# ============================================================================
# Core Agent Factory
# ============================================================================

async def create_agent(agent_name: str, config_path: str = "config/agents.yaml"):
    """
    Create a single agent based on configuration.
    
    Args:
        agent_name: Name of the agent to create (e.g., "research_agent")
        config_path: Path to the YAML configuration file
    
    Returns:
        Configured LangGraph agent
    """
    # Load configuration
    config = ConfigManager.get_config(config_path)
    
    if agent_name not in config.agents:
        raise ValueError(f"Agent '{agent_name}' not found in configuration")
    
    agent_config = config.agents[agent_name]
    
    # Initialize MCP tools
    tools = await _initialize_mcp_tools(agent_config.mcp_servers)
    
    # Create LLM instance
    llm = ChatOpenAI(**agent_config.llm.model_dump())
    
    # Create response parser if specified
    response_format = None
    if agent_config.response_format:
        if agent_config.response_format in RESPONSE_FORMATS:
            response_format_class = RESPONSE_FORMATS[agent_config.response_format]
            parser = JsonOutputParser(pydantic_object=response_format_class)
            response_format = (parser.get_format_instructions(), response_format_class)
        else:
            logger.warning(f"Unknown response format: {agent_config.response_format}")
    
    # Create the agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=agent_config.system_prompt,
        response_format=response_format,
        name=agent_config.name
    )
    
    logger.info(f"Created agent '{agent_name}' with {len(tools)} tools")
    return agent


async def _initialize_mcp_tools(mcp_servers: Dict[str, Any]) -> List[Any]:
    """
    Initialize MCP tools from server configuration.
    
    Args:
        mcp_servers: Dictionary of MCP server configurations
    
    Returns:
        List of MCP tools
    """
    if not mcp_servers:
        logger.warning("No MCP servers configured for agent")
        return []
    
    try:
        # Convert MCP server config to the format expected by MultiServerMCPClient
        server_configs = {}
        for server_name, server_config in mcp_servers.items():
            server_configs[server_name] = {
                "url": server_config.url,
                "transport": server_config.transport
            }
        
        mcp_client = MultiServerMCPClient(server_configs)
        tools = await mcp_client.get_tools()
        
        logger.info(f"Loaded {len(tools)} tools from MCP servers: {list(mcp_servers.keys())}")
        return tools
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP tools: {e}")
        return []


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_all_agents(config_path: str = "config/agents.yaml") -> Dict[str, Any]:
    """
    Create all agents defined in the configuration.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary mapping agent names to agent instances
    """
    config = ConfigManager.get_config(config_path)
    agents = {}
    
    for agent_name in config.agents.keys():
        try:
            agents[agent_name] = await create_agent(agent_name, config_path)
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_name}': {e}")
    
    return agents


async def create_supervisor_llm(config_path: str = "config/agents.yaml") -> ChatOpenAI:
    """
    Create the supervisor LLM based on configuration.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Configured ChatOpenAI instance for supervisor
    """
    config = ConfigManager.get_config(config_path)
    return ChatOpenAI(**config.supervisor.llm.model_dump())


def get_supervisor_prompt(config_path: str = "config/agents.yaml") -> str:
    """
    Get the supervisor prompt with agent descriptions injected.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Formatted supervisor prompt
    """
    config = ConfigManager.get_config(config_path)
    return config.get_supervisor_prompt() 