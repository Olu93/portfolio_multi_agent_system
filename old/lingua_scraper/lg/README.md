# LangGraph Supervisor Multi-Agent System

A configuration-driven multi-agent system using LangGraph Supervisor, with agents defined through YAML configuration files.

## Features

- **Configuration-Driven**: All agents, LLMs, and MCP servers defined in YAML
- **Type-Safe**: Uses Pydantic Settings for validation and type safety
- **Dynamic Agent Creation**: Agents created automatically from configuration
- **MCP Integration**: Seamless integration with MCP servers for tools
- **Flexible LLM Configuration**: Each agent can use different LLM models and settings

## Configuration

### YAML Configuration Structure

The system uses a YAML file (`config/agents.yaml`) to define all agents and their configurations:

```yaml
agents:
  research_agent:
    name: "research_agent"
    description: "Specializes in web search and finding relevant URLs using DuckDuckGo"
    llm:
      model: "gpt-4o-mini"
      temperature: 0.0
      max_tokens: 4000
    system_prompt: |
      You are a specialized research agent...
    mcp_servers:
      duckduckgo:
        url: "http://localhost:8000/mcp"
        transport: "streamable_http"
    response_format: "ResearchResponse"

  scraper_agent:
    name: "scraper_agent"
    description: "Specializes in extracting content from web pages using Playwright"
    llm:
      model: "gpt-4o-mini"
      temperature: 0.1
      max_tokens: 4000
    system_prompt: |
      You are a specialized assistant for scraping web content...
    mcp_servers:
      playwright:
        url: "http://localhost:8001/mcp"
        transport: "streamable_http"
      playwright_extension:
        url: "http://localhost:8002/mcp"
        transport: "streamable_http"
    response_format: "ScrapedData"

supervisor:
  llm:
    model: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 4000
  prompt_template: |
    You manage a multi-agent system with the following specialized agents:

    {agent_descriptions}
    
    Assign work to the appropriate agent based on the user's request...
```

### Configuration Components

#### Agent Configuration
- **name**: Unique identifier for the agent
- **description**: Human-readable description (used in supervisor prompt)
- **llm**: LLM configuration (model, temperature, max_tokens)
- **system_prompt**: The system prompt for the agent
- **mcp_servers**: Dictionary of MCP server configurations
- **response_format**: Optional response format class name

#### MCP Server Configuration
- **url**: MCP server endpoint URL
- **transport**: Transport type (usually "streamable_http")

#### Supervisor Configuration
- **llm**: LLM configuration for the supervisor
- **prompt_template**: Template for supervisor prompt (uses {agent_descriptions} placeholder)

## Usage

### Basic Usage

```python
from lingua_scraper.lg.main import main
import asyncio

# Run with default configuration
asyncio.run(main())

# Run with custom configuration file
# python main.py custom_config.yaml "Find real estate agents in Amsterdam"
```

### Command Line Usage

```bash
# Run with default configuration
python main.py

# Run with custom configuration
python main.py config/custom_agents.yaml

# Run with query
python main.py "Find real estate agents in Amsterdam"

# Run with custom config and query
python main.py config/custom_agents.yaml "Find real estate agents in Amsterdam"
```

### Programmatic Usage

```python
from lingua_scraper.lg.agent_factory import create_agent, create_all_agents
from lingua_scraper.lg.config_manager import ConfigManager

# Load configuration
config = ConfigManager.get_config("config/agents.yaml")

# Create a specific agent
research_agent = await create_agent("research_agent")

# Create all agents
all_agents = await create_all_agents()

# Get supervisor prompt with agent descriptions
supervisor_prompt = config.get_supervisor_prompt()
```

## Architecture

### Core Components

1. **ConfigManager**: Singleton for loading and managing configuration
2. **AgentFactory**: Creates agents from configuration
3. **DataModels**: Pydantic models for response formats
4. **Main**: Orchestrates the supervisor system

### Configuration Flow

1. **YAML Loading**: Configuration loaded from YAML file using custom Pydantic Settings source
2. **Validation**: Pydantic validates all configuration values
3. **Agent Creation**: Factory creates agents with MCP tools and LLM instances
4. **Supervisor Assembly**: Supervisor created with dynamic prompt and agent list

### Environment Variables

The system supports environment variable overrides for any configuration value:

```bash
# Override LLM model for all agents
export LLM_MODEL="gpt-4"

# Override specific agent's model
export RESEARCH_AGENT_LLM_MODEL="gpt-3.5-turbo"

# Override MCP server URL
export RESEARCH_AGENT_MCP_SERVERS_DUCKDUCKGO_URL="http://localhost:9000/mcp"
```

## Adding New Agents

To add a new agent:

1. **Add to YAML**: Define the agent in `config/agents.yaml`
2. **Add Response Format** (if needed): Create Pydantic model in `data_models.py`
3. **Register Response Format**: Add to `RESPONSE_FORMATS` in `agent_factory.py`

Example new agent:

```yaml
agents:
  new_agent:
    name: "new_agent"
    description: "Specializes in specific task"
    llm:
      model: "gpt-4o-mini"
      temperature: 0.0
      max_tokens: 4000
    system_prompt: |
      You are a specialized agent for...
    mcp_servers:
      custom_server:
        url: "http://localhost:8003/mcp"
        transport: "streamable_http"
    response_format: "CustomResponse"
```

## Dependencies

- `pydantic-settings[yaml]`: Configuration management
- `pyyaml`: YAML file parsing
- `langgraph-supervisor`: Multi-agent orchestration
- `langchain-mcp-adapters`: MCP server integration
- `langchain-openai`: LLM integration

## Configuration Priority

Configuration values are resolved in this order (highest to lowest priority):

1. Command line arguments
2. Environment variables
3. YAML configuration file
4. Default values in Pydantic models 