# Agent System Redesign

This directory contains the redesigned agent system that follows the notebook pattern using LangGraph/LangChain instead of ADK.

## Architecture Overview

The new system consists of:

1. **BaseAgentExecutor** - A base class that provides common A2A server functionality
2. **A2AToolClient** - A client for agent-to-agent communication
3. **Individual Agents** - Specialized agents that inherit from the base pattern
4. **Factory Functions** - Static methods to create A2A servers

## Key Components

### BaseAgentExecutor (`base_agent_executor.py`)

A base class that provides:
- Common A2A server setup
- Standardized agent execution
- Factory method `create_agent_a2a_server()` for creating A2A servers
- Error handling and logging

### A2AToolClient (`a2a_tool_client.py`)

A client for agent-to-agent communication that provides:
- Agent discovery and caching
- Message sending with timeout handling
- Response parsing and error handling

### Individual Agents

Each agent follows the same pattern:
- **Agent Class** - Contains the LangGraph agent logic
- **Factory Function** - Creates the A2A server configuration
- **CLI Entry Point** - Provides command-line interface for running the server

### Orchestrator Pattern

The orchestrator agent follows the notebook pattern correctly:
- **Only uses A2AToolClient methods as tools**: `list_remote_agents` and `create_task`
- **Dynamic agent discovery**: Discovers available agents at runtime
- **Task delegation**: Sends tasks to appropriate agents based on their capabilities
- **No custom tools**: Does not create specialized tools for specific agents

## Agent Types

### Research Agent (`research_agent.py`)
- Searches the web for information
- Uses MCP tools for web search
- Returns structured research results

### Scraper Agent (`scraper_agent.py`)
- Scrapes content from URLs
- Uses Playwright MCP tools
- Returns structured scraped data

### Orchestrator Agent (`orchestrator_agent.py`)
- Coordinates between research and scraper agents
- Uses A2AToolClient's `list_remote_agents` and `create_task` methods directly as tools
- Provides intelligent task orchestration following the notebook pattern
- Discovers available agents dynamically and delegates tasks appropriately

## Usage

### Running Individual Agents

Each agent can be run independently:

```bash
# Research Agent
python -m lingua_scraper.agents.research_agent --host localhost --port 5001

# Scraper Agent
python -m lingua_scraper.agents.scraper_agent --host localhost --port 5002

# Orchestrator Agent
python -m lingua_scraper.agents.orchestrator_agent --host localhost --port 5000
```

### Testing the System

Run the test script to verify all components work:

```bash
python -m lingua_scraper.agents.test_agents
```

### Programmatic Usage

```python
from lingua_scraper.agents import ResearchAgent, ScraperAgent, OrchestratorAgent

# Create agents
research_agent = ResearchAgent()
scraper_agent = ScraperAgent()
orchestrator_agent = OrchestratorAgent(remote_agent_addresses=[
    "http://localhost:5001",
    "http://localhost:5002"
])

# Use agents directly
research_result = await research_agent.search("Find Python tutorials")
scraping_result = await scraper_agent.scrape_urls("https://example.com")
orchestration_result = await orchestrator_agent.process_query("Find real estate agents")
```

## Key Improvements

1. **Simplified Architecture** - Removed complex RemoteAgentConnections class
2. **Consistent Pattern** - All agents follow the same structure
3. **Better Separation** - Each agent is self-contained
4. **Easier Testing** - Agents can be tested independently
5. **Cleaner Code** - Removed streaming complexity (can be added back later)
6. **Factory Pattern** - Easy server creation with `create_agent_a2a_server()`

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for all agents

### MCP Servers

The agents expect MCP servers to be running:
- DuckDuckGo MCP server on `http://localhost:8000/mcp` (for research agent)
- Playwright MCP server on `http://localhost:8001/mcp` (for scraper agent)

## Launch Configurations

Each agent maintains its own launch configuration for easy debugging and development. The agents are designed to be run independently, making it easy to:

- Debug individual agents
- Scale agents independently
- Deploy agents to different servers
- Test agents in isolation

## Future Enhancements

1. **Streaming Support** - Can be added back to BaseAgentExecutor
2. **Authentication** - Add security for production use
3. **Monitoring** - Add observability and metrics
4. **Configuration Management** - Centralized configuration
5. **Health Checks** - Agent health monitoring 