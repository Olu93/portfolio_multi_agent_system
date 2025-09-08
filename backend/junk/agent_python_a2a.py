import asyncio
import logging
import os
from typing import Literal
import click
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from python_a2a import A2AServer, AgentSkill, skill, agent, run_server, TaskStatus, TaskState
from langchain_mcp_adapters.client import MultiServerMCPClient
from a2a_servers.config_loader import load_agent_config, load_model_config, load_prompt_config
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
import requests
logger = logging.getLogger(__name__)

# Global cache to track agent registration status
_registration_cache = {}

# Initialize memory saver for LangGraph
memory = InMemorySaver()

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = Field(default='input_required', description=(
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    ))
    message: str = Field(description="The message to respond with")


@agent(
    name="Dynamic Agent",
    description="A configurable agent with MCP tools",
    version="1.0.0"
)
class DynamicAgent(A2AServer):
    """A dynamic agent that can be configured with different tools and prompts."""

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name
        self.agent_config = load_agent_config(agent_name)
        self.model_config = load_model_config(self.agent_config.get("model", "default"))
        self.prompt_config = load_prompt_config(self.agent_config.get("prompt_file", f"{agent_name}.txt"))
        self.meta_prompt = self.agent_config.get("meta_prompt", "You are a helpful assistant that can use multiple tools")

        # Initialize the model once
        
        # Initialize MCP client and tools
        self.mcp_client = None
        self.tools = []
        self.skills = []

    async def initialize(self):
        self.model = init_chat_model(
            self.model_config["name"], 
            **self.model_config["parameters"], 
            model_provider=self.model_config["provider"]
        )
        self.tools = await self._initialize_tools()
        self.skills = await self._initialize_skills()
        
        # Create the LangGraph react agent for reasoning
        self.graph = create_react_agent(
            name=self.agent_name,
            model=self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.meta_prompt + "\n\n" + self.prompt_config,
            response_format=ResponseFormat,
        )

        
        # Set up the agent card with dynamic configuration
        self.agent_card.name = self.agent_config["name"]
        self.agent_card.description = self.agent_config["description"]
        self.agent_card.url = self.agent_config["agent_url"]
        self.agent_card.capabilities = {
            "streaming": True,
            "pushNotifications": False,
            "mcp_tools": True,
            "google_a2a_compatible": True
        }
        self.agent_card.skills = self.skills
        self.agent_card.default_output_modes = ["text/plain", "application/json", "text/event-stream"]
    

    async def _initialize_skills(self):
        tool_config = self.agent_config.get("tools", [])
        return [
            AgentSkill(
                id=tool["name"],
                name=tool["name"],
                description=tool["description"],
                tags=tool["tags"],
                examples=tool["examples"],
            ) for tool in tool_config
        ]
    
    async def _initialize_tools(self):
        """Initialize MCP tools from configuration."""
        try:
            tool_config = self.agent_config.get("tools", [])
            tool_config_dict = {tool["name"]: tool["mcp_server"] for tool in tool_config}
            
            # Validate tool URLs
            for tool_name, tool_info in tool_config_dict.items():
                if not ("/mcp" in tool_info["url"] or "/sse" in tool_info["url"]):
                    logger.warning(
                        f"Tool {tool_name} is not using the correct URL format. "
                        f"Please use the correct URL format. The URL is {tool_info['url']}"
                    )
            
            # Initialize MCP client
            self.mcp_client = MultiServerMCPClient(tool_config_dict)
            
            # Get tools from the MCP client
            return await self.mcp_client.get_tools()
            
        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            self.mcp_client = None
            return []
    

    
    def handle_task(self, task) -> TaskStatus:
        """Handle incoming tasks using the configured prompt and tools."""
        import asyncio
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.handle_task_async(task))
        return res

    async def handle_task_async(self, task) -> TaskStatus:
        """Async implementation of task handling."""
        try:
            # Get user input from the task
            message_data = task.message or {}
            content = message_data.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else ""
            
            if not text:
                return TaskStatus(
                    state=TaskState.INPUT_REQUIRED,
                    message={"role": "agent", "content": {"type": "text", "text": "Please provide a question or request."}}
                )
            
            # Use the LangGraph react agent for reasoning and tool execution
            inputs = {'messages': [('user', text)]}
            config = {'configurable': {'thread_id': task.id}}
            
            # Use ainvoke (async) since our tools are async
            result = await self.graph.ainvoke(inputs, config)
            messages = result.get('messages', [])
            
            if messages:
                final_message = messages[-1]
                response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                # Check if the response indicates completion or requires more input
                if any(keyword in response_text.lower() for keyword in ['complete', 'finished', 'done', 'answer']):
                    # Task is complete
                    task.artifacts = [{
                        "parts": [{"type": "text", "text": response_text}]
                    }]
                    return TaskStatus(state=TaskState.COMPLETED)
                else:
                    # Task requires more input
                    return TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"type": "text", "text": response_text}}
                    )
            else:
                return TaskStatus(
                    state=TaskState.INPUT_REQUIRED,
                    message={"role": "agent", "content": {"type": "text", "text": "Unable to process request. Please try again."}}
                )
                
        except Exception as e:
            logger.error(f"Error handling task: {e}")
            return TaskStatus(
                state=TaskState.FAILED,
                message={"role": "agent", "content": {"type": "text", "text": f"Error processing request: {str(e)}"}}
            )


async def create_sub_agent(agent_name: str) -> DynamicAgent:
    """
    Create a sub agent for the main agent based on the agents.yml file.
    """
    try:
        agent = DynamicAgent(agent_name)
        await agent.initialize()
        return agent
    except Exception as e:
        logger.error(f"Error creating agent {agent_name}: {e}")
        raise e


async def periodic_registration_check(agent: DynamicAgent, interval_seconds: int = None):
    """
    Periodically check agent registration status and maintain health.
    """
    if interval_seconds is None:
        interval_seconds = int(os.getenv("AGENT_HEARTBEAT_INTERVAL", "30"))
    
    logger.info(f"Starting periodic registration check for agent {agent.agent_card.name} with {interval_seconds}s interval")
    
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            # Run the registration check and heartbeat in a thread pool since requests is blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, check_and_maintain_registration, agent)
            
        except asyncio.CancelledError:
            logger.info(f"Periodic registration check for agent {agent.agent_card.name} cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic registration check for agent {agent.agent_card.name}: {e}")
            continue


def check_if_agent_registered(agent_card) -> bool:
    """Check if the agent is already registered with the registry."""
    REGISTRY_URL = os.getenv("REGISTRY_URL")
    try:
        import requests
        # Check if agent is already registered by URL
        response = requests.post(f"{REGISTRY_URL}/registry/lookup", json={"url": agent_card.url})
        if response.status_code == 200:
            logger.info(f"Agent {agent_card.name} is already registered")
            return True
        elif response.status_code == 404:
            logger.info(f"Agent {agent_card.name} is not registered")
            return False
        else:
            logger.warning(f"Unexpected response when checking agent registration: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking if agent {agent_card.name} is registered: {e}")
        return False


def check_and_maintain_registration(agent):
    """
    Unified function to check registration status and maintain agent health.
    """
    REGISTRY_URL = os.getenv("REGISTRY_URL")
    if not REGISTRY_URL:
        logger.error("REGISTRY_URL environment variable not set")
        return
    
    cache_key = agent.agent_card.url
    cached_registration = _registration_cache.get(cache_key, False)
    
    try:
        
        
        # Step 1: Check if agent is already registered (skip if we know we're registered)
        if not cached_registration:
            is_registered = check_if_agent_registered(agent.agent_card)
        else:
            is_registered = True
            logger.info(f"Agent {agent.agent_card.name} registration status cached as registered")
        
        if not is_registered:
            # Step 2: Attempt to register if not already registered
            logger.info(f"Agent {agent.agent_card.name} not found in registry, attempting registration...")
            try:
                response = requests.post(f"{REGISTRY_URL}/registry/register", json=agent.agent_card.model_dump(mode='python'))
                if response.status_code == 201:
                    logger.info(f"Agent {agent.agent_card.name} registered successfully")
                    is_registered = True
                    # Cache successful registration
                    _registration_cache[cache_key] = True
                else:
                    logger.error(f"Failed to register agent {agent.agent_card.name}: {response.status_code}")
                    return  # Don't send heartbeat if registration failed
            except Exception as e:
                logger.error(f"Error registering agent {agent.agent_card.name}: {e}")
                return  # Don't send heartbeat if registration failed
        
        # Step 3: Send heartbeat only if agent is registered
        if is_registered:
            try:
                response = requests.post(f"{REGISTRY_URL}/registry/heartbeat", json={"url": agent.agent_card.url})
                if response.status_code == 200:
                    logger.info(f"Heartbeat sent successfully for agent {agent.agent_card.name}")
                else:
                    logger.warning(f"Failed to send heartbeat for agent {agent.agent_card.name}: {response.status_code}")
                    # If heartbeat fails, the agent might have been removed from registry
                    # Clear the cache to force a re-check next time
                    if response.status_code == 404:
                        _registration_cache[cache_key] = False
                        logger.info(f"Agent {agent.agent_card.name} appears to have been removed from registry, will re-register next cycle")
            except Exception as e:
                logger.error(f"Error sending heartbeat for agent {agent.agent_card.name}: {e}")
        else:
            logger.warning(f"Agent {agent.agent_card.name} is not registered, skipping heartbeat")
            
    except Exception as e:
        logger.error(f"Unexpected error in check_and_maintain_registration for agent {agent.agent_card.name}: {e}")


async def setup_and_run(agent_name: str):
    # Create the agent
    agent = await create_sub_agent(agent_name)
    
    # Start background task for periodic registration
    registration_task = asyncio.create_task(periodic_registration_check(agent))
    
    # Return both for the main function to handle
    return agent, registration_task


@click.command()
@click.option("--agent-name", default="web_search_agent", help="Name of the agent to run")
def run_agent_server(agent_name: str):
    """Run an A2A agent server using the python-a2a library."""
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 10020))


    
    # Run the async setup
    agent, registration_task = asyncio.run(setup_and_run(agent_name))
    
    try:
        # Run the server synchronously (this is what python-a2a expects)
        run_server(agent, host=HOST, port=PORT)
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        # Clean up the background task
        if not registration_task.done():
            registration_task.cancel()
            # We need to run the cleanup in an event loop
            asyncio.run(cleanup_task(registration_task))
            logger.info(f"Stopped periodic registration check for agent {agent.agent_card.name}")


async def cleanup_task(task):
    """Clean up the background task."""
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    run_agent_server()
