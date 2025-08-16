import asyncio
import logging
import os
from typing import Any, AsyncIterable, Literal
import click
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.errors import GraphRecursionError
from langchain.chat_models.base import BaseChatModel
from python_a2a import AgentSkill
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    # DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, ToolMessage
import requests
import uvicorn
from langchain.chat_models import init_chat_model

from a2a_servers.config_loader import load_agent_config, load_model_config, load_prompt_config
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

memory = InMemorySaver()

# Global cache to track agent registration status
# This prevents repeated registration attempts during the same agent session
_registration_cache = {}


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class Agent:
    """CurrencyAgent - a specialized assistant for currency convesions."""


    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, model, tools, prompt):
        self.model = model
        self.tools = tools
        self.prompt = prompt

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.prompt,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the request...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Executing the tool...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']



class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: Agent,
        status_message: str = "Processing request...",
        artifact_name: str = "response",
    ):
        """Initialize a generic LangGraph agent executor.

        Args:
            agent: The LangGraph agent instance that implements AgentInterface
            status_message: Message to display while processing
            artifact_name: Name for the response artifact
        """
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
                    break

        except GraphRecursionError as e:
            logger.error(f"Error: {e!s}")
            all_parts = []
            for msg in task.history:
                logger.error(f"Message: {msg!s}")
                all_parts.extend(msg.parts)
            all_parts.append(Part(root=TextPart(text=f"Error: {e!s}")))

            await updater.update_status(
                TaskState.failed,
                new_agent_parts_message(all_parts, task.contextId, task.id),
                final=True,
            )
        except Exception as e:
            logger.error(f"Error: {e!s}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.contextId, task.id),
                final=True,
            )

    async def execute_streaming(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute with streaming - not supported by this agent."""
        # Since streaming is disabled, we'll just call the regular execute method
        # and let the framework handle the streaming response
        await self.execute(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Implementation for cancelling tasks
        pass


async def create_sub_agent(agent_name: str):
    """
    Create a sub agent for the main agent based on the agents.yml file in which all the agents are registered.
    """
    try:


        capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
        agent_config = load_agent_config(agent_name)
        tool_config = agent_config.get("tools", [])
        model_config = load_model_config(agent_config.get("model", "default"))	
        meta_prompt = agent_config.get("meta_prompt", "You are a helpful assistant that can use multiple tools")
        prompt_config = load_prompt_config(agent_config.get("prompt_file", f"{agent_name}.txt"))

        tool_config_dict = {tool["name"]: tool["mcp_server"] for tool in tool_config}

        # TODO: Add skills from the agent config
        skills = [
            AgentSkill(
                id=tool["name"],
                name=tool["name"],
                description=tool["description"],
                tags=tool["tags"],
                examples=tool["examples"],
            ) for tool in tool_config
        ]


        agent_card = AgentCard(
            name=agent_config["name"],
            description=agent_config["description"],
            # url=f"http://{host}:{port}/",
            url=agent_config["agent_url"],
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        model = init_chat_model(model_config["name"], **model_config["parameters"], model_provider=model_config["provider"])


        for tool in tool_config_dict:
            if not ("/mcp" in tool_config_dict[tool]["url"] or "/sse" in tool_config_dict[tool]["url"]):
                logger.warning(f"Tool {tool} is not using the correct URL format. Please use the correct URL format. The URL is {tool_config_dict[tool]['url']}")

        client = MultiServerMCPClient(tool_config_dict)
        tools = await client.get_tools()

        prompt = f"{meta_prompt}\n\n{prompt_config}"

        agent = Agent(model, tools, prompt)

        # Create executor with custom parameters
        executor = BaseAgentExecutor(
            agent=agent,
            status_message="Processing request...",
            artifact_name="response",
        )
        request_handler = DefaultRequestHandler(  
            agent_executor=executor,  
            task_store=InMemoryTaskStore()  
        ) 
        return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    except Exception as e:
        logger.error(f"Error: {e!s}")
        raise e


async def periodic_registration_check(agent_card: AgentCard, interval_seconds: int = None):
    """
    Periodically check agent registration status and maintain health.
    This runs as a background task with the following flow:
    1. Check if agent is registered
    2. If not registered, attempt to register
    3. If registered, send heartbeat to maintain health
    """
    if interval_seconds is None:
        interval_seconds = int(os.getenv("AGENT_HEARTBEAT_INTERVAL", "30"))
    
    logger.info(f"Starting periodic registration check for agent {agent_card.name} with {interval_seconds}s interval")
    
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            # Run the registration check and heartbeat in a thread pool since requests is blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, check_and_maintain_registration, agent_card)
            
        except asyncio.CancelledError:
            logger.info(f"Periodic registration check for agent {agent_card.name} cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic registration check for agent {agent_card.name}: {e}")
            # Continue running even if there's an error
            continue


def check_if_agent_registered(agent_card: AgentCard) -> bool:
    """
    Check if the agent is already registered with the registry.
    """
    REGISTRY_URL = os.getenv("REGISTRY_URL")
    try:
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

def check_and_maintain_registration(agent_card: AgentCard):
    """
    Unified function to check registration status and maintain agent health.
    
    Flow:
    1. Check if agent is already registered
    2. If not registered, attempt to register
    3. If registered (or newly registered), send heartbeat
    
    This prevents unnecessary repeated registration attempts and only sends
    heartbeats when the agent is actually registered.
    """
    REGISTRY_URL = os.getenv("REGISTRY_URL")
    if not REGISTRY_URL:
        logger.error("REGISTRY_URL environment variable not set")
        return
    
    cache_key = agent_card.url
    cached_registration = _registration_cache.get(cache_key, False)
    
    try:
        # Step 1: Check if agent is already registered (skip if we know we're registered)
        if not cached_registration:
            is_registered = check_if_agent_registered(agent_card)
        else:
            is_registered = True
            logger.debug(f"Agent {agent_card.name} registration status cached as registered")
        
        if not is_registered:
            # Step 2: Attempt to register if not already registered
            logger.info(f"Agent {agent_card.name} not found in registry, attempting registration...")
            try:
                response = requests.post(f"{REGISTRY_URL}/registry/register", json=agent_card.model_dump(mode='python'))
                if response.status_code == 201:
                    logger.info(f"Agent {agent_card.name} registered successfully")
                    is_registered = True
                    # Cache successful registration
                    _registration_cache[cache_key] = True
                else:
                    logger.error(f"Failed to register agent {agent_card.name}: {response.status_code}")
                    return  # Don't send heartbeat if registration failed
            except Exception as e:
                logger.error(f"Error registering agent {agent_card.name}: {e}")
                return  # Don't send heartbeat if registration failed
        
        # Step 3: Send heartbeat only if agent is registered
        if is_registered:
            try:
                response = requests.post(f"{REGISTRY_URL}/registry/heartbeat", json={"url": agent_card.url})
                if response.status_code == 200:
                    logger.debug(f"Heartbeat sent successfully for agent {agent_card.name}")
                else:
                    logger.warning(f"Failed to send heartbeat for agent {agent_card.name}: {response.status_code}")
                    # If heartbeat fails, the agent might have been removed from registry
                    # Clear the cache to force a re-check next time
                    if response.status_code == 404:
                        _registration_cache[cache_key] = False
                        logger.info(f"Agent {agent_card.name} appears to have been removed from registry, will re-register next cycle")
            except Exception as e:
                logger.error(f"Error sending heartbeat for agent {agent_card.name}: {e}")
        else:
            logger.warning(f"Agent {agent_card.name} is not registered, skipping heartbeat")
            
    except Exception as e:
        logger.error(f"Unexpected error in check_and_maintain_registration for agent {agent_card.name}: {e}")


def clear_registration_cache(agent_card: AgentCard = None):
    """
    Clear the registration cache for a specific agent or all agents.
    
    Args:
        agent_card: If provided, clear cache for this specific agent.
                   If None, clear cache for all agents.
    """
    if agent_card is None:
        # Clear all cached registrations
        _registration_cache.clear()
        logger.info("Cleared all agent registration caches")
    else:
        # Clear cache for specific agent
        cache_key = agent_card.url
        if cache_key in _registration_cache:
            del _registration_cache[cache_key]
            logger.info(f"Cleared registration cache for agent {agent_card.name}")


def get_registration_status(agent_card: AgentCard) -> dict:
    """
    Get the current registration status of an agent.
    
    Returns:
        dict: Status information including:
            - is_registered: bool
            - is_cached: bool
            - last_check: str (timestamp)
            - registry_url: str
    """
    cache_key = agent_card.url
    is_cached = _registration_cache.get(cache_key, False)
    
    # Force a fresh check
    is_registered = check_if_agent_registered(agent_card)
    
    return {
        'is_registered': is_registered,
        'is_cached': is_cached,
        'last_check': 'Just checked',
        'registry_url': os.getenv("REGISTRY_URL", "Not set")
    }


@click.command()
@click.option("--agent-name", default="web_search_agent", help="Name of the agent to run")
@click.option("--host", default="0.0.0.0", help="Host to run the agent on")
@click.option("--port", default=10020, help="Port to run the agent on")
@click.option("--log-level", default="info", help="Log level to run the agent on")
def run_agent_server(agent_name: str, host: str, port: int, log_level: str):
    async def run_with_background_tasks():
        app = await create_sub_agent(agent_name)
        fastapi_app = app.build()
        
        # Add background task for periodic registration
        registration_task = None
        
        @fastapi_app.on_event("startup")
        async def startup_event():
            nonlocal registration_task
            # Do initial registration attempt and health check
            logger.info(f"Attempting initial registration for agent {app.agent_card.name}")
            await asyncio.get_event_loop().run_in_executor(None, check_and_maintain_registration, app.agent_card)
            
            # Start the periodic registration task
            registration_task = asyncio.create_task(periodic_registration_check(app.agent_card))
            interval = os.getenv("AGENT_HEARTBEAT_INTERVAL", "30")
            logger.info(f"Started periodic registration check for agent {app.agent_card.name} with {interval}s interval")
        
        @fastapi_app.on_event("shutdown")
        async def shutdown_event():
            if registration_task and not registration_task.done():
                registration_task.cancel()
                try:
                    await registration_task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Stopped periodic registration check for agent {app.agent_card.name}")
        
        return fastapi_app
    
    # Run the async function and get the FastAPI app
    fastapi_app = asyncio.run(run_with_background_tasks())
    
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_agent_server()
