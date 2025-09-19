import asyncio
import contextlib
import logging
import os
import time
from typing import Any, AsyncIterable, Literal
from urllib.parse import urlparse

import click
import requests
import uvicorn
from a2a.types import (
    AgentCard,
    Artifact,
    # DataPart,
    TaskState,
)
from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from a2a_servers.a2a_client import (
    BaseAgent,
    ChunkMetadata,
    ChunkResponse,
)
from a2a_servers.config_loader import (
    load_agent_config,
)

logger = logging.getLogger(__name__)

memory = InMemorySaver()

# Global cache to track agent registration status
# This prevents repeated registration attempts during the same agent session
_registration_cache = {}

# Cache expiration time (in seconds) - how long to trust cached registration status
CACHE_EXPIRY_SECONDS = int(os.getenv("REGISTRATION_CACHE_EXPIRY", "300"))  # 5 minutes default


def _is_cache_valid(cache_entry):
    """Check if a cache entry is still valid based on timestamp."""
    if isinstance(cache_entry, dict):
        timestamp = cache_entry.get("timestamp", 0)
        return (time.time() - timestamp) < CACHE_EXPIRY_SECONDS
    return False


def _get_cached_registration(cache_key):
    """Get cached registration status if it's still valid."""
    cache_entry = _registration_cache.get(cache_key)
    if cache_entry and _is_cache_valid(cache_entry):
        return cache_entry.get("registered", False)
    return None  # Cache expired or doesn't exist


def _set_cached_registration(cache_key, registered: bool):
    """Set cached registration status with timestamp."""
    _registration_cache[cache_key] = {
        "registered": registered,
        "timestamp": time.time(),
    }


def _cleanup_expired_cache():
    """Remove expired entries from the registration cache."""
    current_time = time.time()
    expired_keys = []

    for key, entry in _registration_cache.items():
        if not _is_cache_valid(entry):
            expired_keys.append(key)

    for key in expired_keys:
        del _registration_cache[key]
        logger.debug(f"Removed expired cache entry for {key}")

    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class SubAgent(BaseAgent):
    """CurrencyAgent - a specialized assistant for currency convesions."""

    SUPPORTED_CONTENT_TYPES = [
        "text",
        "text/plain",
        "text/event-stream",
        "application/json",
    ]

    FORMAT_INSTRUCTION = (
        "Set response status to input_required if the user needs to provide more information to complete the request."
        "Set response status to error if there is an error while processing the request."
        "Set response status to completed if the request is complete."
    )

    async def build_tools(self) -> list[StructuredTool]:
        tool_config = self.agent_config.get("tools", [])
        tool_config_dict = {tool["name"]: tool["mcp_server"] for tool in tool_config}
        for tool in tool_config_dict:
            if not ("/mcp" in tool_config_dict[tool]["url"] or "/sse" in tool_config_dict[tool]["url"]):
                logger.warning(
                    f"Tool {tool} is not using the correct URL format. Please use the correct URL format. The URL is {tool_config_dict[tool]['url']}"
                )
        client = MultiServerMCPClient(tool_config_dict)
        tools = await client.get_tools()
        return tools

    async def build_graph(self) -> CompiledStateGraph:
        return create_react_agent(
            name=self.name,
            model=self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.prompt,
            response_format=(self.FORMAT_INSTRUCTION, ChunkResponse),
        )

    async def stream(self, artifacts: list[Artifact], context_id: str, task_id: str) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [("user", self._extract_parts(artifact)) for artifact in artifacts]}
        config = {"configurable": {"thread_id": context_id}}
        latest_tool_call = None
        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls and len(message.tool_calls) > 0:
                latest_tool_call = message.tool_calls[0].get("name")
                yield ChunkResponse(
                    status=TaskState.working,
                    content=f"{self.name} is calling the tool... {latest_tool_call}",
                    tool_name=latest_tool_call,
                    metadata=ChunkMetadata(message_type="tool_call", step_number=len(item["messages"])),
                )
            elif isinstance(message, ToolMessage):
                yield ChunkResponse(
                    status=TaskState.working,
                    content=f"{self.name} executed the tool successfully... {latest_tool_call}",
                    tool_name=latest_tool_call,
                    metadata=ChunkMetadata(message_type="tool_execution", step_number=len(item["messages"])),
                )

        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        if structured_response and isinstance(structured_response, ChunkResponse):
            yield structured_response
        else:
            yield ChunkResponse(
                status=TaskState.failed,
                content=f"{self.name} is unable to process your request at the moment. Please try again.",
                tool_name=None,
                metadata=ChunkMetadata(message_type="error", error="no_messages", step_number=len(item["messages"])),
            )


async def create_sub_agent(agent_name: str, url: str):
    """
    Create a sub agent for the main agent based on the agents.yml file in which all the agents are registered.
    """
    try:
        agent_config = load_agent_config(agent_name)
        agent = await SubAgent(agent_name, url, agent_config).build()
        return agent
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

            # Clean up expired cache entries periodically
            _cleanup_expired_cache()

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
    cached_registration = _get_cached_registration(cache_key)

    try:
        # Step 1: Always check if agent is actually registered (don't trust cache blindly)
        # This ensures we detect when agents are removed from the registry
        is_registered = check_if_agent_registered(agent_card)

        # Update cache based on actual registry status
        if is_registered:
            _set_cached_registration(cache_key, True)
            logger.info(f"Agent {agent_card.name} confirmed as registered in registry {REGISTRY_URL} with agent url {agent_card.url}")
        else:
            _set_cached_registration(cache_key, False)
            logger.info(f"Agent {agent_card.name} not found in registry {REGISTRY_URL} with agent url {agent_card.url}")

        if not is_registered:
            # Step 2: Attempt to register if not already registered
            logger.info(f"Agent {agent_card.name} not found in registry {REGISTRY_URL} with agent url {agent_card.url}, attempting registration...")
            try:
                response = requests.post(
                    f"{REGISTRY_URL}/registry/register",
                    json=agent_card.model_dump(mode="python"),
                )
                if response.status_code == 201:
                    logger.info(f"Agent {agent_card.name} registered successfully in registry {REGISTRY_URL} with agent url {agent_card.url}")
                    is_registered = True
                    # Cache successful registration
                    _set_cached_registration(cache_key, True)
                else:
                    logger.error(
                        f"Failed to register agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}: {response.status_code}"
                    )
                    return  # Don't send heartbeat if registration failed
            except Exception as e:
                logger.error(f"Error registering agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}: {e}")
                return  # Don't send heartbeat if registration failed

        # Step 3: Send heartbeat only if agent is registered
        if is_registered:
            try:
                response = requests.post(f"{REGISTRY_URL}/registry/heartbeat", json={"url": agent_card.url})
                if response.status_code == 200:
                    logger.info(f"Heartbeat sent successfully for agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}")
                else:
                    logger.warning(
                        f"Failed to send heartbeat for agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}: {response.status_code}"
                    )
                    # If heartbeat fails, the agent might have been removed from registry
                    # Clear the cache to force a re-check next time
                    if response.status_code == 404:
                        _set_cached_registration(cache_key, False)
                        logger.info(f"Agent {agent_card.name} appears to have been removed from registry, will re-register next cycle")
            except Exception as e:
                logger.error(f"Error sending heartbeat for agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}: {e}")
        else:
            logger.warning(f"Agent {agent_card.name} is not registered, skipping heartbeat")

    except Exception as e:
        logger.error(
            f"Unexpected error in check_and_maintain_registration for agent {agent_card.name} in registry {REGISTRY_URL} with agent url {agent_card.url}: {e}"
        )


@click.command()
@click.option("--agent-name", default="web_search_agent", help="Name of the agent to run")
def run_agent_server(agent_name: str):
    URL = os.getenv("URL", "http://0.0.0.0:10020")
    HOST = urlparse(URL).hostname
    PORT = int(urlparse(URL).port)

    async def run_with_background_tasks():
        agent = await create_sub_agent(agent_name, url=URL)
        starlette_app = agent.http_handler
        agent_card = agent.agent_card
        fastapi_app = starlette_app

        # Add background task for periodic registration
        registration_task = None

        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI):
            nonlocal registration_task
            # Startup
            # Do initial registration attempt and health check
            logger.info(f"Attempting initial registration for agent {agent_card.name}")
            await asyncio.get_event_loop().run_in_executor(None, check_and_maintain_registration, agent_card)

            # Start the periodic registration task
            registration_task = asyncio.create_task(periodic_registration_check(agent_card))
            interval = os.getenv("AGENT_HEARTBEAT_INTERVAL", "30")
            logger.info(f"Started periodic registration check for agent {agent_card.name} with {interval}s interval")

            yield

            # Shutdown
            if registration_task and not registration_task.done():
                registration_task.cancel()
                try:
                    await registration_task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Stopped periodic registration check for agent {agent_card.name}")

        # Set the lifespan context manager
        fastapi_app.router.lifespan_context = lifespan

        return fastapi_app

    # Run the async function and get the FastAPI app
    fastapi_app = asyncio.run(run_with_background_tasks())
    logger.info(f"Access agent at: {URL}")
    logger.debug(f"HOST: {HOST}")
    logger.debug(f"PORT: {PORT}")

    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=PORT,
    )


if __name__ == "__main__":
    run_agent_server()
