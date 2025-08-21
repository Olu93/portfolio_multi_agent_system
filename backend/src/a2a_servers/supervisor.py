# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (a2a package)
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional, List, Callable, AsyncIterable

import click
import httpx
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
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
import uvicorn
from langchain.chat_models import init_chat_model

from a2a_servers.config_loader import (
    load_agent_config,
    load_model_config,
    load_prompt_config,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.structured import StructuredTool

logger = logging.getLogger(__name__)

# Initialize memory saver for LangGraph
memory = InMemorySaver()

# --- Config helper -----------------------------------------------------------

# --- Registry Client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")
REFRESH_SECS = int(os.getenv("DISCOVERY_REFRESH_SECS", "30"))


async def fetch_agents() -> List[AgentCard]:
    logger.info(f"Fetching agents from registry at {REGISTRY_URL}")
    async with httpx.AsyncClient(timeout=10) as s:
        r = await s.get(f"{REGISTRY_URL}/registry/agents")
        r.raise_for_status()
        agents = [AgentCard(**a) for a in r.json()]
        logger.info(f"Discovered {len(agents)} agents from registry")
        return agents


# --- Tool factory from AgentCards -------------------------------------------


async def build_tools_from_registry(
    allow_urls: set, allow_caps: set
) -> List[StructuredTool]:
    logger.info(
        f"Building tools from registry with allow_urls={allow_urls}, allow_caps={allow_caps}"
    )

    def _safe_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()

    def _create_tool_for_card(card: AgentCard) -> StructuredTool:
        """Create a tool function for a specific agent card"""

        async def tool_impl(content: str, context: Dict[str, Any] = {}):
            """
            content: str - the content of the message to send to the agent
            context: Dict[str, Any] - the context of the message

            The context is a dictionary of key-value pairs that are passed to the agent.
            It is used to pass information to the agent, such as the task_id, message_id, session_id, conversation_id, and other metadata.
            The task_id is a unique identifier for the task, the message_id is a unique identifier for the message, and the session_id is a unique identifier for the session.
            The task_id, message_id, and session_id are used to identify the task, message, and session in the A2A protocol.
            The conversation_id is a unique identifier for the conversation. The conversation_id is used to identify the conversation in the A2A protocol.
            """

            logger.debug(f"Tool {card.name} called with content: {content[:100]}...")

            # For now, return a placeholder response
            # TODO: Implement proper A2A client communication
            response_text = f"Tool {card.name} would process: {content[:100]}... (URL: {card.url})"
            logger.info(f"Tool {card.name} returned response: {response_text}")
            return response_text

        tool_name = _safe_name(card.name) or _safe_name(card.url)
        desc_caps = ", ".join(sorted((card.capabilities or {}).keys()))
        summary = (
            f"{card.description or 'A2A agent'}. Caps: {desc_caps or 'unspecified'}"
        )

        logger.debug(
            f"Creating tool '{tool_name}' for agent '{card.name}' at {card.url}"
        )
        return StructuredTool.from_function(
            coroutine=tool_impl,
            name=tool_name,
            description=summary,
        )

    cards = await fetch_agents()
    logger.info(f"Processing {len(cards)} agent cards")

    if allow_urls:
        original_count = len(cards)
        cards = [c for c in cards if c.url in allow_urls]
        logger.info(f"Filtered by allow_urls: {original_count} -> {len(cards)} agents")

    if allow_caps:
        original_count = len(cards)
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]
        logger.info(f"Filtered by allow_caps: {original_count} -> {len(cards)} agents")

    tools: List[StructuredTool] = []
    for card in cards:
        tools.append(_create_tool_for_card(card))

    logger.info(f"Successfully created {len(tools)} tools from registry")
    return tools


# --- LangGraph supervisor agent ----------------------------------------------


async def build_supervisor_graph(agent_name=None):
    logger.info("Building supervisor graph")

    # Load configuration directly
    logger.debug(f"Loading configuration for supervisor agent: {agent_name}")
    agent_cfg = load_agent_config(agent_name)
    logger.debug(f"Agent config loaded: {agent_cfg}")

    model_cfg = load_model_config(agent_cfg.get("model", "default"))
    logger.debug(f"Model config loaded: {model_cfg}")

    prompt = load_prompt_config(agent_cfg.get("prompt_file", "supervisor_agent.txt"))
    logger.debug(f"Prompt loaded, length: {len(prompt)} characters")

    # optional routing limits
    allow_urls = set(agent_cfg.get("allow_urls", []) or [])
    allow_caps = set(agent_cfg.get("allow_caps", []) or [])
    logger.info(f"Routing limits: allow_urls={allow_urls}, allow_caps={allow_caps}")

    logger.info(f"Initializing chat model: {model_cfg.get('name', 'unknown')}")
    model = init_chat_model(
        model_cfg["name"],
        **model_cfg.get("parameters", {}),
        model_provider=model_cfg.get("provider"),
    )

    logger.info("Building tools from registry")
    tools = await build_tools_from_registry(allow_urls, allow_caps)

    logger.info("Creating react agent with tools and prompt")
    return create_react_agent(model, tools, prompt=prompt, name=agent_name, checkpointer=InMemorySaver())


# --- Supervisor Agent Class -------------------------------------------------


class SupervisorAgent:
    """Supervisor agent that orchestrates other agents using LangGraph."""

    def __init__(self, name, model, tools, prompt):
        self.name = name
        self.model = model
        self.tools = tools
        self.prompt = prompt

        self.graph = create_react_agent(
            name=name,
            model=self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=prompt,
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
        messages = current_state.values.get('messages', [])
        
        if messages:
            final_message = messages[-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': response_text,
            }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }


# --- Base Agent Executor ---------------------------------------------------


class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: SupervisorAgent,
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
        config = {'configurable': {'thread_id': task.context_id}}
        try:
            async for item in self.agent.stream(query, config):
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
                        name=self.artifact_name,
                    )
                    await updater.complete()
                    break

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


# --- Agent Creation ---------------------------------------------------------


async def create_supervisor_agent(agent_name: str):
    """
    Create a supervisor agent based on the agents.yml file.
    """
    try:
        agent_config = load_agent_config(agent_name)
        model_config = load_model_config(agent_config.get("model", "default"))
        meta_prompt = agent_config.get("meta_prompt", "You are a helpful supervisor that can orchestrate multiple agents")
        prompt_config = load_prompt_config(agent_config.get("prompt_file", f"{agent_name}.txt"))

        # Build tools from registry
        allow_urls = set(agent_config.get("allow_urls", []) or [])
        allow_caps = set(agent_config.get("allow_caps", []) or [])
        tools = await build_tools_from_registry(allow_urls, allow_caps)

        # Initialize the model
        model = init_chat_model(
            model_config["name"], 
            **model_config["parameters"], 
            model_provider=model_config["provider"]
        )

        # Create the supervisor agent
        prompt = f"{meta_prompt}\n\n{prompt_config}"
        agent = SupervisorAgent(agent_name, model, tools, prompt)

        # Create agent card
        skills = [
            AgentSkill(
                name=skill.get("name", ""),
                description=skill.get("description", ""),
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
            )
            for skill in agent_config.get("skills", [])
        ]
        description = agent_config.get("description", "")
        capabilities = agent_config.get("capabilities", {})
        agent_card = AgentCard(
            name=agent_name,
            description=description,
            url=agent_config["agent_url"],
            capabilities=capabilities,
            skills=skills,
            default_output_modes=["application/json"],
        )

        # Create executor with custom parameters
        executor = BaseAgentExecutor(
            agent=agent,
            status_message="Processing request...",
            artifact_name="supervisor_response",
        )
        
        request_handler = DefaultRequestHandler(  
            agent_executor=executor,  
            task_store=InMemoryTaskStore()  
        ) 
        
        return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        
    except Exception as e:
        logger.error(f"Error creating supervisor agent {agent_name}: {e}")
        raise e


# --- CLI entrypoint -----------------------------------------------------------


@click.command()
@click.option(
    "--agent-name",
    default="supervisor",
    help="Name of the supervisor agent config to load",
)
def run_supervisor(agent_name: str):
    """Run the Supervisor agent server."""
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 10020))

    async def run_with_background_tasks():
        app = await create_supervisor_agent(agent_name)
        fastapi_app = app.build()
        return fastapi_app
    
    # Run the async function and get the FastAPI app
    fastapi_app = asyncio.run(run_with_background_tasks())
    
    uvicorn.run(
        fastapi_app,
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    run_supervisor()
