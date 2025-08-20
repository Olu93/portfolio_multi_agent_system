# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (python-a2a)
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional, List, Callable

import click
import httpx
from python_a2a import (
    A2AClient,
    AgentCard,
    A2AServer,
    Task,
    run_server,
    TaskStatus,
    TaskState,
    AgentSkill,
)
from python_a2a.models.message import Message, MessageRole
from python_a2a.models.content import TextContent, ErrorContent
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
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


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

            # Create a dedicated A2AClient for this specific agent
            agent_client = A2AClient(endpoint_url=card.url, google_a2a_compatible=True)

            try:
                # Use the ask method for simple text queries
                logger.info(f"Sending message to {card.url}")
                response = agent_client.send_message(Message(content=TextContent(text=content), role=MessageRole.AGENT, metadata={**context}))
                logger.info(f"Received response from {card.url}")
                logger.debug(
                    f"Tool {card.name} returned response: {str(response)[:200]}..."
                )
                return response
            except Exception as e:
                logger.error(f"Error calling agent {card.name} at {card.url}: {e}")
                return f"Error communicating with agent {card.name}: {str(e)}"

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


# --- Public API ---------------------------------------------------------------
class Supervisor(A2AServer):
    def __init__(self, agent_name=None, host=None, port=None):
        # Set the URL for this supervisor agent
        url = f"http://{host}:{port}"
        self.agent_name = agent_name
        self.agent_cfg = load_agent_config(agent_name)
        skills = [
            AgentSkill(
                name=skill.get("name", ""),
                description=skill.get("description", ""),
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
            )
            for skill in self.agent_cfg.get("skills", [])
        ]
        description = self.agent_cfg.get("description", "")
        capabilities = self.agent_cfg.get("capabilities", {})
        agent_card = AgentCard(
            name=self.agent_name,
            description=description,
            url=self.agent_cfg["agent_url"],
            capabilities=capabilities,
            skills=skills,
            default_output_modes=["application/json"],
            # output_modes=["application/json", "text/plain"],
        )
        super().__init__(agent_card=agent_card)

        self.graph: CompiledStateGraph = None

        self._lock = asyncio.Lock()
        self._last_refresh = 0
        self.host = host
        self.port = port
        logger.info(f"Supervisor A2A agent initialized at {url}")

    def handle_task(self, task: Task):
        """Handle incoming A2A tasks - this is the A2A protocol method."""
        try:
            # Extract content from task
            message_data = task.message or {}
            content = message_data.get("content", {})
            text = content.get("text", "") if isinstance(content, dict) else ""

            logger.info(f"Received A2A task: {text[:100]}...")

            # Validate input
            if not text.strip():
                raise ValueError("Empty or whitespace-only message received")

            # Use the LangGraph orchestration to process the task
            result = asyncio.run(self.orchestrate(text, {
                "task_id": task.id,
                "session_id": task.session_id,
                "message_id": message_data.get("message_id", ""),
                "conversation_id": message_data.get("conversation_id", ""),
                "metadata": task.metadata,
                

            }))

            # Update task with the orchestration results
            if result.get("ok"):
                task.artifacts = [
                    {"parts": [{"type": "text", "text": result["content"]}]}
                ]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            else:
                task.status = TaskStatus(
                    state=TaskState.FAILED, 
                    message={"error": "Orchestration failed"}
                )

        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation error in A2A task: {e}")
            task.artifacts = [{
                "parts": [{"type": "text", "text": f"Validation error: {str(e)}"}]
            }]
            task.status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message={"error": f"Input validation failed: {str(e)}"}
            )

        except asyncio.TimeoutError as e:
            # Handle timeout errors
            logger.error(f"Timeout error in A2A task orchestration: {e}")
            task.artifacts = [{
                "parts": [{"type": "text", "text": "Request timed out. Please try again."}]
            }]
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"error": "Orchestration timeout"}
            )

        except ConnectionError as e:
            # Handle connection errors
            logger.error(f"Connection error in A2A task: {e}")
            task.artifacts = [{
                "parts": [{"type": "text", "text": f"Service unavailable: {str(e)}"}]
            }]
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"error": f"Connection failed: {str(e)}"}
            )

        except Exception as e:
            # Handle unexpected errors
            import traceback
            logger.error(f"Unexpected error handling A2A task: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            task.artifacts = [{
                "parts": [{"type": "text", "text": "An unexpected error occurred. Please try again or contact support."}]
            }]
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"error": f"Unexpected error: {str(e)}"}
            )

        return task

    def get_agent_card(self) -> AgentCard:
        """Get the supervisor's agent card for discovery"""
        return self.agent_card

    async def _ensure_graph(self):
        now = time.time()
        if self.graph is None or (now - self._last_refresh) > REFRESH_SECS:
            logger.debug("Graph needs refresh or initialization")
            async with self._lock:
                if (
                    self.graph is None
                    or (time.time() - self._last_refresh) > REFRESH_SECS
                ):
                    logger.info("Building new supervisor graph")
                    self.graph = await build_supervisor_graph(self.agent_name)
                    self._last_refresh = time.time()
                    logger.info("Supervisor graph built and ready")
                else:
                    logger.debug("Graph was refreshed by another task")

    async def orchestrate(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Internal method for orchestrating agents via LangGraph."""
        logger.info(f"Orchestrating with LangGraph: {content[:100]}...")
        logger.debug(f"Task context: {context}")

        await self._ensure_graph()

        inputs = {"messages": [("user", content)], "context": context or {}}
        logger.debug("Invoking supervisor graph")

        config = {"configurable": {"thread_id": context.get("task_id", uuid.uuid4())}}
        result = await self.graph.ainvoke(inputs, config=config)
        logger.debug(f"Graph invocation completed, result keys: {list(result.keys())}")

        msgs = result.get("messages", [])
        final = msgs[-1].content if msgs else ""
        logger.info(f"Task completed, final response length: {len(final)}")

        return {"ok": True, "content": final, "raw": result}

    def run_server(self):
        """Run the supervisor as an A2A server"""
        logger.info(f"Starting Supervisor A2A server on {self.host}:{self.port}")
        # Use the imported run_server function from python_a2a
        run_server(self, host=self.host, port=self.port)


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

    logger.info(f"Starting Supervisor agent '{agent_name}'")
    sup = Supervisor(agent_name=agent_name, host=HOST, port=PORT)
    asyncio.run(sup._ensure_graph())
    logger.info("Supervisor is initialized and ready.")

    # Run as A2A server
    sup.run_server()


if __name__ == "__main__":
    run_supervisor()
