# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (a2a package)
# ChatGPT convo: https://chatgpt.com/share/68b8361d-6b24-8009-ba31-aaaebdc96cc9
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
from urllib.parse import urlparse
import uuid
from typing import Annotated, Any, Dict, Optional, List, AsyncIterable, TypedDict

import click
import httpx
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    Artifact,
    TaskState,
    TextPart,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Task,
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain.chat_models import init_chat_model

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from a2a_servers.a2a_client import A2ASubAgentClient, BaseAgent, BaseAgentExecutor, ChunkResponse, ChunkMetadata, ToolEmission
from a2a_servers.config_loader import (
    load_agent_config,
    load_model_config,
    load_prompt_config,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field
from langgraph.config import get_stream_writer
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request
import uvicorn


logger = logging.getLogger(__name__)

# --- State -------------------------------------------------------------------
memory = InMemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]



# --- Registry client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")

async def fetch_agents() -> List[AgentCard]:
    logger.info(f"Fetching agents from registry at {REGISTRY_URL}")
    async with httpx.AsyncClient(timeout=10) as s:
        r = await s.get(f"{REGISTRY_URL}/registry/agents")
        r.raise_for_status()
        return [AgentCard(**a) for a in r.json()]

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
        async def tool_impl(content: str, config: RunnableConfig):
            cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
            context_id = cfg.get("thread_id")
            task_id = cfg.get("task_id")

            client = A2ASubAgentClient()
            writer = get_stream_writer()

            buf: list[str] = []
            text = ""
            async for chunk in client.async_send_message_streaming(card.url, content, context_id, task_id):
                try:
                    result = chunk
                    if isinstance(result, TaskArtifactUpdateEvent):
                        state = TaskState.working
                        timestamp = datetime.now(timezone.utc).isoformat()
                        artifact = result.artifact
                        text = BaseAgent._extract_parts(artifact)
                        emission = ToolEmission(
                            tool=card.name, 
                            text=f"{timestamp} - Received Artifact from {card.name}", 
                            state=state,
                            timestamp=timestamp,
                            )
                        buf.append(f"Agent {card.name} received artifact at {timestamp}:\n{text}")
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, TaskStatusUpdateEvent):
                        status = result.status
                        state = status.state
                        timestamp = status.timestamp
                        parts = result.status.message.parts
                        emission = ToolEmission(
                            tool=card.name, 
                            text=f'{timestamp} - Received Status {state} from {card.name}: {BaseAgent._format_parts(parts)}', 
                            state=state, 
                            timestamp=timestamp,
                            )
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, Task):
                        state = TaskState.working
                        timestamp = datetime.now(timezone.utc).isoformat()
                        history = result.history
                        last_element = history[-1]
                        element_parts: List[Part] = last_element.parts
                        first_part = BaseAgent._format_parts(element_parts)
                        emission = ToolEmission(
                            tool=card.name, 
                            text=f'{timestamp} - Received Task from {card.name}: {first_part}', 
                            state=state, 
                            timestamp=timestamp,
                            )
                        logger.info(emission)
                        writer(emission)
                    else:
                        raise Exception(f"Unknown result type: {type(result)}")
                except Exception as e:
                    logger.exception("Error processing tool output", e)
                   # custom stream

            # ToolNode will turn this into ToolMessage.content for the next model step
            if not buf:
                return ""
            return "### TOOL_OUTPUT_START\n" + "\n".join(buf) + "\n### TOOL_OUTPUT_END"


        tool_name = _safe_name(card.name) or _safe_name(card.url)
        tool_examples = [e for s in card.skills for e in s.examples]
        tool_tags = [t for s in card.skills for t in s.tags]
        capabilities = list(card.capabilities.model_dump(mode="python").items()) if card.capabilities else []
        capabilities_strings =[f"{k}={v}" for k, v in sorted(capabilities, key=lambda x: x[0])]
        summary = (
                    f"Skill to call {tool_name}\n"
                    f"Description: {card.description or 'A2A agent'}\n"
                    f"Examples:\n{'\n'.join(tool_examples)}\n"
                    f"Tags: {','.join(tool_tags)}\n"
                    f"Capabilities: {','.join(capabilities_strings)}\n"
                   )
        tool_config = {"name": tool_name, "description": summary, "examples": tool_examples, "tags": tool_tags, "capabilities": capabilities}
        t = StructuredTool.from_function(
            coroutine=tool_impl,
            name=tool_name,
            description=summary,
        )
        return t, tool_config

    cards = await fetch_agents()
    if allow_urls:
        cards = [c for c in cards if c.url in allow_urls]
    if allow_caps:
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]

    tools: List[StructuredTool] = []
    tool_configs: List[dict] = []
    for card in cards:
        tool, tool_config = _create_tool_for_card(card)
        tools.append(tool)
        tool_configs.append(tool_config)

    logger.info(f"Successfully created {len(tools)} tools from registry")
    return tools, tool_configs

# --- Supervisor Agent --------------------------------------------------------
class SupervisorAgent(BaseAgent):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/event-stream", "application/json"]


    async def build_model(self) -> BaseChatModel:
        model = await super().build_model()     
        self.model = model.bind_tools(self.tools)
        return self.model

    async def build_tools(self) -> List[StructuredTool]:
        allow_urls = set(self.agent_config.get("allow_urls", []) or [])
        allow_caps = set(self.agent_config.get("allow_caps", []) or [])
        tools, tool_configs = await build_tools_from_registry(allow_urls, allow_caps)
        self.agent_config["tools"] = tool_configs
        return tools

    async def build_graph(self) -> CompiledStateGraph:
        def model_execution(state: State) -> State:
            message: AIMessage = self.model.invoke(state["messages"])
            # force only one tool call per step
            if message.tool_calls and len(message.tool_calls) > 1:
                message.tool_calls = [message.tool_calls[0]]
            return {"messages": [message]}

        gb = StateGraph(State)
        gb.add_node("model", model_execution)
        gb.add_node("tools", ToolNode(self.tools))

        gb.add_edge(START, "model")
        # decide dynamically: tool call → tools, else → END
        gb.add_conditional_edges("model", tools_condition)
        gb.add_edge("tools", "model")

        return gb.compile(checkpointer=memory, name=self.name)


    async def stream(self, artifacts: list[Artifact], context_id: str, task_id: str) -> AsyncIterable[ChunkResponse]:
        inputs = {"messages": [("user", BaseAgent._extract_parts(artifact)) for artifact in artifacts]}
        config = {"configurable": {"thread_id": context_id, "task_id": task_id, "stream_id": str(uuid.uuid4())}}

        async for evt in self.graph.astream(inputs, config, stream_mode=["values", "custom"]):
            # evt is either (mode, payload) or just payload depending on LG version
            if isinstance(evt, tuple) and len(evt) == 2:
                mode, payload = evt
            else:
                mode, payload = "values", evt

            if mode == "custom":
                # your tool's writer(...) payload
                if isinstance(payload, ToolEmission):
                    emission = payload
                    response = ChunkResponse(
                        status=TaskState.working,
                        content=emission.text,
                        tool_name=emission.tool,
                        metadata=ChunkMetadata(message_type="tool_stream", step_number=0),
                    )
                    yield response
                # elif isinstance(payload, dict):
                #     emission = payload
                #     response = ChunkResponse(
                #         status=TaskState.working,
                #         content=emission["text"],
                #         tool_name=emission["tool"],
                #         metadata=ChunkMetadata(message_type="tool_stream", step_number=0),
                #     )
                #     yield response
                else:
                    raise Exception(f"Unknown payload type: {type(payload)}")
                continue

            # mode == "values" → graph state snapshot
            item: Dict[str, Any] = payload
            messages = item.get("messages", [])
            if not messages:
                continue
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                last_tool_call = last.tool_calls[0]
                response = ChunkResponse(
                    status=TaskState.working,
                    content=f"{self.name} is calling the tool... {last_tool_call.get('name')}",
                    tool_name=last_tool_call.get("name") if last_tool_call else None,
                    metadata=ChunkMetadata(message_type="tool_call", step_number=len(messages)),
                )
                yield response
            elif isinstance(last, ToolMessage):
                response = ChunkResponse(
                    status=TaskState.working,
                    content=f"{self.name} executed the tool successfully... {last.name}",
                    metadata=ChunkMetadata(message_type="tool_execution", step_number=len(messages)),
                )
                yield response
        # final state
        response = self._get_agent_response(config)
        yield response







# --- Factory -----------------------------------------------------------------
async def create_supervisor_agent(agent_name: str, url: str):
    try:
        agent_config = load_agent_config(agent_name)
        agent = await SupervisorAgent(agent_name, url, agent_config).build()
        return agent
    except Exception as e:
        logger.error(f"Error: {e!s}")
        raise e

# --- CLI ---------------------------------------------------------------------
@click.command()
@click.option("--agent-name", default="supervisor", help="Name of the supervisor agent config to load")
def run_supervisor(agent_name: str):
    URL = os.getenv("URL", "http://0.0.0.0:10020")
    HOST = urlparse(URL).hostname
    PORT = int(urlparse(URL).port)

    logger.info(f"Access agent at: {URL}") 
    logger.debug(f"HOST: {HOST}") 
    logger.debug(f"PORT: {PORT}")

    async def run_with_background_tasks():
        agent = await create_supervisor_agent(agent_name, url=URL)
        app = agent.http_handler
        return app

    fastapi_app = asyncio.run(run_with_background_tasks())

    uvicorn.run(fastapi_app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    run_supervisor()
