# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (a2a package)
# ChatGPT convo: https://chatgpt.com/share/68b8361d-6b24-8009-ba31-aaaebdc96cc9
import asyncio
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
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain.chat_models import init_chat_model

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from a2a_servers.a2a_client import A2ASubAgentClient, call_execute, BaseAgent, ChunkResponse, ChunkMetadata
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
            async for chunk in client.async_send_message_streaming(card.url, content, context_id, task_id):
                try:
                    result = chunk["result"]
                    if result["kind"] == "artifact-update":
                        artifact = result["artifact"]
                        parts = artifact["parts"]
                        text = "\n\n".join([f'{p["kind"]}: {p["text"]}' for p in parts])
                    elif result["kind"] == "status-update":
                        status = result["status"]
                        state = status["state"]
                        timestamp = status["timestamp"]
                        text = f'{state} at {timestamp}'
                    elif result["kind"] == "task":
                        history = result["history"]
                        last_element = history[-1]
                        element_parts = last_element["parts"]
                        first_part = "\n\n".join([f'{p["kind"]}: {p["text"]}' for p in element_parts])
                        text = first_part
                    else:
                        text = json.dumps(chunk, ensure_ascii=False)
                except Exception as e:
                    logger.exception("Error processing tool output", e)
                    text = json.dumps(chunk, ensure_ascii=False)
                writer({"tool": card.name, "text": text})   # custom stream
                buf.append(text)

            # ToolNode will turn this into ToolMessage.content for the next model step
            if not buf:
                return ""
            return "### TOOL_OUTPUT_START\n" + "\n".join(buf) + "\n### TOOL_OUTPUT_END"


        tool_name = _safe_name(card.name) or _safe_name(card.url)
        capabilities = list(card.capabilities.model_dump(mode="python").items()) if card.capabilities else []
        desc_caps = ", ".join([f"{k}={v}" for k, v in sorted(capabilities, key=lambda x: x[0])])
        summary = f"{card.description or 'A2A agent'}. Caps: {desc_caps or 'unspecified'}"

        return StructuredTool.from_function(
            coroutine=tool_impl,
            name=tool_name,
            description=summary,
        )

    cards = await fetch_agents()
    if allow_urls:
        cards = [c for c in cards if c.url in allow_urls]
    if allow_caps:
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]

    tools: List[StructuredTool] = []
    for card in cards:
        tools.append(_create_tool_for_card(card))

    logger.info(f"Successfully created {len(tools)} tools from registry")
    return tools

# --- Supervisor Agent --------------------------------------------------------
class SupervisorAgent(BaseAgent):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/event-stream", "application/json"]

    
    def __init__(self, name, model: BaseChatModel, tools, prompt: str):
        self.name = name
        self.tools = tools
        self.prompt = prompt
        self.model = model.bind_tools(self.tools)
        self.graph = self.build_graph()

    def build_graph(self):
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
        inputs = {"messages": [("user", self._extract_parts(artifact)) for artifact in artifacts]}
        config = {"configurable": {"thread_id": context_id, "task_id": task_id, "stream_id": str(uuid.uuid4())}}

        async for evt in self.graph.astream(inputs, config, stream_mode=["values", "custom"]):
            # evt is either (mode, payload) or just payload depending on LG version
            if isinstance(evt, tuple) and len(evt) == 2:
                mode, payload = evt
            else:
                mode, payload = "values", evt

            if mode == "custom":
                # your tool's writer(...) payload
                text = payload.get("text", "")
                tool = payload.get("tool")
                yield ChunkResponse(
                    status=TaskState.working,
                    content=text,
                    tool_name=tool,
                    metadata=ChunkMetadata(message_type="tool_stream", step_number=0),
                )
                continue

            # mode == "values" → graph state snapshot
            item: Dict[str, Any] = payload
            messages = item.get("messages", [])
            if not messages:
                continue
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                yield ChunkResponse(
                    status=TaskState.working,
                    content="Processing the request...",
                    tool_name=last.tool_calls[0].get("name") if last.tool_calls else None,
                    metadata=ChunkMetadata(message_type="tool_call", step_number=len(messages)),
                )
            elif isinstance(last, ToolMessage):
                yield ChunkResponse(
                    status=TaskState.working,
                    content="Executing the tool...",
                    metadata=ChunkMetadata(message_type="tool_execution", step_number=len(messages)),
                )

        # final state
        yield self.get_agent_response(config)



    def get_agent_response(self, config) -> ChunkResponse:
        current_state = self.graph.get_state(config)
        messages = current_state.values.get("messages", [])
        if messages:
            last = messages[-1]
            text = getattr(last, "content", str(last))
            return ChunkResponse(
                status=TaskState.completed,
                content=text,
                metadata=ChunkMetadata(message_type="final_response", step_number=len(messages)),
            )
        return ChunkResponse(
            status=TaskState.input_required,
            content="We are unable to process your request at the moment. Please try again.",
            metadata=ChunkMetadata(message_type="error", error="no_messages", step_number=len(messages)),
        )

# --- Agent Executor ----------------------------------------------------------
class BaseAgentExecutor(AgentExecutor):
    def __init__(self, agent: SupervisorAgent, status_message: str = "Processing request...", artifact_name: str = "response"):
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await call_execute(self.agent, context, event_queue)


    async def execute_streaming(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self.execute(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

# --- Factory -----------------------------------------------------------------
async def create_supervisor_agent(agent_name: str, url: str):
    agent_config = load_agent_config(agent_name)
    model_config = load_model_config(agent_config.get("model", "default"))
    meta_prompt = agent_config.get("meta_prompt", "You are a helpful supervisor that can orchestrate multiple agents")
    prompt_config = load_prompt_config(agent_config.get("prompt_file", f"{agent_name}.txt"))

    allow_urls = set(agent_config.get("allow_urls", []) or [])
    allow_caps = set(agent_config.get("allow_caps", []) or [])
    tools = await build_tools_from_registry(allow_urls, allow_caps)

    model = init_chat_model(
        model_config["name"],
        **model_config["parameters"],
        model_provider=model_config["provider"],
    )

    prompt = f"{meta_prompt}\n\n{prompt_config}"
    agent = SupervisorAgent(agent_name, model, tools, prompt)

    skills = [
        AgentSkill(
            id=skill.get("name", ""),
            name=skill.get("name", ""),
            description=skill.get("description", ""),
            tags=skill.get("tags", []),
            examples=skill.get("examples", []),
        )
        for skill in agent_config.get("skills", [])
    ]
    description = agent_config.get("description", "")
    # capabilities = agent_config.get("capabilities", {})
    capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
    agent_card = AgentCard(
        version="1.0.0",
        name=agent_name,
        description=description,
        url=url,
        capabilities=capabilities,
        skills=skills,
        default_input_modes=SupervisorAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=SupervisorAgent.SUPPORTED_CONTENT_TYPES,
    )

    executor = BaseAgentExecutor(agent=agent, status_message="Processing request...", artifact_name="supervisor_response")
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    
    # Add health endpoint
    async def health_check(request: Request):
        """Health check endpoint."""
        logger.info("Health check endpoint called")
        return JSONResponse({"status": "healthy"})

    # starlette_app: Starlette = app.build(routes=[Route("/health", health_check, methods=["GET"])])
    starlette_app: Starlette = app.build()
    starlette_app.add_route("/health", health_check, methods=["GET"])

    return starlette_app

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
        app = await create_supervisor_agent(agent_name, url=URL)
        return app

    fastapi_app = asyncio.run(run_with_background_tasks())

    uvicorn.run(fastapi_app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    run_supervisor()
