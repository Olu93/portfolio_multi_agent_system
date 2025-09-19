# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (a2a package)
# ChatGPT convo: https://chatgpt.com/share/68b8361d-6b24-8009-ba31-aaaebdc96cc9
import asyncio
import logging
import os
import uuid
from typing import Any, AsyncIterable
from urllib.parse import urlparse

import click
import httpx
import uvicorn
from a2a.types import (
    Artifact,
    TaskState,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from a2a_servers.a2a_client import (
    A2ASubAgentClient,
    BaseAgent,
    ChunkMetadata,
    ChunkResponse,
    State,
    ToolEmission,
)
from a2a_servers.config_loader import (
    load_agent_config,
)

httpx_client = httpx.AsyncClient(timeout=10)

logger = logging.getLogger(__name__)

# --- State -------------------------------------------------------------------
memory = InMemorySaver()


# --- Registry client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")
httpx_client = httpx.AsyncClient(timeout=10)


# --- Supervisor Agent --------------------------------------------------------
class SupervisorAgent(BaseAgent):
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/event-stream", "application/json"]

    async def build_model(self) -> BaseChatModel:
        model = await super().build_model()
        self.model = model.bind_tools(self.tools)
        return self.model

    async def build_tools(self) -> list[StructuredTool]:
        # BLOG: Explain that these tools are built from the registry once. New tools require restart. Explain how dynamic tools could be built on the fly and benefits and disadvantages.
        allow_urls = set(self.agent_config.get("allow_urls", []) or [])
        allow_caps = set(self.agent_config.get("allow_caps", []) or [])
        self.sub_agent_client = A2ASubAgentClient(httpx_client=httpx_client)
        tools, tool_configs = await self.sub_agent_client.build_tools_from_registry(allow_urls, allow_caps, REGISTRY_URL)
        self.agent_config["tools"] = tool_configs
        return tools

    async def build_graph(self) -> CompiledStateGraph:
        def model_execution(state: State) -> State:
            messages = state["messages"]
            task_id = state.get("task_id")
            message: AIMessage = self.model.invoke(messages)
            # force only one tool call per step
            if message.tool_calls and len(message.tool_calls) > 1:
                message.tool_calls = [message.tool_calls[0]]
            return {"messages": [message], "task_id": task_id}

        gb = StateGraph(State)
        gb.add_node("model", model_execution)
        gb.add_node("tools", ToolNode(self.tools))
        # gb.add_node("state_updater", )

        gb.add_edge(START, "model")
        # decide dynamically: tool call → tools, else → END
        gb.add_conditional_edges("model", tools_condition)

        gb.add_edge("tools", "model")

        return gb.compile(checkpointer=memory, name=self.name)

    async def stream(self, artifacts: list[Artifact], context_id: str, task_id: str) -> AsyncIterable[ChunkResponse]:
        inputs = {"messages": [("user", BaseAgent._extract_parts(artifact)) for artifact in artifacts]}
        config = {
            "configurable": {
                "thread_id": context_id,
                "task_id": task_id,
                "stream_id": str(uuid.uuid4()),
            }
        }

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
                else:
                    raise Exception(f"Unknown payload type: {type(payload)}")
                continue

            # mode == "values" → graph state snapshot
            item: dict[str, Any] = payload
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
                    tool_name=last.name,
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
