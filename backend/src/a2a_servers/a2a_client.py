import asyncio
import json
import logging
import re
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncIterable, AsyncIterator, Optional, TypedDict
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    MessageSendParams,
    Part,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from a2a_servers.config_loader import load_model_config, load_prompt_config
from a2a_servers.utils.models import (
    A2AClientResponse,
    AuthHeaderCb,
    ChatResponse,
    ChunkMetadata,
    ChunkResponse,
    State,
    ToolEmission,
)

logger = logging.getLogger(__name__)


# --- Tool factory from AgentCards -------------------------------------------
async def fetch_agents(httpx_client: httpx.AsyncClient, registry_url: str) -> list[AgentCard]:
    logger.info(f"Fetching agents from registry at {registry_url}")
    # async with httpx_client as s:
    r = await httpx_client.get(f"{registry_url}/registry/agents")
    r.raise_for_status()
    return [AgentCard(**a) for a in r.json()]


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()


class A2ASubAgentClient:
    """Lightweight A2A client with pooled HTTP, card caching, and streaming."""

    def __init__(
        self,
        default_timeout: float = 240.0,
        auth_header_cb: Optional[AuthHeaderCb] = None,
        httpx_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.default_timeout = default_timeout
        self._card_cache: dict[str, AgentCard] = {}
        self._auth_header_cb = auth_header_cb
        self._own_client = httpx_client is None
        self._httpx = httpx_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=default_timeout, connect=10.0, read=default_timeout, write=10.0, pool=5.0)
        )

    async def aclose(self) -> None:
        if self._own_client:
            await self._httpx.aclose()

    async def _resolve_card(self, agent_url: str) -> AgentCard:
        if agent_url in self._card_cache:
            return self._card_cache[agent_url]

        resolver = A2ACardResolver(httpx_client=self._httpx, base_url=agent_url, agent_card_path=AGENT_CARD_WELL_KNOWN_PATH)

        # minimal retry on transient network hiccups
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                public_card: AgentCard = await resolver.get_agent_card()
                logger.info("Fetched public agent card for %s", agent_url)

                final = public_card
                if getattr(public_card, "supports_authenticated_extended_card", False) and self._auth_header_cb:
                    try:
                        headers = self._auth_header_cb(agent_url)
                        extended = await resolver.get_agent_card(
                            relative_card_path=EXTENDED_AGENT_CARD_PATH,
                            http_kwargs={"headers": headers},
                        )
                        logger.info("Fetched extended agent card for %s", agent_url)
                        final = extended
                    except Exception as e_ext:
                        logger.warning("Extended card fetch failed (%s); using public.", e_ext, exc_info=True)

                self._card_cache[agent_url] = final
                return final
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.3)

        assert last_err
        raise last_err

    async def stream(
        self,
        agent_url: str,
        msg: str,
        context_id: str,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """Yield A2AClientResponse chunks."""
        card = await self._resolve_card(agent_url)
        client = A2AClient(httpx_client=self._httpx, agent_card=card)

        payload = new_agent_text_message(msg, context_id=context_id, task_id=task_id)
        req = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(message=payload))

        stream = client.send_message_streaming(req)

        try:
            async for chunk in stream:
                # chunk is SendStreamingMessageResponse -> yield the result object
                result = chunk.root.result  # type: ignore[attr-defined]
                # keep logging cheap; remove if noisy
                logger.debug("%s: got %s", agent_url, type(result).__name__)
                yield result
        except asyncio.CancelledError as e:
            logger.info("Stream cancelled for %s", agent_url)
            raise e

    async def build_tools_from_registry(self, allow_urls: set, allow_caps: set, registry_url: str) -> list[StructuredTool]:
        logger.info(f"Building tools from registry with allow_urls={allow_urls}, allow_caps={allow_caps}")

        cards = await fetch_agents(self._httpx, registry_url)
        if allow_urls:
            cards = [c for c in cards if c.url in allow_urls]
        if allow_caps:
            cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]

        tools: list[StructuredTool] = []
        tool_configs: list[dict] = []
        for card in cards:
            tool, tool_config = self._create_tool_for_card(card)
            tools.append(tool)
            tool_configs.append(tool_config)

        logger.info(f"Successfully created {len(tools)} tools from registry")
        return tools, tool_configs

    def _create_tool_for_card(self, card: AgentCard) -> StructuredTool:
        async def tool_impl(content: State, config: RunnableConfig):
            cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
            context_id = cfg.get("thread_id")
            task_id = content.get("task_id")
            msg = content.get("messages")[-1]["content"]

            writer = get_stream_writer()

            buf: list[str] = []
            async for chunk in self.stream(card.url, msg, context_id, task_id):
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
                        buf.append(f"Agent received artifact at {timestamp}:\n{text}")
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, TaskStatusUpdateEvent) and result.status.state not in [
                        TaskState.working,
                        TaskState.input_required,
                    ]:
                        status = result.status
                        state = status.state
                        timestamp = status.timestamp
                        parts = result.status.message.parts
                        text = BaseAgent._format_parts(parts)
                        emission = ToolEmission(
                            tool=card.name,
                            text=f"{timestamp} - Received Status {state} from {card.name}: {text}",
                            state=state,
                            timestamp=timestamp,
                        )
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, TaskStatusUpdateEvent) and result.status.state == TaskState.working:
                        status = result.status
                        state = status.state
                        timestamp = status.timestamp
                        parts = result.status.message.parts
                        text = BaseAgent._format_parts(parts)
                        emission = ToolEmission(
                            tool=card.name,
                            text=f"{timestamp} - Received Status {state} from {card.name}: {text}",
                            state=state,
                            timestamp=timestamp,
                        )
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, TaskStatusUpdateEvent) and result.status.state == TaskState.input_required:
                        status = result.status
                        # sub_task_id = result.root.
                        state = status.state
                        timestamp = status.timestamp
                        parts = result.status.message.parts
                        text = BaseAgent._format_parts(parts)
                        emission = ToolEmission(
                            tool=card.name,
                            text=f"{timestamp} - Received Status {state} from {card.name}: {text}",
                            state=state,
                            timestamp=timestamp,
                        )
                        buf.append(f"Agent asks for input at {timestamp}:\n{text}")
                        logger.info(emission)
                        writer(emission)
                    elif isinstance(result, Task):
                        state = TaskState.working
                        timestamp = datetime.now(timezone.utc).isoformat()
                        history = result.history
                        last_element = history[-1]
                        element_parts: list[Part] = last_element.parts
                        first_part = BaseAgent._format_parts(element_parts)
                        emission = ToolEmission(
                            tool=card.name,
                            text=f"{timestamp} - Received Task from {card.name}: {first_part}",
                            state=state,
                            timestamp=timestamp,
                        )
                        logger.info(emission)
                        writer(emission)
                    else:
                        raise Exception(f"Unknown result type: {type(result)}")
                except Exception as e:
                    logger.exception("Error processing tool output", e)

            new_task_id = result.task_id if result.status.state == TaskState.input_required else None
            return {
                "messages": ["### TOOL_OUTPUT_START\n" + "\n".join(buf) + "\n### TOOL_OUTPUT_END"],
                "task_id": new_task_id,
            }

        tool_name = _safe_name(card.name) or _safe_name(card.url)
        tool_examples = [e for s in card.skills for e in s.examples]
        tool_tags = [t for s in card.skills for t in s.tags]
        capabilities = list(card.capabilities.model_dump(mode="python").items()) if card.capabilities else []
        capabilities_strings = [f"{k}={v}" for k, v in sorted(capabilities, key=lambda x: x[0])]
        summary = (
            f"Skill to call {tool_name}\n"
            f"Description: {card.description or 'A2A agent'}\n"
            f"Examples:\n{'\n'.join(tool_examples)}\n"
            f"Tags: {','.join(tool_tags)}\n"
            f"Capabilities: {','.join(capabilities_strings)}\n"
        )
        tool_config = {
            "name": tool_name,
            "description": summary,
            "examples": tool_examples,
            "tags": tool_tags,
            "capabilities": capabilities,
        }
        t = StructuredTool.from_function(
            coroutine=tool_impl,
            name=tool_name,
            description=summary,
        )
        return t, tool_config

class AgentConfig(TypedDict):
    name: str
    url: str
    agent_config: dict
    sub_task_id: Optional[str] = None
    prompt: Optional[str] = None
    skillset: Optional[list[StructuredTool]] = None
    http_handler: Optional[Starlette] = None

class BaseAgent:
    graph: CompiledStateGraph = None
    model: BaseChatModel = None
    agent_config: AgentConfig = None
    name: str = None
    url: str = None
    agent_card: AgentCard = None
    prompt: str = None
    tools: list[StructuredTool] = None
    http_handler: Starlette = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/event-stream", "application/json"]

    def __init__(self, name: str, url: str, agent_config: AgentConfig):
        self.name = name
        self.url = url
        self.agent_config = agent_config
        self.sub_task_id = None

    async def build(self):
        self.prompt = await self.build_prompt()
        self.tools = await self.build_tools()
        self.model = await self.build_model()
        self.graph = await self.build_graph()
        self.agent_card = await self.build_agent_card()
        self.http_handler = await self.build_http_handler()
        return self

    @abstractmethod
    async def stream(self, artifacts: list[Artifact], context_id: str, task_id: str) -> AsyncIterable[ChunkResponse]:
        yield None

    @abstractmethod
    async def build_tools(self) -> list[StructuredTool]:
        return []

    @abstractmethod
    async def build_graph(self) -> CompiledStateGraph:
        return None

    async def build_agent_card(self) -> AgentCard:
        skills = [
            AgentSkill(
                id=tool["name"],
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                tags=tool.get("tags", []),
                examples=tool.get("examples", []),
            )
            for tool in self.agent_config.get("skillset", [])
        ]
        description = self.agent_config.get("description", "")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=False)
        self.agent_card = AgentCard(
            name=self.name,
            description=description,
            url=self.url,
            version="1.0.0",
            defaultInputModes=BaseAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=BaseAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )
        return self.agent_card

    async def build_prompt(self) -> str:
        meta_prompt = self.agent_config.get(
            "meta_prompt",
            (
                f"You are {self.name} - A helpful agent that can use multiple tools"
                "Make sure to ask for more input if not enough information was provided to execute the task."
            ),
        )
        err_prompt = "If you repeatedly fail to fulfill your task and the user can't help you, be honest and provide a response that you cannot fulfill the task and mention the reason."
        prompt_config = load_prompt_config(self.agent_config.get("prompt_file", f"{self.name}.txt"))
        self.prompt = f"{meta_prompt}\n{err_prompt}\n\n{prompt_config}"
        return self.prompt

    async def build_model(self) -> BaseChatModel:
        model_config = load_model_config(self.agent_config.get("model", "default"))
        self.model = init_chat_model(
            model_config["name"],
            **model_config["parameters"],
            model_provider=model_config["provider"],
        )
        return self.model

    async def build_http_handler(self) -> Starlette:
        executor = BaseAgentExecutor(agent=self)
        request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
        app = A2AStarletteApplication(agent_card=self.agent_card, http_handler=request_handler)

        # Add health endpoint
        async def health_check(request: Request):
            """Health check endpoint."""
            logger.info("Health check endpoint called")
            return JSONResponse({"status": "healthy"})

        # starlette_app: Starlette = app.build(routes=[Route("/health", health_check, methods=["GET"])])
        starlette_app: Starlette = app.build()
        starlette_app.add_route("/health", health_check, methods=["GET"])

        return starlette_app

    def _get_agent_response(self, config) -> ChunkResponse:
        current_state = self.graph.get_state(config)
        messages = current_state.values.get("messages", [])
        if messages:
            last = messages[-1]
            text = getattr(last, "content", str(last))
            return ChunkResponse(
                status=TaskState.completed,
                content=text,
                tool_name=None,
                metadata=ChunkMetadata(message_type="final_response", step_number=len(messages)),
            )
        return ChunkResponse(
            status=TaskState.input_required,
            content=f"{self.name} is unable to process your request at the moment. Please try again.",
            tool_name=None,
            metadata=ChunkMetadata(message_type="error", error="no_messages", step_number=len(messages)),
        )

    @staticmethod
    def _format_part(part: Part) -> str:
        if isinstance(part.root, FilePart):
            return f"{part.root.kind}: {part.root.file}"
        elif isinstance(part.root, DataPart):
            return f"{part.root.kind}: {part.root.data}"
        return f"{part.root.kind}: {part.root.text}"

    def _format_parts(parts: list[Part]) -> str:
        return "\n".join([BaseAgent._format_part(p) for p in parts])

    @staticmethod
    def _extract_parts(artifact: Artifact):
        return f"artifact: {artifact.name}" + "\n" + BaseAgent._format_parts(artifact.parts)

    async def _call_execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input() or ""
        task = context.current_task or new_task(context.message)

        is_resume = context.current_task is not None  # paused task being resumed

        # Enqueue ONLY for new tasks (resume uses a fresh EventQueue but no enqueue)
        if not is_resume:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, context_id=task.context_id, task_id=task.id)

        # (Optional) persist the new user input as artifact
        user_art = Artifact(
            name="User-Input",
            parts=[Part(root=TextPart(text=query))],
            artifact_id=str(uuid4()),
        )
        await updater.add_artifact(user_art.parts, name=user_art.name, artifact_id=user_art.artifact_id)

        try:
            await updater.start_work(new_agent_text_message(f"{self.name} is working…", task.context_id, task.id))
            logger.info(f"{self.name} is working for context {task.context_id} and task {task.id}")

            async for item in self.stream([user_art], context_id=task.context_id, task_id=task.id):
                msg = item.content

                if item.status == TaskState.working:
                    logger.info(f"{self.name} is working for context {task.context_id} and task {task.id}")
                    await updater.update_status(TaskState.working, new_agent_text_message(msg, task.context_id, task.id))
                    continue

                if item.status == TaskState.input_required:
                    logger.info(f"{self.name} is input required for context {task.context_id} and task {task.id}")
                    await updater.add_artifact([Part(root=TextPart(text=msg))], name="Agent-Response")
                    await updater.requires_input(new_agent_text_message(msg, task.context_id, task.id), True)
                    break

                if item.status == TaskState.completed:
                    logger.info(f"{self.name} is completed for context {task.context_id} and task {task.id}")
                    await updater.add_artifact([Part(root=TextPart(text=msg))], name="Agent-Response")
                    await updater.complete(new_agent_text_message(msg, task.context_id, task.id))
                    break

                if item.status == TaskState.failed:
                    logger.info(f"{self.name} is failed for context {task.context_id} and task {task.id}")
                    await updater.failed(new_agent_text_message(f"Error: {msg}", task.context_id, task.id))
                    break
                if item.status == TaskState.canceled:
                    logger.info(f"{self.name} is canceled for context {task.context_id} and task {task.id}")
                    await updater.cancel(new_agent_text_message(msg, task.context_id, task.id))
                    break
                if item.status == TaskState.rejected:
                    logger.info(f"{self.name} is rejected for context {task.context_id} and task {task.id}")
                    await updater.reject(new_agent_text_message(msg, task.context_id, task.id))
                    break

                logger.info(f"{self.name} is {item.status} for context {task.context_id} and task {task.id}")
                await updater.update_status(item.status, new_agent_text_message(msg, task.context_id, task.id))

        except GraphRecursionError as e:
            logger.exception(f"{self.name} graph recursion error for context {task.context_id} and task {task.id}")
            all_parts = []
            for msg in task.history:
                logger.error(f"Message: {msg!s}")
                all_parts.extend(msg.parts)
            all_parts.append(Part(root=TextPart(text=f"Error: {e!s}")))
            logger.error(f"{self.name} graph recursion error for context {task.context_id} and task {task.id}")
            logger.error(f"Error: {e!s}")
            await updater.failed(new_agent_text_message(f"Error: {e!s}", task.context_id, task.id))
        except Exception as e:
            logger.exception(f"{self.name} execute failed for context {task.context_id} and task {task.id}")
            logger.error(f"{self.name} execute failed for context {task.context_id} and task {task.id}")
            await updater.failed(new_agent_text_message(f"Error: {e!s}", task.context_id, task.id))


# --- Agent Executor ----------------------------------------------------------
class BaseAgentExecutor(AgentExecutor):
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self.agent._call_execute(context, event_queue)

    async def execute_streaming(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self.execute(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


# --- A2A Client ----------------------------------------------------------
async def async_send_message_streaming(agent_url: str, message: str, context_id: str | None, task_id: str | None):
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=agent_url,
            agent_card_path=AGENT_CARD_WELL_KNOWN_PATH,
        )

        public_card = await resolver.get_agent_card()
        final_card = public_card

        if getattr(public_card, "supports_authenticated_extended_card", False):
            try:
                extended = await resolver.get_agent_card(
                    relative_card_path=EXTENDED_AGENT_CARD_PATH,
                )
                final_card = extended
            except Exception:
                pass  # fall back to public card

        client = A2AClient(httpx_client=httpx_client, agent_card=final_card)

        payload = new_agent_text_message(message, context_id=context_id or str(uuid4()))
        params = MessageSendParams(message=payload)
        req = SendStreamingMessageRequest(id=str(uuid4()), params=params)

        stream = client.send_message_streaming(req)
        async for chunk in stream:
            # yield each chunk as JSON lines (NDJSON)
            chunk: SendStreamingMessageResponse = chunk
            result: A2AClientResponse = chunk.root.result
            logger.info(f"Received message of type: {type(result)}")
            response = ""
            if isinstance(result, TaskStatusUpdateEvent):
                logger.info(f"Status of task: {result.status}")
                status = result.status.state
                response = result.status.message.parts[-1].root.text
            elif isinstance(result, Task):
                logger.info(f"Task: {result}")
                status = result.status.state
                # response = result.status.message.parts[-1].text
                continue
            elif isinstance(result, TaskArtifactUpdateEvent):
                logger.info(f"Artifact of task: {result.artifact}")
                status = TaskState.working
                continue
            else:
                logger.error(f"Unknown result type: {type(result)}")
                status = TaskState.unknown
                continue
            chunk_normalized = ChatResponse(response=response, conversation_id=context_id, status=status)
            yield json.dumps(chunk_normalized.model_dump(mode="python", exclude_none=True)) + "\n"
