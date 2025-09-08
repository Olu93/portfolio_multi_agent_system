import logging

from typing import Any, Optional, AsyncIterable
from abc import abstractmethod
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
    Artifact,
    Part,
    TextPart,
    FilePart,
    DataPart,
    TaskState,
    AgentCard,
    AgentSkill,
    AgentCapabilities,
)
from a2a.server.agent_execution import RequestContext, AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from pydantic import BaseModel, Field
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from a2a_servers.config_loader import load_prompt_config, load_model_config
from a2a.types import MessageSendParams, SendStreamingMessageRequest, SendStreamingMessageResponse, Task, Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskStatus, TaskState

logger = logging.getLogger(__name__)


# --- Stream chunk model ------------------------------------------------------
A2AClientResponse = Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent

class ChunkMetadata(BaseModel):
    message_type: str = Field("UNKNOWN", example="tool_stream")
    step_number: int = Field(0, example=0)
    error: Optional[str] = None

class ChunkResponse(BaseModel):
    status: TaskState = Field(..., example=TaskState.working)
    content: str
    tool_name: Optional[str] = None
    metadata: ChunkMetadata = Field(..., example={"message_type": "tool_stream", "step_number": 0})

class ToolEmission(BaseModel):
    tool: str = Field(..., description="The name of the tool that emitted the event", example="tool_name")
    text: str = Field(..., description="The text of the event", example="text")
    state: TaskState = Field(..., description="The status of the event", example=TaskState.working)
    timestamp: str = Field(..., description="The timestamp of the event in iso format", example="2025-09-08T02:45:12.964656+00:00")


class A2ASubAgentClient:
    """A2A Simple to call A2A servers."""

    def __init__(self, default_timeout: float = 240.0):
        self._agent_info_cache: dict[
            str, dict[str, Any] | None
        ] = {}  # Cache for agent metadata
        self.default_timeout = default_timeout

    async def async_send_message_streaming(self, agent_url: str, message: str, context_id: str, task_id: str) -> AsyncIterable[A2AClientResponse]:
        """Send a message following the official A2A SDK pattern."""


        # Configure httpx client with timeout
        timeout_config = httpx.Timeout(
            timeout=self.default_timeout,
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0,
        )

        async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=agent_url,
                agent_card_path=AGENT_CARD_WELL_KNOWN_PATH,
            )
            logger.info(
                f'Attempting to fetch public agent card from: {agent_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )      
            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        f'\nPublic card supports authenticated extended card. Attempting to fetch from: {agent_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
                        exc_info=True,
                    )
            elif (
                _public_card
            ):  # supports_authenticated_extended_card is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )                  

            client = A2AClient(
                httpx_client=httpx_client, agent_card=final_agent_card_to_use
            )
            logger.info(f'A2AClient initialized. Connecting to {agent_url}')

            payload = new_agent_text_message(message, context_id=context_id)
            msg_params = MessageSendParams(message=payload)
            
            # request =  SendMessageRequest(
            #     id=str(uuid4()), params=msg_params
            # )

            # response = await client.send_message(request)
            # print(response.model_dump(mode='json', exclude_none=True))

            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()), params=msg_params
            )

            stream_response = client.send_message_streaming(streaming_request)

            # async for chunk in stream_response:
            #     yield chunk.model_dump(mode='python', exclude_none=True)

            async for chunk in stream_response:
                # yield each chunk as JSON lines (NDJSON)
                chunk: SendStreamingMessageResponse = chunk
                result: A2AClientResponse = chunk.root.result
                logger.info(f"{agent_url} - Received message of type: {type(result)}")
                # if isinstance(result, TaskStatusUpdateEvent) or isinstance(result, Task):
                #     logger.info(f"Status of task: {result.status}")
                #     status = result.status.state
                # else:
                #     logger.error(f"Unknown result type: {type(result)}")
                #     status = TaskState.unknown
                # chunk_normalized =ChatResponse(response=result, conversation_id=context_id, status=status)
                yield result


class BaseAgent:

    graph: CompiledStateGraph = None
    model: BaseChatModel = None
    agent_config: dict = None
    name: str = None
    url: str = None
    agent_card: AgentCard = None
    prompt: str = None
    tools: list[StructuredTool] = None
    http_handler: Starlette = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/event-stream", "application/json"]

    def __init__(self, name: str, url: str, agent_config: dict):
        self.name = name
        self.url = url
        self.agent_config = agent_config

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
            for tool in self.agent_config.get("tools", [])
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
        meta_prompt = self.agent_config.get("meta_prompt", 
        (
            f"You are {self.name} - A helpful agent that can use multiple tools"
            "Make sure to ask for more input if not enough information was provided to execute the task."        
        ))
        prompt_config = load_prompt_config(self.agent_config.get("prompt_file", f"{self.name}.txt"))
        self.prompt = f"{meta_prompt}\n\n{prompt_config}"
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
                metadata=ChunkMetadata(message_type="final_response", step_number=len(messages)),
            )
        return ChunkResponse(
            status=TaskState.input_required,
            content=f"{self.name} is unable to process your request at the moment. Please try again.",
            metadata=ChunkMetadata(message_type="error", error="no_messages", step_number=len(messages)),
        )

    @staticmethod
    def _format_part(part: Part) -> str:
        if isinstance(part.root, FilePart):
            return f'{part.root.kind}: {part.root.file}'
        elif isinstance(part.root, DataPart):
            return f'{part.root.kind}: {part.root.data}'
        return f'{part.root.kind}: {part.root.text}'
    
    def _format_parts(parts: list[Part]) -> str:
        return "\n".join([BaseAgent._format_part(p) for p in parts])
        
    @staticmethod
    def _extract_parts(artifact: Artifact):
        return f"artifact: {artifact.name}" + "\n"  + BaseAgent._format_parts(artifact.parts)

    async def _call_execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input() or ""
        task = context.current_task or new_task(context.message)
        artifacts: list[Artifact] = (task.artifacts or []) + [Artifact(name="User-Input", parts=[Part(root=TextPart(text=query))], artifact_id=str(uuid4()))]
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, context_id=task.context_id, task_id=task.id)

        parts_buffer: list[Part] = []

        try:
            await updater.start_work(new_agent_text_message(f"{self.name} is working…", task.context_id, task.id))

            async for item in self.stream(artifacts, context_id=task.context_id, task_id=task.id):
                msg = item.content  # guaranteed non-empty per your contract
                logger.info(f"{self.name} received message: {msg}")

                if item.status == TaskState.working:
                    logger.info(f"{self.name} working for task {task.id}")
                    await updater.update_status(item.status,
                                                new_agent_text_message(msg, task.context_id, task.id))
                    # parts_buffer.append(Part(root=TextPart(text=msg)))
                    continue

                if item.status == TaskState.input_required:
                    logger.info(f"{self.name} input required for task {task.id}")
                    await updater.requires_input(new_agent_text_message(msg, task.context_id, task.id))
                    break

                if item.status == TaskState.completed:
                    logger.info(f"{self.name} completed task {task.id}")
                    parts_buffer.append(Part(root=TextPart(text=msg)))
                    logger.info(f"{self.name} adding artifact to context {task.context_id} - task{task.id}")
                    await updater.add_artifact(parts_buffer, name="Agent-Response")
                    await updater.complete(new_agent_text_message(f"{self.name} persisted artifact to context {task.context_id} - task{task.id}", task.context_id, task.id))
                    break

                if item.status == TaskState.failed:
                    logger.info(f"{self.name} failed task {task.id}")
                    logger.error(f"{self.name} failed task {task.id} with error: {msg}")
                    await updater.failed(new_agent_text_message(f"Error: {msg}", task.context_id, task.id))
                    break

                if item.status == TaskState.canceled:
                    logger.info(f"{self.name} canceled task {task.id}")
                    logger.error(f"{self.name} canceled task {task.id} with error: {msg}")
                    await updater.cancel(new_agent_text_message(msg, task.context_id, task.id))
                    break

                if item.status == TaskState.rejected:
                    logger.info(f"{self.name} rejected task {task.id}")
                    logger.error(f"{self.name} rejected task {task.id} with error: {msg}")
                    await updater.reject(new_agent_text_message(msg, task.context_id, task.id))
                    break

                # Unknown: surface and buffer
                logger.info(f"{self.name} unknown status {item.status} for task {task.id}")
                logger.error(f"{self.name} unknown status {item.status} for task {task.id} with error: {msg}")
                await updater.update_status(item.status, new_agent_text_message(msg, task.context_id, task.id))
                parts_buffer.append(Part(root=TextPart(text=msg)))
        except GraphRecursionError as e:
            logger.exception(f"{self.name} graph recursion error for task {task.id}")
            all_parts = []
            for msg in task.history:
                logger.error(f"Message: {msg!s}")
                all_parts.extend(msg.parts)
            all_parts.append(Part(root=TextPart(text=f"Error: {e!s}")))

            await updater.failed(new_agent_text_message(f"Error: {e!s}", task.context_id, task.id))
        except Exception as e:
            logger.exception(f"{self.name} execute failed for task {task.id}")
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