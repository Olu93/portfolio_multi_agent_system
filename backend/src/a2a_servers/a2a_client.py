import logging

from typing import Any, Optional, AsyncIterable
from abc import abstractmethod
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Artifact,
    Part,
    TextPart,
    TaskState,
)
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from pydantic import BaseModel, Field
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message

logger = logging.getLogger(__name__)


# --- Stream chunk model ------------------------------------------------------
class ChunkMetadata(BaseModel):
    message_type: str = Field("UNKNOWN", example="tool_stream")
    step_number: int = Field(0, example=0)
    error: Optional[str] = None

class ChunkResponse(BaseModel):
    status: TaskState = Field(..., example=TaskState.working)
    content: str
    tool_name: Optional[str] = None
    metadata: ChunkMetadata = Field(..., example={"message_type": "tool_stream", "step_number": 0})



class A2ASubAgentClient:
    """A2A Simple to call A2A servers."""

    def __init__(self, default_timeout: float = 240.0):
        self._agent_info_cache: dict[
            str, dict[str, Any] | None
        ] = {}  # Cache for agent metadata
        self.default_timeout = default_timeout

    async def async_send_message_streaming(self, agent_url: str, message: str, context_id: str, task_id: str) -> str:
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

            async for chunk in stream_response:
                yield chunk.model_dump(mode='python', exclude_none=True)


class BaseAgent:
    @abstractmethod
    async def stream(self, artifacts: list[Artifact], context_id: str, task_id: str) -> AsyncIterable[ChunkResponse]:
        yield None

async def call_execute(agent:BaseAgent, context: RequestContext, event_queue: EventQueue):
    query = context.get_user_input() or ""
    task = context.current_task or new_task(context.message)
    artifacts: list[Artifact] = task.artifacts + [Artifact(name="User-Input", parts=[Part(root=TextPart(text=query))])]
    await event_queue.enqueue_event(task)
    updater = TaskUpdater(event_queue, context_id=task.context_id, task_id=task.id)

    parts_buffer: list[Part] = []

    try:
        await updater.start_work(new_agent_text_message("Working…", task.context_id, task.id))

        async for item in agent.stream(artifacts, context_id=task.context_id, task_id=task.id):
            msg = item.content  # guaranteed non-empty per your contract

            if item.status == TaskState.working:
                await updater.update_status(TaskState.working,
                                            new_agent_text_message(msg, task.context_id, task.id))
                parts_buffer.append(Part(root=TextPart(text=msg)))
                continue

            if item.status == TaskState.input_required:
                await updater.requires_input(new_agent_text_message(msg, task.context_id, task.id))
                break

            if item.status == TaskState.completed:
                parts_buffer.append(Part(root=TextPart(text=msg)))
                await updater.add_artifact(parts_buffer, name="Agent-Response")
                await updater.complete(new_agent_text_message(msg, task.context_id, task.id))
                break

            if item.status == TaskState.failed:
                await updater.failed(new_agent_text_message(f"Error: {msg}", task.context_id, task.id))
                break

            if item.status == TaskState.canceled:
                await updater.cancel(new_agent_text_message(msg, task.context_id, task.id))
                break

            if item.status == TaskState.rejected:
                await updater.reject(new_agent_text_message(msg, task.context_id, task.id))
                break

            # Unknown: surface and buffer
            await updater.update_status(item.status, new_agent_text_message(msg, task.context_id, task.id))
            parts_buffer.append(Part(root=TextPart(text=msg)))

    except Exception as e:
        logger.exception("Supervisor execute failed")
        await updater.failed(new_agent_text_message(f"Error: {e!s}", task.context_id, task.id))    