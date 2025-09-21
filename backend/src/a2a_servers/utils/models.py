from typing import Annotated, Callable, Optional, TypedDict

from a2a.types import Message, Task, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, HttpUrl

A2AClientResponse = Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent


class ChatReq(BaseModel):
    message: str = Field(..., example="What's the latest news about Apple's stock?")
    context_id: Optional[str] = Field(None, example="123")
    task_id: Optional[str] = Field(None, example="123")


class ChatResponse(BaseModel):
    response: str = Field(..., example="Hello, how can I help you today?")
    conversation_id: str = Field(..., example="123")
    status: TaskState = Field(..., example="success")
    error: str | None = Field(None)


# Data model for Agent registration
class AgentRegistration(BaseModel):
    name: str
    description: str
    url: HttpUrl
    version: str
    capabilities: dict = {}
    skills: list[dict] = []


class HeartbeatRequest(BaseModel):
    url: HttpUrl


class LookupRequest(BaseModel):
    url: HttpUrl


class State(TypedDict):
    messages: Annotated[list, add_messages]
    task_id: Annotated[str | None, lambda old, new: new if new is not None else old]


class ModelResponse(BaseModel):
    status: TaskState = Field(..., example=TaskState.working)
    content: str

class ChunkMetadata(BaseModel):
    message_type: str = Field("message", example="tool_stream")
    step_number: int = Field(0, example=0)
    error: Optional[str] = None


class ChunkResponse(BaseModel):
    status: TaskState = Field(..., example=TaskState.working)
    content: str
    tool_name: Optional[str] = None
    metadata: Optional[ChunkMetadata] = Field(..., example={"message_type": "tool_stream", "step_number": 0, "sub_task_id": "123"})


class ToolEmission(BaseModel):
    tool: str = Field(..., description="The name of the tool that emitted the event", example="tool_name")
    text: str = Field(..., description="The text of the event", example="text")
    state: TaskState = Field(..., description="The status of the event", example=TaskState.working)
    timestamp: str = Field(
        ...,
        description="The timestamp of the event in iso format",
        example="2025-09-08T02:45:12.964656+00:00",
    )


AuthHeaderCb = Callable[[str], dict[str, str]]
