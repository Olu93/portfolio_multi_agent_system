import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from python_a2a import A2AClient
from python_a2a.models.message import Message, MessageRole
from python_a2a.models.content import TextContent

A2A_AGENT_URL = os.getenv("A2A_AGENT_URL", "http://localhost:41241")

# Global variable for A2A client
a2a_client: Optional[A2AClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global a2a_client
    try:
        print(f"Connecting to A2A agent at: {A2A_AGENT_URL}")
        a2a_client = A2AClient(A2A_AGENT_URL)
        print("A2A client initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize A2A client: {e}")
    yield
    if a2a_client:
        try:
            print("Shutting down A2A client...")
        except Exception as e:
            print(f"Error during A2A client shutdown: {e}")

class ChatRequest(BaseModel):
    message: str = Field(..., example="What's the latest news about Apple's stock?")
    task_id: Optional[str] = Field(None, example="1234567890")
    conversation_id: Optional[str] = Field(None, example="1234567890")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    status: str
    error: Optional[str] = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="A2A Chat Service",
        description="FastAPI service for communicating with A2A agents",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "A2A Chat Service is running", "a2a_agent_url": A2A_AGENT_URL}

    @app.get("/health")
    async def health_check(request: Request):
        global a2a_client
        if a2a_client is None:
            return {"status": "unhealthy", "a2a_agent_url": A2A_AGENT_URL, "error": "A2A client not initialized"}
        return {"status": "healthy", "a2a_agent_url": A2A_AGENT_URL}

    @app.post("/chat", response_model=ChatResponse)
    async def chat_with_agent(body: ChatRequest, request: Request):
        global a2a_client
        if a2a_client is None:
            raise HTTPException(status_code=503, detail="A2A client not available")

        conversation_id = body.conversation_id or f"conv_{len(body.message)}_{hash(body.message) % 10000}"
        msg = Message(
            content=TextContent(text=body.message),
            role=MessageRole.USER,
            metadata={"conversation_id": conversation_id, "task_id": body.task_id} if body.task_id else {"conversation_id": conversation_id},
        )

        # Offload sync client call if needed
        try:
            resp = await a2a_client.send_message_async(msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

        if getattr(resp, "error", None):
            err = f"Error from A2A agent: {resp.error}"
            return ChatResponse(response="", conversation_id=conversation_id, status="error", error=err)

        if hasattr(resp, "content") and resp.content:
            response_text = getattr(resp.content, "text", resp.content)
        else:
            response_text = "Response received from agent"

        return ChatResponse(response=str(response_text), conversation_id=conversation_id, status="success")

    @app.get("/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        return {"conversation_id": conversation_id, "message": "Conversation history endpoint - implement as needed"}

    return app


