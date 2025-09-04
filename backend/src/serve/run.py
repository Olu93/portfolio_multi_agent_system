import os, json, asyncio
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- A2A SDK imports (adjust paths to your SDK) ---
from a2a.client import A2AClient
from a2a.client import A2ACardResolver
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH
from a2a.types import MessageSendParams, SendStreamingMessageRequest
from a2a.utils import new_agent_text_message
import logging

logger = logging.getLogger(__name__)   

# --------- request models ----------
class ChatReq(BaseModel):
    message: str = Field(..., example="What's the latest news about Apple's stock?")
    context_id: Optional[str] = Field(None, example="123")
    task_id: Optional[str] = Field(None, example="123")

app = FastAPI(title="A2A Chat Service", version="1.0.0")
A2A_AGENT_URL = os.getenv("A2A_AGENT_URL", "http://ohcm_supervisor_agent:10020")
logger.info(f"A2A_AGENT_URL: {A2A_AGENT_URL}")



# --------- per-request streaming helper ----------
async def async_send_message_streaming(
    agent_url: str, message: str, context_id: str | None, task_id: str | None
):
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
            yield json.dumps(chunk.model_dump(mode="json", exclude_none=True)) + "\n"


# --------- endpoints ----------
@app.post("/chat/stream")
async def chat_stream(body: ChatReq):
    agent_url = A2A_AGENT_URL

    async def gen():
        async for line in async_send_message_streaming(
            agent_url=agent_url,
            message=body.message,
            context_id=body.context_id,
            task_id=body.task_id,
        ):
            yield line

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat")
async def chat_non_stream(body: ChatReq):
    agent_url = A2A_AGENT_URL
    lines = []
    async for line in async_send_message_streaming(
        agent_url=agent_url,
        message=body.message,
        context_id=body.context_id,
        task_id=body.task_id,
    ):
        lines.append(json.loads(line))
    return {"chunks": lines}
