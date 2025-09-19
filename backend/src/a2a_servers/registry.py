# File Name : registry_server.py
# This program Creates In-memory based AI Agents regisry-server using Google A2A protocol
# Author: Sreeni Ramadurai 
# https://dev.to/sreeni5018/building-an-ai-agent-registry-server-with-fastapi-enabling-seamless-agent-discovery-via-a2a-15dj

import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import  HttpUrl
import time
import asyncio
from contextlib import asynccontextmanager
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from python_a2a import AgentCard
from python_a2a.discovery import AgentRegistry


from a2a_servers.a2a_client import async_send_message_streaming
from a2a_servers.utils.models import AgentRegistration, ChatReq, HeartbeatRequest, LookupRequest

logger = logging.getLogger(__name__)    



# Create registry server and FastAPI app
registry_server = AgentRegistry(
    name="A2A Registry Server",
    description="Registry server for agent discovery"
)

# Constants for cleanup
HEARTBEAT_TIMEOUT = 30  # seconds
CLEANUP_INTERVAL = 10   # seconds
A2A_AGENT_URL = os.getenv("A2A_AGENT_URL", "http://ohcm_supervisor_agent:10020")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the cleanup task when the server starts."""
    cleanup_task = asyncio.create_task(cleanup_stale_agents())
    logger.info("Starting A2A Chat Service") 
    yield
    logger.info("Shutting down A2A Chat Service")
    cleanup_task.cancel()

app = FastAPI(title="A2A Agent Registry Server", description="FastAPI server for agent discovery", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def cleanup_stale_agents():
    """Periodically clean up agents that haven't sent heartbeats."""
    while True:
        try:
            current_time = time.time()
            agents_to_remove = []

            # Check each agent's last heartbeat time
            for url, last_seen in registry_server.last_seen.items():
                if current_time - last_seen > HEARTBEAT_TIMEOUT:
                    agents_to_remove.append(url)
                    logger.warning(f"Agent {url} has not sent heartbeat for {HEARTBEAT_TIMEOUT} seconds, removing from registry")

            # Remove stale agents
            for url in agents_to_remove:
                registry_server.unregister_agent(url)
                logger.info(f"Removed stale agent: {url}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        await asyncio.sleep(CLEANUP_INTERVAL)

@app.post("/registry/register", response_model=AgentCard, status_code=201)
async def register_agent(registration: AgentRegistration):
    """Registers a new agent with the registry."""
    agent_card = AgentCard(**registration.model_dump(mode='python'))
    registry_server.register_agent(agent_card)
    return agent_card

@app.get("/registry/agents", response_model=list[AgentCard])
async def list_registered_agents():
    """Lists all currently registered agents."""
    return list(registry_server.get_all_agents())

@app.post("/registry/lookup", response_model=AgentCard)
async def lookup_agent(lookup_request: LookupRequest):
    """Lookup an agent by URL."""
    agent = registry_server.get_agent(lookup_request.url)
    if agent:
        return agent
    raise HTTPException(status_code=404, detail=f"Agent with URL '{lookup_request.url}' not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

@app.post("/registry/heartbeat")
async def heartbeat(request: HeartbeatRequest):
    """Handle agent heartbeat."""
    try:
        if request.url in registry_server.agents:
            registry_server.last_seen[request.url] = time.time()
            logger.info(f"Received heartbeat from agent at {request.url}")
            return {"success": True}
        logger.warning(f"Received heartbeat from unregistered agent: {request.url}")
        return {"success": False, "error": "Agent not registered"}, 404
    except Exception as e:
        logger.error(f"Error processing heartbeat: {e}")
        return {"success": False, "error": str(e)}, 400

@app.get("/registry/agents/{url}", response_model=AgentCard)
async def get_agent(url: HttpUrl):
    """Get a specific agent by URL."""
    agent = registry_server.get_agent(url)
    if agent:
        return agent
    raise HTTPException(status_code=404, detail=f"Agent with URL '{url}' not found")


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

if __name__ == "__main__":
    REGISTRY_HOST = os.getenv("HOST", "0.0.0.0")
    REGISTRY_PORT = int(os.getenv("PORT", 8000))
    logger.info(f"Starting registry server on {REGISTRY_HOST}:{REGISTRY_PORT}")
    uvicorn.run(app, host=REGISTRY_HOST, port=REGISTRY_PORT) 