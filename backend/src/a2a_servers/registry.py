# File Name : registry_server.py
# This program Creates In-memory based AI Agents regisry-server using Google A2A protocol
# Author: Sreeni Ramadurai
# https://dev.to/sreeni5018/building-an-ai-agent-registry-server-with-fastapi-enabling-seamless-agent-discovery-via-a2a-15dj

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from a2a.types import AgentCard
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import HttpUrl
# from python_a2a.discovery import AgentRegistry
# from python_a2a import AgentCard as PythonA2AAgentCard
from a2a_servers.a2a_client import async_send_message_streaming
from a2a_servers.utils.models import AgentRegistration, ChatReq, HeartbeatRequest, LookupRequest

logger = logging.getLogger(__name__)

class AgentRegistry:
    last_seen: dict[str, float] = {}
    agents: dict[str, AgentCard] = {}
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def register_agent(self, agent_card: AgentCard) -> bool:
        # Remove trailing slash from url
        agent_card.url = agent_card.url.rstrip("/") if agent_card.url[-1] == "/" else agent_card.url
        self.agents[agent_card.url] = agent_card
        self.last_seen[agent_card.url] = time.time()
        return True
    
    def unregister_agent(self, url: str|HttpUrl) -> bool:
        url = str(url)
        url = url.rstrip("/") if url[-1] == "/" else url
        self.agents.pop(url, None)
        self.last_seen.pop(url, None)
        return True
    
    def get_agent(self, url: str|HttpUrl) -> AgentCard | None:
        url = str(url)
        url = url.rstrip("/") if url[-1] == "/" else url
        agent = self.agents.get(url, None)
        if agent:
            self.last_seen[url] = time.time()
            return agent
        return None
    
    def get_all_agents(self) -> list[AgentCard]:
        agents = list(self.agents.values())
        for agent in agents:
            self.last_seen[agent.url] = time.time()
        return agents
    
    def get_all_agents_urls(self) -> list[str]:
        urls = list(self.agents.keys())
        for url in urls:
            self.last_seen[url] = time.time()
        return urls

# Create registry server and FastAPI app
registry_server = AgentRegistry(name="A2A Registry Server", description="Registry server for agent discovery")

# Constants for cleanup
HEARTBEAT_TIMEOUT = 30  # seconds
CLEANUP_INTERVAL = 10  # seconds
A2A_AGENT_URL = os.getenv("A2A_AGENT_URL", "http://ohcm_supervisor_agent:10020")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the cleanup task when the server starts."""
    logger.info("Starting A2A Registry Service")
    cleanup_task = asyncio.create_task(cleanup_stale_agents())
    logger.info("Registry service startup complete, cleanup task started")
    yield
    logger.info("Shutting down A2A Registry Service")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Cleanup task cancelled successfully")


app = FastAPI(
    title="A2A Agent Registry Server",
    description="FastAPI server for agent discovery",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def _convert_agent_registration_to_agent_card(registration: AgentCard|None) -> AgentCard|None:
#     if registration is None:
#         return None
#     agent_card_dict = registration.model_dump(mode="python")
#     agent_card_dict["url"] = str(registration.url)
#     agent_card = AgentCard(**agent_card_dict)
#     return agent_card

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
async def register_agent(registration: AgentCard):
    """Registers a new agent with the registry."""
    logger.info(f"Registering agent: {registration.name} at {registration.url}")
    try:

        agent_card = registration
        is_ok = registry_server.register_agent(agent_card)
        if not is_ok:
            logger.error(f"Failed to register agent: {registration.name} at {registration.url}")
            raise HTTPException(status_code=400, detail=f"Registration failed: {registration.name} at {registration.url}")
        logger.info(f"Successfully registered agent: {registration.name} at {registration.url}")
        return agent_card
    except Exception as e:
        logger.error(f"Failed to register agent {registration.name}: {e}")
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")


@app.get("/registry/agents", response_model=list[AgentCard])
async def list_registered_agents():
    """Lists all currently registered agents."""
    logger.info("Listing all registered agents")
    agents = registry_server.get_all_agents()
    logger.info(f"Found {len(agents)} registered agents")
    return agents


@app.post("/registry/lookup", response_model=AgentCard)
async def lookup_agent(lookup_request: LookupRequest):
    """Lookup an agent by URL."""
    logger.info(f"Looking up agent with URL: {lookup_request.url}")
    agent = registry_server.get_agent(lookup_request.url)
    if agent:
        logger.info(f"Found agent: {agent.name} at {lookup_request.url}")
        return agent
    logger.warning(f"Agent not found with URL: {lookup_request.url}")
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
        url = str(request.url)
        url = url.rstrip("/") if url[-1] == "/" else url
        if url in registry_server.agents:
            registry_server.last_seen[url] = time.time()
            logger.info(f"Received heartbeat from agent at {url}")
            return {"success": True}
        logger.warning(f"Received heartbeat from unregistered agent: {url}")
        return {"success": False, "error": "Agent not registered"}, 404
    except Exception as e:
        logger.error(f"Error processing heartbeat: {e}")
        return {"success": False, "error": str(e)}, 400


@app.get("/registry/agents/{url}", response_model=AgentCard)
async def get_agent(url: HttpUrl):
    """Get a specific agent by URL."""
    logger.info(f"Getting agent by URL: {url}")
    agent = registry_server.get_agent(url)
    if agent:
        logger.info(f"Retrieved agent: {agent.name} at {url}")
        return agent
    logger.warning(f"Agent not found with URL: {url}")
    raise HTTPException(status_code=404, detail=f"Agent with URL '{url}' not found")


@app.post("/chat/stream")
async def chat_stream(body: ChatReq):
    logger.info(f"Starting chat stream for context_id: {body.context_id}, task_id: {body.task_id}")
    agent_url = A2A_AGENT_URL

    async def gen():
        try:
            async for line in async_send_message_streaming(
                agent_url=agent_url,
                message=body.message,
                context_id=body.context_id,
                task_id=body.task_id,
            ):
                yield line
        except Exception as e:
            logger.error(f"Error in chat stream for context_id {body.context_id}: {e}")
            error_response = json.dumps({"error": str(e), "context_id": body.context_id})
            yield error_response

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat")
async def chat_non_stream(body: ChatReq):
    logger.info(f"Starting non-streaming chat for context_id: {body.context_id}, task_id: {body.task_id}")
    agent_url = A2A_AGENT_URL
    lines = []
    try:
        async for line in async_send_message_streaming(
            agent_url=agent_url,
            message=body.message,
            context_id=body.context_id,
            task_id=body.task_id,
        ):
            lines.append(json.loads(line))
        logger.info(f"Completed non-streaming chat for context_id: {body.context_id}, received {len(lines)} chunks")
        return {"chunks": lines}
    except Exception as e:
        logger.error(f"Error in non-streaming chat for context_id {body.context_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


if __name__ == "__main__":
    REGISTRY_HOST = os.getenv("HOST", "0.0.0.0")
    REGISTRY_PORT = int(os.getenv("PORT", 8000))
    logger.info(f"Starting registry server on {REGISTRY_HOST}:{REGISTRY_PORT}")
    uvicorn.run(app, host=REGISTRY_HOST, port=REGISTRY_PORT)
