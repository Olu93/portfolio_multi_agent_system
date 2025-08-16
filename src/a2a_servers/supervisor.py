# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (python-a2a)
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional, List, Callable

import click
import httpx
from python_a2a import A2AClient, AgentCard  # python-a2a client
from a2a_servers.config_loader import load_agent_config, load_model_config, load_prompt_config

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# --- Defaults / constants ----------------------------------------------------
SYSTEM_PROMPT = (
    "You are the Supervisor. Choose the most relevant TOOL based on the task and the tool descriptions (capabilities). "
    "You may call multiple tools sequentially if needed and then produce a concise final answer."
)
SUPERVISOR_AGENT = os.getenv("SUPERVISOR_AGENT", "supervisor")  # which agent YAML to load

# --- Config helper -----------------------------------------------------------

# --- Registry Client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")
REFRESH_SECS = int(os.getenv("DISCOVERY_REFRESH_SECS", "30"))

async def fetch_agents() -> List[AgentCard]:
    logger.info(f"Fetching agents from registry at {REGISTRY_URL}")
    async with httpx.AsyncClient(timeout=10) as s:
        r = await s.get(f"{REGISTRY_URL}/registry/agents")
        r.raise_for_status()
        agents = [AgentCard(**a) for a in r.json()]
        logger.info(f"Discovered {len(agents)} agents from registry")
        return agents

# --- Tool factory from AgentCards -------------------------------------------

async def build_tools_from_registry(a2a_client: A2AClient, *, allow_urls: set, allow_caps: set) -> List[Tool]:
    logger.info(f"Building tools from registry with allow_urls={allow_urls}, allow_caps={allow_caps}")
    
    def _mk_payload(content: str, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": content,
            "context": {"task_id": str(uuid.uuid4()), "timestamp": int(time.time()), **(ctx or {})},
        }

    def _safe_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()
    
    def _create_tool_for_card(card: AgentCard, a2a_client: A2AClient) -> Tool:
        """Create a tool function for a specific agent card"""
        async def tool_impl(content: str, context: Optional[Dict[str, Any]] = None):
            logger.debug(f"Tool {card.name} called with content: {content[:100]}...")
            payload = _mk_payload(content, context)
            res = await a2a_client.send_request(card.url, payload)
            logger.debug(f"Tool {card.name} returned response: {str(res)[:200]}...")
            return json.dumps(res)
        
        tool_name = _safe_name(card.name) or _safe_name(card.url)
        desc_caps = ", ".join(sorted((card.capabilities or {}).keys()))
        summary = f"{card.description or 'A2A agent'}. Caps: {desc_caps or 'unspecified'}"
        
        logger.debug(f"Creating tool '{tool_name}' for agent '{card.name}' at {card.url}")
        return Tool(
            name=tool_name,
            description=summary,
            func=tool_impl
        )
    
    cards = await fetch_agents()
    logger.info(f"Processing {len(cards)} agent cards")
    
    if allow_urls:
        original_count = len(cards)
        cards = [c for c in cards if c.url in allow_urls]
        logger.info(f"Filtered by allow_urls: {original_count} -> {len(cards)} agents")
    
    if allow_caps:
        original_count = len(cards)
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]
        logger.info(f"Filtered by allow_caps: {original_count} -> {len(cards)} agents")

    tools: List[Tool] = []
    for card in cards:
        tools.append(_create_tool_for_card(card, a2a_client))
    
    logger.info(f"Successfully created {len(tools)} tools from registry")
    return tools

# --- LangGraph supervisor agent ----------------------------------------------

async def build_supervisor_graph(a2a_client: A2AClient):
    logger.info("Building supervisor graph")
    
    # Load configuration directly
    logger.debug(f"Loading configuration for supervisor agent: {SUPERVISOR_AGENT}")
    agent_cfg = load_agent_config(SUPERVISOR_AGENT)
    logger.debug(f"Agent config loaded: {agent_cfg}")
    
    model_cfg = (
        load_model_config(agent_cfg["model"]) if isinstance(agent_cfg.get("model"), str) else agent_cfg.get("model", {})
    )
    logger.debug(f"Model config loaded: {model_cfg}")
    
    prompt = load_prompt_config(agent_cfg.get("prompt_file", "supervisor.txt")) or SYSTEM_PROMPT
    logger.debug(f"Prompt loaded, length: {len(prompt)} characters")
    
    # optional routing limits
    allow_urls = set(agent_cfg.get("allow_urls", []) or [])
    allow_caps = set(agent_cfg.get("allow_caps", []) or [])
    logger.info(f"Routing limits: allow_urls={allow_urls}, allow_caps={allow_caps}")
    
    logger.info(f"Initializing chat model: {model_cfg.get('name', 'unknown')}")
    model = init_chat_model(
        model_cfg["name"],
        **model_cfg.get("parameters", {}),
        model_provider=model_cfg.get("provider"),
    )
    
    logger.info("Building tools from registry")
    tools = await build_tools_from_registry(a2a_client, allow_urls=allow_urls, allow_caps=allow_caps)
    
    logger.info("Creating react agent with tools and prompt")
    return create_react_agent(model, tools, prompt=prompt)

# --- Public API ---------------------------------------------------------------
class Supervisor:
    def __init__(self, graph=None):
        self.graph = graph
        self._lock = asyncio.Lock()
        self._last_refresh = 0
        self.a2a_client = A2AClient()
        logger.info("Supervisor instance initialized")

    async def _ensure_graph(self):
        now = time.time()
        if self.graph is None or (now - self._last_refresh) > REFRESH_SECS:
            logger.debug("Graph needs refresh or initialization")
            async with self._lock:
                if self.graph is None or (time.time() - self._last_refresh) > REFRESH_SECS:
                    logger.info("Building new supervisor graph")
                    self.graph = await build_supervisor_graph(self.a2a_client)
                    self._last_refresh = time.time()
                    logger.info("Supervisor graph built and ready")
                else:
                    logger.debug("Graph was refreshed by another task")

    async def handle_task(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Handling task: {content[:100]}...")
        logger.debug(f"Task context: {context}")
        
        await self._ensure_graph()
        
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", content)], "context": context or {}}
        logger.debug("Invoking supervisor graph")
        
        result = await self.graph.ainvoke(inputs)
        logger.debug(f"Graph invocation completed, result keys: {list(result.keys())}")
        
        msgs = result.get("messages", [])
        final = msgs[-1].content if msgs else ""
        logger.info(f"Task completed, final response length: {len(final)}")
        
        return {"ok": True, "content": final, "raw": result}

# --- CLI entrypoint -----------------------------------------------------------

@click.command()
@click.option("--agent-name", default="supervisor", help="Name of the supervisor agent config to load")
@click.option("--host", default="0.0.0.0", help="Host to run the supervisor on")
@click.option("--port", default=10030, help="Port to run the supervisor on")
@click.option("--log-level", default="info", help="Log level")
def run_supervisor(agent_name: str, host: str, port: int, log_level: str):
    """Run the Supervisor agent server."""
    os.environ["SUPERVISOR_AGENT"] = agent_name
    logger.info(f"Starting Supervisor agent '{agent_name}' on {host}:{port}")
    sup = Supervisor()
    asyncio.run(sup._ensure_graph())
    logger.info("Supervisor is initialized and ready.")

if __name__ == "__main__":
    run_supervisor()
