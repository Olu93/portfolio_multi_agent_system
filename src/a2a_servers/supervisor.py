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

def _load_sup_cfg():
    """Load supervisor agent YAML -> model+prompt and optional filters."""
    agent_cfg = load_agent_config(SUPERVISOR_AGENT)
    model_cfg = (
        load_model_config(agent_cfg["model"]) if isinstance(agent_cfg.get("model"), str) else agent_cfg.get("model", {})
    )
    prompt = load_prompt_config(agent_cfg.get("prompt_file", "supervisor.txt")) or SYSTEM_PROMPT
    # optional routing limits
    allow_urls = set(agent_cfg.get("allow_urls", []) or [])
    allow_caps = set(agent_cfg.get("allow_caps", []) or [])
    return agent_cfg, model_cfg, prompt, allow_urls, allow_caps

# --- Registry Client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")
REFRESH_SECS = int(os.getenv("DISCOVERY_REFRESH_SECS", "30"))

async def fetch_agents() -> List[AgentCard]:
    async with httpx.AsyncClient(timeout=10) as s:
        r = await s.get(f"{REGISTRY_URL}/registry/agents")
        r.raise_for_status()
        return [AgentCard(**a) for a in r.json()]

# --- Tool factory from AgentCards -------------------------------------------


async def build_tools_from_registry(a2a_client: A2AClient, *, allow_urls: set, allow_caps: set) -> List[Tool]:
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
            payload = _mk_payload(content, context)
            res = await a2a_client.send_request(card.url, payload)
            return json.dumps(res)
        
        tool_name = _safe_name(card.name) or _safe_name(card.url)
        desc_caps = ", ".join(sorted((card.capabilities or {}).keys()))
        summary = f"{card.description or 'A2A agent'}. Caps: {desc_caps or 'unspecified'}"
        
        return Tool(
            name=tool_name,
            description=summary,
            func=tool_impl
        )
    
    cards = await fetch_agents()
    if allow_urls:
        cards = [c for c in cards if c.url in allow_urls]
    if allow_caps:
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]

    tools: List[Tool] = []
    for card in cards:
        tools.append(_create_tool_for_card(card, a2a_client))
    return tools

# --- LangGraph supervisor agent ----------------------------------------------

async def build_supervisor_graph(a2a_client: A2AClient):
    agent_cfg, model_cfg, prompt, allow_urls, allow_caps = _load_sup_cfg()
    model = init_chat_model(
        model_cfg["name"],
        **model_cfg.get("parameters", {}),
        model_provider=model_cfg.get("provider"),
    )
    tools = await build_tools_from_registry(a2a_client, allow_urls=allow_urls, allow_caps=allow_caps)
    return create_react_agent(model, tools, prompt=prompt)

# --- Public API ---------------------------------------------------------------
class Supervisor:
    def __init__(self, graph=None):
        self.graph = graph
        self._lock = asyncio.Lock()
        self._last_refresh = 0
        self.a2a_client = A2AClient()

    async def _ensure_graph(self):
        now = time.time()
        if self.graph is None or (now - self._last_refresh) > REFRESH_SECS:
            async with self._lock:
                if self.graph is None or (time.time() - self._last_refresh) > REFRESH_SECS:
                    self.graph = await build_supervisor_graph(self.a2a_client)
                    self._last_refresh = time.time()

    async def handle_task(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        await self._ensure_graph()
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", content)], "context": context or {}}
        result = await self.graph.ainvoke(inputs)
        msgs = result.get("messages", [])
        final = msgs[-1].content if msgs else ""
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
