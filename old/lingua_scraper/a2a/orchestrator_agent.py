import asyncio
import base64
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any, List
from uuid import uuid4

import click
import httpx
import uvicorn

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Task,
    TaskState,
    TextPart,
    JSONRPCErrorResponse,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater, InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    InternalError,
    InvalidParamsError,
    UnsupportedOperationError,
    AgentCapabilities,
    AgentCard as A2AAgentCard,
    AgentSkill,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from dotenv import load_dotenv, find_dotenv

# LangGraph imports
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .base_agent_executor import BaseAgentExecutor, run_agent_server
from .a2a_tool_client import A2AToolClient

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory = InMemorySaver()


TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]


class OrchestratorAgent:
    """The intelligent orchestrator agent using LangGraph React Agent."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(
        self,
        remote_agent_addresses: List[str],
        task_callback: TaskUpdateCallback | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.task_callback = task_callback
        self.httpx_client = http_client or httpx.AsyncClient()
        self.remote_agent_addresses = remote_agent_addresses
        self.a2a_client = A2AToolClient(httpx_client=self.httpx_client)
        self._initialized = False
        
        # Initialize LLM for the orchestrator
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0,
        )
        
        # Initialize tools and graph (will be set up after agent initialization)
        self.tools = []
        self.graph = None
        self.config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit":15}

    async def ensure_initialized(self):
        """Ensure the agent connections are initialized"""
        if not self._initialized:
            await self.init_remote_agent_addresses(self.remote_agent_addresses)
            await self.setup_tools_and_graph()
            self._initialized = True

    async def init_remote_agent_addresses(
        self, remote_agent_addresses: List[str]
    ):
        """Initialize A2A client with remote agents"""
        for address in remote_agent_addresses:
            self.a2a_client.add_remote_agent(address)

    async def setup_tools_and_graph(self):
        """Set up the tools and LangGraph agent after remote agents are initialized"""
        # Create tools using A2AToolClient methods directly (following notebook pattern)
        self.tools = [
            self.a2a_client.list_remote_agents,
            self.a2a_client.create_task,
        ]
        
        # Create the LangGraph React Agent
        self.graph = create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.get_system_prompt(),
            name="OrchestratorAgent",
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the orchestrator"""
        return """You are an intelligent orchestrator agent that coordinates between research and scraper agents to accomplish complex information gathering tasks.

Your capabilities:
1. **list_remote_agents**: Use this to discover available remote agents and their capabilities
2. **create_task**: Use this to send tasks to specific remote agents

Workflow Strategy:
1. **Analyze the task** to understand what information is needed
2. **Use list_remote_agents** to discover available agents and their capabilities
3. **Use create_task** to send appropriate tasks to the right agents
4. **Process and summarize** the final results

For complex tasks like "Find 30 real estate agents and collect contact information":
- First use list_remote_agents to see what agents are available
- Use create_task to send a research query to find real estate agent websites
- Extract URLs from the research results
- Use create_task to send scraping requests for those URLs
- Compile the final results

For simple tasks, use the appropriate single agent.

Always think step by step and use the tools available to you."""

    async def process_query(self, query: str) -> str:
        """Process a query using the orchestrator agent"""
        await self.ensure_initialized()
        
        messages = {"messages": [{"role": "user", "content": query}]}
        result = await self.graph.ainvoke(messages, self.config)
        
        # Extract the final response from the result
        if 'messages' in result and result['messages']:
            final_message: AIMessage = result['messages'][-1]
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(result)
        else:
            return str(result)


def create_orchestrator_agent_server(host="localhost", port=5000):
    """Create A2A server for Orchestrator Agent."""
    # Create the orchestrator agent
    orchestrator_agent = OrchestratorAgent(
        remote_agent_addresses=[
            "http://localhost:5001",  # Research Agent
            "http://localhost:5002",  # Scraper Agent
        ]
    )
    
    # Create the agent skills
    skills = [
        AgentSkill(
            id='intelligent_executor',
            name='Intelligent Task Executor',
            description='Orchestrates complex workflows by coordinating research and scraper agents to gather comprehensive information. Can handle tasks like finding real estate agent websites and collecting contact information.',
            tags=['intelligent orchestration', 'workflow coordination', 'research', 'scraping', 'data gathering'],
            examples=[
                'Find a list of 30 real estate agent websites that are not platforms and collect their contact information',
                'Find sustainable energy companies in Rotterdam and extract their contact details',
                'Research local restaurants and gather their menu information'
            ],
        )
    ]
    
    return BaseAgentExecutor.create_agent_a2a_server(
        agent=orchestrator_agent,
        name='Intelligent Orchestrator Agent',
        description='Intelligently coordinates research and scraper agents to accomplish complex information gathering tasks',
        skills=skills,
        host=host,
        port=port,
        status_message="Orchestrating task execution...",
        artifact_name="orchestration_result"
    )


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=5000)
@click.option('--research-agent', 'research_agent_url', default='http://localhost:5001')
@click.option('--scraper-agent', 'scraper_agent_url', default='http://localhost:5002')
def main(host, port, research_agent_url, scraper_agent_url):
    """Starts the Orchestrator Agent server."""
    run_agent_server(
        create_orchestrator_agent_server,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == '__main__':
    main()
