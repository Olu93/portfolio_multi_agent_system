import os
import logging
import sys
from typing import Any, Literal, List, Optional
import uuid
import asyncio

import click
import httpx
import uvicorn

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
import mcp
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv, find_dotenv

from .base_agent_executor import BaseAgentExecutor, run_agent_server

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory = InMemorySaver()


class Query(BaseModel):
    query: str = Field(description="The query to search the web for information")
    reasoning: str = Field(description="The reasoning for the query")


class WebResource(BaseModel):
    url: str = Field(description="The url of the web resource")
    title: str = Field(description="The title of the web resource")
    description: str = Field(description="The description of the web resource")
    query: Optional[Query] = Field(
        description="The query that was used to find the web resource", default=None
    )


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    urls: List[WebResource]
    message: str


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


@tool
async def plan_query(query: str) -> Query:
    """Plan a query to search the web for information"""
    parser = JsonOutputParser(pydantic_object=Query)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research agent that can plan queries to search the web for information."
                "Plan a query to search the web for information",
            ),
            ("system", "{format_instructions}"),
            ("user", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | JsonOutputParser(pydantic_object=Query)
    return await chain.ainvoke({"query": query})


class ResearchAgent:
    """ResearchAgent - a specialized assistant for research."""

    SYSTEM_INSTRUCTION = (
        "You are a specialized research agent designed to help users find information on the web. "
        "Your primary goal is to conduct thorough web searches and return relevant, high-quality resources.\n\n"
        
        "## Your Process:\n"
        "1. **Analyze the user's query** to understand what information they need\n"
        "2. **Plan effective search queries** using the 'plan_query' tool to optimize search results\n"
        "3. **Execute web searches** using the 'search' tool to find relevant information\n"
        "4. **Iterate and refine** your search strategy based on initial results\n"
        "5. **Return a curated list** of the most relevant URLs with descriptions\n\n"
        
        "## Search Strategy Guidelines:\n"
        "- **Be specific**: Use targeted keywords and location-based terms when relevant\n"
        "- **Include location** when searching for local services: 'doctors contact information London'\n"
        "- **Use site-specific searches** for known websites gathered from previous searches: 'site:example.com keyword'\n"
        "- **Iterate strategically**: Start broad, then narrow down based on promising results\n\n"
        
        "## Examples of Effective Queries:\n"
        "- Real estate: 'real estate agent' 'Rotterdam' 'Netherlands'\n"
        "- Professional services: 'lawyer' 'Berlin' 'Germany' 'contact information'\n"
        "- Specific searches: 'site:linkedin.com' 'lawyer' 'John Doe' 'Berlin'\n"
        "- Research topics: 'latest research' 'sustainable energy' '2024'\n\n"
        
        "## Quality Standards:\n"
        "- Prioritize authoritative and recent sources\n"
        "- Verify information relevance to the user's query\n"
        "- Provide diverse perspectives when appropriate\n"
        "- Focus on actionable and useful information"
    )

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, tools=[]):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

        # Use provided tools or default to just plan_query
        if not tools:
            tools = [plan_query]
        else:
            tools = [*tools, plan_query]

        self.tools = tools
        self._initialized = False
        self.graph = None
        self.config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 15,
        }

    async def ensure_initialized(self):
        """Ensure the agent is fully initialized with all tools"""
        if not self._initialized:
            await self._initialize_tools()
            await self._create_graph()
            self._initialized = True

    async def _initialize_tools(self):
        """Initialize MCP tools if not already done"""
        if len(self.tools) == 1:  # Only has plan_query
            try:
                mcp_client = MultiServerMCPClient(
                    {
                        "duckduckgo": {
                            "url": "http://localhost:8000/mcp",
                            "transport": "streamable_http",
                        },
                    }
                )
                search_tools = await mcp_client.get_tools()
                self.tools = [*search_tools, plan_query]
                logger.info("Successfully loaded MCP tools")
            except Exception as e:
                logger.warning(
                    f"Failed to load MCP tools: {e}. Using basic tools only."
                )
                self.tools = [plan_query]

    async def _create_graph(self):
        """Create the LangGraph React Agent"""
        parser = JsonOutputParser(pydantic_object=ResponseFormat)
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(parser.get_format_instructions(), ResponseFormat),
            name="ResearchAgent",
        )

    async def search(self, query: str) -> str:
        """Search the web for information"""
        await self.ensure_initialized()

        messages = {"messages": [{"role": "user", "content": query}]}

        return await self.graph.ainvoke(messages, self.config)

    async def process_query(self, query: str) -> str:
        """Process a query using the research agent (implements AgentInterface)"""
        return await self.search(query)


def create_research_agent_server(host="localhost", port=5001):
    """Create A2A server for Research Agent."""
    # Create the research agent
    research_agent = ResearchAgent()

    # Create the agent skills
    skills = [
        AgentSkill(
            id="search",
            name="Web Research Tool",
            description="Searches the web for information and provides relevant URLs",
            tags=["research", "web search", "information gathering"],
            examples=["Find me websites with real estate listings in Rotterdam"],
        )
    ]

    return BaseAgentExecutor.create_agent_a2a_server(
        agent=research_agent,
        name="Research Agent",
        description="Provides research information by searching the web",
        skills=skills,
        host=host,
        port=port,
        status_message="Searching for information...",
        artifact_name="research_result",
    )


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=5001)
def main(host, port):
    """Starts the Research Agent server."""
    run_agent_server(
        create_research_agent_server, host=host, port=port, log_level="info"
    )


if __name__ == "__main__":
    main()
