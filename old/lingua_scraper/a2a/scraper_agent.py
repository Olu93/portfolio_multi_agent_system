import os
import logging
import sys
from typing import Any, Literal, List
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


class URLContent(BaseModel):
    url: str = Field(description="The URL that was scraped")
    title: str = Field(description="The title of the webpage")
    content: str = Field(description="The main content extracted from the webpage")
    status: str = Field(description="The status of the scraping operation (success/error)")
    error_message: str = Field(description="Error message if scraping failed", default="")


class ScrapedData(BaseModel):
    """Response format for scraped content."""
    scraped_urls: List[URLContent]
    summary: str = Field(description="A summary of the scraping operation")
    total_urls: int = Field(description="Total number of URLs processed")
    successful_scrapes: int = Field(description="Number of successfully scraped URLs")


llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
)

@tool
async def download_current_page_html(page) -> str:
    """
    Takes a Playwright Page object and returns the full HTML content.
    """
    return await page.content()


@tool
async def scrape_url(url: str) -> URLContent:
    """Scrape content from a single URL using Playwright"""
    try:
        # This will be handled by the Playwright MCP tool
        # The actual implementation will be provided by the MCP server
        return URLContent(
            url=url,
            title="",
            content="",
            status="pending",
            error_message=""
        )
    except Exception as e:
        return URLContent(
            url=url,
            title="",
            content="",
            status="error",
            error_message=str(e)
        )


class ScraperAgent:
    """ScraperAgent - a specialized assistant for scraping web content."""

    SYSTEM_INSTRUCTION = (
        'You are a specialized assistant for scraping web content. '
        "Your purpose is to use the 'scrape_url' tool to fetch content from the provided URLs. "
        'You will process each URL and extract the relevant content. '
        'Do not attempt to answer unrelated questions or use tools for other purposes.'
    )

    FORMAT_INSTRUCTION = (
        'You will receive a list of URLs from the user and you will scrape content from each URL. '
        'You will use the scrape_url tool to fetch content from each URL. '
        'You will return a list of scraped content with titles and main content for each URL. '
        'You will provide a summary of the scraping operation including success/failure counts.'
    )

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, tools=[]):
        self.model = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0,
        )
        
  
        self.tools = tools
        self._initialized = False
        self.graph = None
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit":10}

    async def ensure_initialized(self):
        """Ensure the agent is fully initialized with all tools"""
        if not self._initialized:
            await self._initialize_tools()
            await self._create_graph()
            self._initialized = True

    async def _initialize_tools(self):
        """Initialize MCP tools if not already done"""
        mcp_client = MultiServerMCPClient(
            {
                "html_fetcher": {
                    "url": "http://localhost:8005/mcp",
                    "transport": "streamable_http"
                }
            }
        )
        mcp_tools = await mcp_client.get_tools()
        self.tools = [*mcp_tools]
        logger.info("Successfully loaded MCP tools")

    async def _create_graph(self):
        """Create the LangGraph React Agent"""
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ScrapedData),
            name="ScraperAgent",
        )

    async def scrape_urls(self, urls: str) -> str:
        """Scrape content from the provided URLs"""
        await self.ensure_initialized()
        
        messages = {"messages":[
            {"role": "system", "content": 
             "You will receive a list of URLs from the user and you will scrape content from each URL."
             "Use your browsertools to find the information you need and scrape the content from the URLs."
             "You will return a list of scraped content with titles and main content for each URL."
             "You will provide a summary of the scraping operation including success/failure counts."
             },
            {"role": "user", "content": f"Please scrape content from these URLs: {urls}"}]}

        return await self.graph.ainvoke(messages, self.config)

    async def process_query(self, query: str) -> str:
        """Process a query using the scraper agent (implements AgentInterface)"""
        # Extract URLs from the query or use the query as URLs
        # For now, assume the query contains URLs
        return await self.scrape_urls(query)


def create_scraper_agent_server(host="localhost", port=5002):
    """Create A2A server for Scraper Agent."""
    # Create the scraper agent
    scraper_agent = ScraperAgent()
    
    # Create the agent skills
    skills = [
        AgentSkill(
            id='scrape',
            name='Web Content Scraper',
            description='Scrapes content from URLs and extracts relevant information',
            tags=['scraping', 'web content', 'data extraction'],
            examples=['Scrape content from these URLs: https://example.com, https://example.org'],
        )
    ]
    
    return BaseAgentExecutor.create_agent_a2a_server(
        agent=scraper_agent,
        name='Scraper Agent',
        description='Scrapes web content from provided URLs using Playwright',
        skills=skills,
        host=host,
        port=port,
        status_message="Scraping content from URLs...",
        artifact_name="scraped_content"
    )


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=5002)
def main(host, port):
    """Starts the Scraper Agent server."""
    run_agent_server(
        create_scraper_agent_server,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == '__main__':
    main() 