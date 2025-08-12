"""
Agent definitions for LangGraph Supervisor Multi-Agent System
Uses real MCP servers for research and scraping
"""

import asyncio
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import AgentConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class Query(BaseModel):
    query: str = Field(description="The query to search the web for information")
    reasoning: str = Field(description="The reasoning for the query")


class WebResource(BaseModel):
    url: str = Field(description="The url of the web resource")
    title: str = Field(description="The title of the web resource")
    description: str = Field(description="The description of the web resource")


class ResearchResponse(BaseModel):
    """Response format for research results."""
    urls: List[WebResource]
    message: str


class URLContent(BaseModel):
    url: str = Field(description="The URL that was scraped")
    title: str = Field(description="The title of the webpage")
    content: str = Field(description="The main content extracted from the webpage")
    status: str = Field(description="The status of the scraping operation (success/error)")


class ScrapedData(BaseModel):
    """Response format for scraped content."""
    scraped_urls: List[URLContent]
    summary: str = Field(description="A summary of the scraping operation")


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
async def plan_query(query: str) -> Query:
    """Plan a query to search the web for information"""
    parser = JsonOutputParser(pydantic_object=Query)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research agent that can plan queries to search the web for information. Plan a query to search the web for information"),
        ("system", "{format_instructions}"),
        ("user", "{query}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    model_config = AgentConfig.get_model_config()
    llm = ChatOpenAI(**model_config)
    chain = prompt | llm | JsonOutputParser(pydantic_object=Query)
    return await chain.ainvoke({"query": query})


@tool
def coordinate_research_and_scraping(research_query: str, urls_to_scrape: str) -> str:
    """Coordinate between research and scraping tasks"""
    return f"Coordinated task: Researched '{research_query}' and scraped content from {urls_to_scrape}"


# ============================================================================
# Agent Factory Functions
# ============================================================================

async def create_research_agent():
    """Create the research agent with web search capabilities using MCP"""
    
    async def initialize_research_tools():
        """Initialize MCP tools for research"""
        try:
            mcp_config = AgentConfig.get_mcp_config()
            mcp_client = MultiServerMCPClient({
                "duckduckgo": mcp_config["duckduckgo"]
            })
            search_tools = await mcp_client.get_tools()
            return [*search_tools, plan_query]
        except Exception as e:
            logger.warning(f"Failed to load MCP tools: {e}. Using basic tools only.")
            return [plan_query]
    
    # Initialize tools
    tools = await initialize_research_tools()
    
    # Create response parser
    parser = JsonOutputParser(pydantic_object=ResearchResponse)
    
    # Get model configuration
    model_config = AgentConfig.get_model_config()
    
    return create_react_agent(
        model=model_config["model"],
        tools=tools,
        prompt=AgentConfig.RESEARCH_PROMPT,
        response_format=(parser.get_format_instructions(), ResearchResponse),
        name="research_agent"
    )


async def create_scraper_agent():
    """Create the scraper agent with web scraping capabilities using MCP"""
    
    async def initialize_scraper_tools():
        """Initialize MCP tools for scraping"""
        try:
            mcp_config = AgentConfig.get_mcp_config()
            mcp_client = MultiServerMCPClient({
                "playwright": mcp_config["playwright"],
                "playwright-extension": mcp_config["playwright-extension"]
            })
            scraper_tools = await mcp_client.get_tools()
            return scraper_tools
        except Exception as e:
            logger.warning(f"Failed to load Playwright MCP tools: {e}. Using basic tools only.")
            return []
    
    # Initialize tools
    tools = await initialize_scraper_tools()
    
    # Create response parser
    parser = JsonOutputParser(pydantic_object=ScrapedData)
    
    # Get model configuration
    model_config = AgentConfig.get_model_config()
    
    return create_react_agent(
        model=model_config["model"],
        tools=tools,
        prompt=AgentConfig.SCRAPER_PROMPT,
        response_format=(parser.get_format_instructions(), ScrapedData),
        name="scraper_agent"
    )


async def create_orchestrator_agent():
    """Create the orchestrator agent with coordination capabilities"""
    
    orchestrator_tools = [coordinate_research_and_scraping]
    
    # Get model configuration
    model_config = AgentConfig.get_model_config()
    
    return create_react_agent(
        model=model_config["model"],
        tools=orchestrator_tools,
        prompt=AgentConfig.ORCHESTRATOR_PROMPT,
        name="orchestrator_agent"
    ) 