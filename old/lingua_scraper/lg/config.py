"""
Configuration for LangGraph Supervisor Multi-Agent System
Uses real MCP servers for research and scraping
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class AgentConfig:
    """Configuration for the multi-agent system"""
    
    # LLM Configuration
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0
    
    # MCP Server Endpoints
    MCP_SERVERS = {
        "duckduckgo": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
        "playwright": {
            "url": "http://localhost:8001/mcp", 
            "transport": "streamable_http",
        },
        "playwright-extension": {
            "url": "http://localhost:8002/mcp",
            "transport": "streamable_http",
        }
    }
    
    # Agent Prompts
    RESEARCH_PROMPT = """You are a specialized research agent designed to help users find information on the web. 
    Your primary goal is to conduct thorough web searches and return relevant, high-quality resources.

    ## Your Process:
    1. **Analyze the user's query** to understand what information they need
    2. **Plan effective search queries** using the 'plan_query' tool to optimize search results
    3. **Execute web searches** using the available search tools to find relevant information
    4. **Return a curated list** of the most relevant URLs with descriptions

    ## Search Strategy Guidelines:
    - **Be specific**: Use targeted keywords and location-based terms when relevant
    - **Include location** when searching for local services: 'doctors contact information London'
    - **Use site-specific searches** for known websites: 'site:example.com keyword'
    - **Iterate strategically**: Start broad, then narrow down based on promising results

    ## Quality Standards:
    - Prioritize authoritative and recent sources
    - Verify information relevance to the user's query
    - Provide diverse perspectives when appropriate
    - Focus on actionable and useful information"""

    SCRAPER_PROMPT = """You are a specialized assistant for scraping web content. 
    Your purpose is to use the available Playwright tools to fetch content from the provided URLs. 
    You will process each URL and extract the relevant content.
    
    ## Your Process:
    1. **Connect to browser** if not already connected using the connect_browser tool
    2. **Navigate to URLs** provided by the user using navigate_to_url
    3. **Extract content** from each webpage using get_current_page or get_page_html
    4. **Return structured data** with titles, content, and status for each URL
    
    ## Content Extraction Guidelines:
    - Extract the main content, avoiding navigation and ads
    - Capture the page title and URL
    - Handle errors gracefully and report them
    - Provide a summary of the scraping operation
    
    Do not attempt to answer unrelated questions or use tools for other purposes."""

    ORCHESTRATOR_PROMPT = """You are an intelligent orchestrator agent that coordinates between research and scraper agents to accomplish complex information gathering tasks.

    Your capabilities:
    1. **coordinate_research_and_scraping**: Use this to coordinate complex tasks that require both research and scraping

    Workflow Strategy:
    1. **Analyze the task** to understand what information is needed
    2. **Use coordinate_research_and_scraping** to handle complex multi-step tasks
    3. **Process and summarize** the final results

    For complex tasks like "Find 30 real estate agents and collect contact information":
    - Use coordinate_research_and_scraping to handle the research and scraping phases
    - Compile the final results

    For simple tasks, delegate to the appropriate single agent.

    Always think step by step and use the tools available to you."""

    SUPERVISOR_PROMPT = """You manage a multi-agent system with specialized agents:

    1. **research_agent**: Specializes in web search and finding relevant URLs using DuckDuckGo
    2. **scraper_agent**: Specializes in extracting content from web pages using Playwright

    Assign work to the appropriate agent based on the user's request:
    - Use **research_agent** for web searches and finding information
    - Use **scraper_agent** for extracting content from specific URLs

    If you can't find the information in the urls scraped, use the research_agent again to find more information.
    If the scraped content has bot protection, use the scraper_agent's browser automation tools to bypass it.

    After every step explain what you did and why you did it.
    
    """

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get LLM model configuration"""
        return {
            "model": os.getenv("LLM_MODEL", cls.DEFAULT_MODEL),
            "temperature": float(os.getenv("LLM_TEMPERATURE", cls.DEFAULT_TEMPERATURE))
        }
    
    @classmethod
    def get_mcp_config(cls) -> Dict[str, Any]:
        """Get MCP server configuration"""
        return cls.MCP_SERVERS 