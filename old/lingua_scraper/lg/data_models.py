"""
Data Models for LangGraph Supervisor Multi-Agent System
Response format classes for different agent types
"""

from typing import List
from pydantic import BaseModel, Field


# ============================================================================
# Research Agent Response Models
# ============================================================================

class WebResource(BaseModel):
    """Individual web resource found by research agent"""
    url: str = Field(description="The URL of the web resource")
    title: str = Field(description="The title of the web resource")
    description: str = Field(description="The description of the web resource")


class ResearchResponse(BaseModel):
    """Response format for research results"""
    urls: List[WebResource] = Field(description="List of relevant web resources found")
    message: str = Field(description="Summary message about the research results")


# ============================================================================
# Scraper Agent Response Models
# ============================================================================

class URLContent(BaseModel):
    """Content extracted from a single URL"""
    url: str = Field(description="The URL that was scraped")
    title: str = Field(description="The title of the webpage")
    content: str = Field(description="The main content extracted from the webpage")
    status: str = Field(description="The status of the scraping operation (success/error)")


class ScrapedData(BaseModel):
    """Response format for scraped content"""
    scraped_urls: List[URLContent] = Field(description="List of scraped URLs with their content")
    summary: str = Field(description="A summary of the scraping operation") 