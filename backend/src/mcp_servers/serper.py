import os
from fastmcp import FastMCP, Context
from mcp_servers.utils.models import MCPResponse
from mcp_servers.utils.helper import log, start_mcp_server
from mcp_servers.utils.constants import MCP_HOST, MCP_PORT
import logging
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import List, Optional
from dataclasses import dataclass
import sys
import traceback
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int
    source: Optional[str] = None
    date: Optional[str] = None


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req for req in self.requests if now - req < timedelta(minutes=1)]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class GoogleSerperSearcher:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.serper_api_key = SERPER_API_KEY

        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY environment variable is required. Please set it with your Serper API key from https://serper.dev")

        # Initialize the LangChain wrapper
        self.search_wrapper = GoogleSerperAPIWrapper(serper_api_key=self.serper_api_key)

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. Please try rephrasing your search or check your API key."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            if result.source:
                output.append(f"   Source: {result.source}")
            if result.date:
                output.append(f"   Date: {result.date}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(
        self,
        query: str,
        ctx: Context,
        max_results: int = 10,
        search_type: str = "search",
        time_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform a search using Google Serper API via LangChain

        Args:
            query: Search query
            ctx: MCP context
            max_results: Maximum number of results to return
            search_type: Type of search ("search", "news", "places", "images")
            time_filter: Time filter for news searches (e.g., "qdr:h", "qdr:d", "qdr:w", "qdr:m", "qdr:y")
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            await log(f"Searching Google Serper for: {query} (type: {search_type})", "info", logger, ctx)

            # Configure search parameters
            search_params = {"type": search_type, "num": max_results}

            # Add time filter for news searches
            if search_type == "news" and time_filter:
                search_params["tbs"] = time_filter

            # Create a new wrapper instance with the parameters
            search_wrapper = GoogleSerperAPIWrapper(serper_api_key=self.serper_api_key, **search_params)

            # Perform the search
            results_data = search_wrapper.results(query)

            # Parse results based on search type
            results = []

            if search_type == "news":
                news_results = results_data.get("news", [])
                for i, item in enumerate(news_results[:max_results]):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            position=i + 1,
                            source=item.get("source", ""),
                            date=item.get("date", ""),
                        )
                    )
            elif search_type == "places":
                places_results = results_data.get("places", [])
                for i, item in enumerate(places_results[:max_results]):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            link=item.get("website", ""),
                            snippet=f"Address: {item.get('address', '')} | Rating: {item.get('rating', 'N/A')} | Category: {item.get('category', '')}",
                            position=i + 1,
                            source=item.get("category", ""),
                        )
                    )
            elif search_type == "images":
                images_results = results_data.get("images", [])
                for i, item in enumerate(images_results[:max_results]):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            position=i + 1,
                            source=item.get("source", ""),
                        )
                    )
            else:  # Regular search
                organic_results = results_data.get("organic", [])
                for i, item in enumerate(organic_results[:max_results]):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            link=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            position=i + 1,
                            source=item.get("source", ""),
                        )
                    )

            await log(f"Successfully found {len(results)} results", "info", logger, ctx)
            return results

        except Exception as e:
            await log(f"Error during search: {str(e)}", "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return []


# Initialize the MCP server
mcp = FastMCP("google-serper", host=MCP_HOST, port=MCP_PORT)

# Initialize the searcher
searcher = GoogleSerperSearcher()


@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 10, search_type: str = "search") -> MCPResponse:
    """
    Search the web using Google Serper API

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)
        search_type: Type of search - "search", "news", "places", or "images" (default: "search")

    Returns:
        Formatted search results as a string
    """
    results = await searcher.search(query, ctx, max_results, search_type)
    formatted_results = searcher.format_results_for_llm(results)
    return MCPResponse(status="OK", payload=formatted_results)


@mcp.tool()
async def search_news(query: str, ctx: Context, max_results: int = 10, time_filter: str = "qdr:d") -> MCPResponse:
    """
    Search for news using Google Serper API

    Args:
        query: The news search query
        max_results: Maximum number of results to return (default: 10)
        time_filter: Time filter - "qdr:h" (hour), "qdr:d" (day), "qdr:w" (week), "qdr:m" (month), "qdr:y" (year) (default: "qdr:d")

    Returns:
        Formatted news search results as a string
    """
    results = await searcher.search(query, ctx, max_results, "news", time_filter)
    formatted_results = searcher.format_results_for_llm(results)
    return MCPResponse(status="OK", payload=formatted_results)


@mcp.tool()
async def search_places(query: str, ctx: Context, max_results: int = 10) -> MCPResponse:
    """
    Search for places using Google Serper API

    Args:
        query: The places search query (e.g., "Italian restaurants in Upper East Side")
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Formatted places search results as a string
    """
    results = await searcher.search(query, ctx, max_results, "places")
    formatted_results = searcher.format_results_for_llm(results)
    return MCPResponse(status="OK", payload=formatted_results)


@mcp.tool()
async def search_images(query: str, ctx: Context, max_results: int = 10) -> MCPResponse:
    """
    Search for images using Google Serper API

    Args:
        query: The image search query
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Formatted image search results as a string
    """
    results = await searcher.search(query, ctx, max_results, "images")
    formatted_results = searcher.format_results_for_llm(results)
    return MCPResponse(status="OK", payload=formatted_results)


@mcp.tool()
async def wait_before_trying_again(seconds: int, ctx: Context) -> MCPResponse:
    """
    Wait for a specified number of seconds before trying again

    Args:
        seconds: Number of seconds to wait

    Returns:
        Confirmation message
    """
    await log(f"Waiting for {seconds} seconds...", "info", logger, ctx)
    await asyncio.sleep(seconds)
    return MCPResponse(status="OK", payload=f"Waited for {seconds} seconds. You can now try your search again.")


async def main():
    """Main function to start the Google Serper MCP server"""

    await start_mcp_server(mcp, MCP_HOST, MCP_PORT, logger, None)


if __name__ == "__main__":
    asyncio.run(main())
