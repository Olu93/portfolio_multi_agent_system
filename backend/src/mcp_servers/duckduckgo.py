from fastmcp import FastMCP, Context
from mcp_servers.utils.models import MCPResponse
from mcp_servers.utils.helper import log, start_mcp_server
from mcp_servers.utils.constants import MCP_HOST, MCP_PORT
import logging
import httpx
from bs4 import BeautifulSoup
from typing import List
from dataclasses import dataclass
import urllib.parse
import sys
import traceback
import asyncio
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[SearchResult]:
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            await log(f"Searching DuckDuckGo for: {query}", "info", logger, ctx)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                await log("Failed to parse HTML response", "error", logger, ctx)
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            await log(f"Successfully found {len(results)} results", "info", logger, ctx)
            return results

        except httpx.TimeoutException:
            await log("Search request timed out", "error", logger, ctx)
            return []
        except httpx.HTTPError as e:
            await log(f"HTTP error occurred: {str(e)}", "error", logger, ctx, exception=e)
            return []
        except Exception as e:
            await log(f"Unexpected error during search: {str(e)}", "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return []


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await log(f"Fetching content from: {url}", "info", logger, ctx)

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await log(f"Successfully fetched and parsed content ({len(text)} characters)", "info", logger, ctx)
            return text

        except httpx.TimeoutException:
            await log(f"Request timed out for URL: {url}", "error", logger, ctx)
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await log(f"HTTP error occurred while fetching {url}: {str(e)}", "error", logger, ctx, exception=e)
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await log(f"Error fetching content from {url}: {str(e)}", "error", logger, ctx, exception=e)
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# Initialize FastMCP server
mcp = FastMCP("ddg-search", host=MCP_HOST, port=MCP_PORT)
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()

@mcp.tool()
async def wait_before_trying_again(seconds: int, ctx: Context) -> MCPResponse:
    """
    If the search or the fetch fails, wait before trying again. because it's likely that the search or the fetch failed because of a bot protection.

    Args:
        seconds: The number of seconds to wait before trying again. Best strategy is something between 15 and 30 seconds.
        ctx: MCP context for logging
    """
    await asyncio.sleep(seconds)
    return MCPResponse(status="OK", payload="Done")


@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 10) -> MCPResponse:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        formatted_results = searcher.format_results_for_llm(results)
        return MCPResponse(status="OK", payload=formatted_results)
    except Exception as e:
        await log(f"An error occurred while searching: {str(e)}", "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=f"An error occurred while searching: {str(e)}")


# @mcp.tool()
# async def fetch_content(url: str, ctx: Context) -> str:
#     """
#     Fetch and parse content from a webpage URL.

#     Args:
#         url: The webpage URL to fetch content from
#         ctx: MCP context for logging
#     """
#     return await fetcher.fetch_and_parse(url, ctx)




async def main():
    """Main function to start the DuckDuckGo MCP server"""
    
    await start_mcp_server(mcp, MCP_HOST, MCP_PORT, logger, None)


if __name__ == "__main__":
    asyncio.run(main()) 