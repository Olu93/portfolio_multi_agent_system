import logging
import json
import os
import sys
import traceback
from pydantic import BaseModel, Field
from enum import Enum
from mcp.server.fastmcp import FastMCP, Context
from spider_rs import Website
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8004"))

# Initialize FastMCP server
mcp = FastMCP("spider", host=MCP_HOST, port=MCP_PORT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpiderArgs(BaseModel):
    url: str = Field(description="The url to crawl/scrape")
    headers: dict[str, str] | None = Field(description="Additional headers to pass along crawling/scraping requests")
    user_agent: str | None = Field(description="User agent for the crawl/scrape request")
    depth: int = Field(description="Crawl/Scrape depth limit. Use only upon user request", default=0)
    blacklist: list[str] | None = Field(description="Regex that blacklists urls from the crawl/scrape process. Use only upon user request.")
    whitelist: list[str] | None = Field(description="Regex that whitelists urls from the crawl/scrape process. Use only upon user request.")
    respect_robots_txt: bool = Field(description="Whether to respect robots.txt file. Use only upon user request.", default=False)
    accept_invalid_certs: bool = Field(description="Accept invalid certificates - should be used as last resort. Use only upon user request", default=False)

    def build_website(self, is_scrape: bool) -> Website:
        website = Website(self.url)
    
        if self.headers:
            website = website.with_headers(self.headers)
    
        if self.user_agent:
            website = website.with_user_agent(self.user_agent)

        if self.depth:
            website = website.with_depth(self.depth)
    
        if self.blacklist:
            website = website.with_blacklist_url(self.blacklist)
    
        if self.whitelist:
            website = website.with_whitelist_url(self.whitelist)
    
        if self.respect_robots_txt is not None:
            website = website.with_respect_robots_txt(self.respect_robots_txt)
    
        if self.accept_invalid_certs is not None:
            website = website.with_danger_accept_invalid_certs(self.accept_invalid_certs)

        if is_scrape:
            website = website.with_return_page_links(True)

        website = website.build()
        return website


async def crawl_url(args: SpiderArgs, ctx: Context) -> str:
    """Crawl a URL and return a list of discovered URLs"""
    try:
        await ctx.info(f"Starting crawl of: {args.url}")
        website = args.build_website(False)
        website.crawl(headless=True)
        links = website.get_links()
        await ctx.info(f"Crawl completed. Found {len(links)} links")
        return '\n'.join(links)
    except Exception as e:
        await ctx.error(f"Error during crawl: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error during crawl: {str(e)}"


async def scrape_url(args: SpiderArgs, ctx: Context) -> str:
    """Scrape a URL and return pages with their content"""
    try:
        await ctx.info(f"Starting scrape of: {args.url}")
        website = args.build_website(True)
        website.scrape(headless=True)
        
        pages = []
        for page in website.get_pages():
            pages.append(dict(links=list(page.links), url=page.url, html=page.content))
        
        await ctx.info(f"Scrape completed. Found {len(pages)} pages")
        return json.dumps(pages, indent=2)
    except Exception as e:
        await ctx.error(f"Error during scrape: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error during scrape: {str(e)}"


@mcp.tool()
async def crawl(url: str, ctx: Context, headers: dict[str, str] | None = None, 
                user_agent: str | None = None, depth: int = 0, 
                blacklist: list[str] | None = None, whitelist: list[str] | None = None,
                respect_robots_txt: bool = False, accept_invalid_certs: bool = False) -> str:
    """
    Crawls the given url and returns a list of URLs.
    
    Args:
        url: The URL to crawl
        headers: Additional headers to pass along crawling requests
        user_agent: User agent for the crawl request
        depth: Crawl depth limit
        blacklist: Regex patterns that blacklist URLs from the crawl process
        whitelist: Regex patterns that whitelist URLs from the crawl process
        respect_robots_txt: Whether to respect robots.txt file
        accept_invalid_certs: Accept invalid certificates (use as last resort)
        ctx: MCP context for logging
    """
    try:
        args = SpiderArgs(
            url=url,
            headers=headers,
            user_agent=user_agent,
            depth=depth,
            blacklist=blacklist,
            whitelist=whitelist,
            respect_robots_txt=respect_robots_txt,
            accept_invalid_certs=accept_invalid_certs
        )
        return await crawl_url(args, ctx)
    except Exception as e:
        await ctx.error(f"Error creating crawl arguments: {str(e)}")
        return f"Error: {str(e)}"


@mcp.tool()
async def scrape(url: str, ctx: Context, headers: dict[str, str] | None = None,
                user_agent: str | None = None, depth: int = 0,
                blacklist: list[str] | None = None, whitelist: list[str] | None = None,
                respect_robots_txt: bool = False, accept_invalid_certs: bool = False) -> str:
    """
    Scrapes the given url and returns a list of URLs along with their contents in JSON format.
    
    Args:
        url: The URL to scrape
        headers: Additional headers to pass along scraping requests
        user_agent: User agent for the scrape request
        depth: Scrape depth limit
        blacklist: Regex patterns that blacklist URLs from the scrape process
        whitelist: Regex patterns that whitelist URLs from the scrape process
        respect_robots_txt: Whether to respect robots.txt file
        accept_invalid_certs: Accept invalid certificates (use as last resort)
        ctx: MCP context for logging
    """
    try:
        args = SpiderArgs(
            url=url,
            headers=headers,
            user_agent=user_agent,
            depth=depth,
            blacklist=blacklist,
            whitelist=whitelist,
            respect_robots_txt=respect_robots_txt,
            accept_invalid_certs=accept_invalid_certs
        )
        return await scrape_url(args, ctx)
    except Exception as e:
        await ctx.error(f"Error creating scrape arguments: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    print("=== Starting Spider MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== Spider MCP Server shutting down ===")