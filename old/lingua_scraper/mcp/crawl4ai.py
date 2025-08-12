import logging
import json
import os
import sys
import traceback
import asyncio
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8005"))

# Initialize FastMCP server
mcp = FastMCP("crawl4ai", host=MCP_HOST, port=MCP_PORT)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Crawl4AI components
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
    from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("Crawl4AI not available. Install with: pip install crawl4ai")


class CrawlRequest(BaseModel):
    url: str = Field(description="The URL to crawl/scrape")
    headless: bool = Field(description="Run browser in headless mode", default=True)
    verbose: bool = Field(description="Enable verbose logging", default=False)
    cache_mode: str = Field(description="Cache mode: ENABLED, DISABLED, or BYPASS", default="ENABLED")
    word_count_threshold: int = Field(description="Minimum word count for content", default=1)
    max_pages: int = Field(description="Maximum number of pages to crawl", default=1)
    depth: int = Field(description="Crawl depth limit", default=0)
    user_agent: Optional[str] = Field(description="Custom user agent string")
    proxy: Optional[str] = Field(description="Proxy URL (e.g., http://user:pass@host:port)")
    timeout: int = Field(description="Request timeout in seconds", default=30)
    wait_for: Optional[str] = Field(description="CSS selector to wait for before extracting content")
    js_code: Optional[List[str]] = Field(description="JavaScript code to execute before extraction")
    screenshot: bool = Field(description="Take screenshot during crawl", default=False)
    capture_network: bool = Field(description="Capture network traffic", default=False)
    capture_console: bool = Field(description="Capture console logs", default=False)
    mhtml: bool = Field(description="Generate MHTML snapshot", default=False)
    locale: Optional[str] = Field(description="Browser locale (e.g., en-US)")
    timezone_id: Optional[str] = Field(description="Timezone ID (e.g., America/Los_Angeles)")
    geolocation_lat: Optional[float] = Field(description="Geolocation latitude")
    geolocation_lng: Optional[float] = Field(description="Geolocation longitude")
    geolocation_accuracy: Optional[float] = Field(description="Geolocation accuracy")


class ExtractionRequest(BaseModel):
    url: str = Field(description="The URL to extract data from")
    extraction_type: str = Field(description="Type of extraction: 'schema', 'css', or 'llm'")
    schema: Optional[Dict[str, Any]] = Field(description="JSON schema for extraction")
    css_schema: Optional[Dict[str, Any]] = Field(description="CSS-based extraction schema")
    llm_provider: Optional[str] = Field(description="LLM provider (e.g., openai/gpt-4o)")
    llm_api_key: Optional[str] = Field(description="LLM API key")
    instruction: Optional[str] = Field(description="Instruction for LLM extraction")
    headless: bool = Field(description="Run browser in headless mode", default=True)
    verbose: bool = Field(description="Enable verbose logging", default=False)


class DeepCrawlRequest(BaseModel):
    url: str = Field(description="The starting URL for deep crawling")
    strategy: str = Field(description="Crawl strategy: bfs, dfs, or bestfirst", default="bfs")
    max_pages: int = Field(description="Maximum number of pages to crawl", default=10)
    max_depth: int = Field(description="Maximum crawl depth", default=3)
    include_external: bool = Field(description="Include external links", default=False)
    filters: Optional[List[str]] = Field(description="URL filters (regex patterns)")
    headless: bool = Field(description="Run browser in headless mode", default=True)
    verbose: bool = Field(description="Enable verbose logging", default=False)


def get_cache_mode(mode: str) -> CacheMode:
    """Convert string to CacheMode enum"""
    mode_map = {
        "ENABLED": CacheMode.ENABLED,
        "DISABLED": CacheMode.DISABLED,
        "BYPASS": CacheMode.BYPASS
    }
    return mode_map.get(mode.upper(), CacheMode.ENABLED)


async def create_browser_config(request: CrawlRequest) -> BrowserConfig:
    """Create browser configuration from request"""
    config = BrowserConfig(
        headless=request.headless,
        verbose=request.verbose,
        timeout=request.timeout * 1000  # Convert to milliseconds
    )
    
    if request.user_agent:
        config.user_agent = request.user_agent
    
    if request.proxy:
        config.proxy = request.proxy
    
    return config


async def create_run_config(request: CrawlRequest) -> CrawlerRunConfig:
    """Create run configuration from request"""
    config = CrawlerRunConfig(
        cache_mode=get_cache_mode(request.cache_mode),
        word_count_threshold=request.word_count_threshold,
        max_pages=request.max_pages,
        depth=request.depth
    )
    
    if request.wait_for:
        config.wait_for = request.wait_for
    
    if request.js_code:
        config.js_code = request.js_code
    
    if request.screenshot:
        config.screenshot = True
    
    if request.capture_network:
        config.capture_network = True
    
    if request.capture_console:
        config.capture_console = True
    
    if request.mhtml:
        config.mhtml = True
    
    if request.locale:
        config.locale = request.locale
    
    if request.timezone_id:
        config.timezone_id = request.timezone_id
    
    if request.geolocation_lat and request.geolocation_lng:
        from crawl4ai import GeolocationConfig
        config.geolocation = GeolocationConfig(
            latitude=request.geolocation_lat,
            longitude=request.geolocation_lng,
            accuracy=request.geolocation_accuracy or 10.0
        )
    
    return config


@mcp.tool()
async def crawl_webpage(url: str, ctx: Context, headless: bool = True, verbose: bool = False,
                       cache_mode: str = "ENABLED", word_count_threshold: int = 1,
                       max_pages: int = 1, depth: int = 0, user_agent: Optional[str] = None,
                       proxy: Optional[str] = None, timeout: int = 30, wait_for: Optional[str] = None,
                       js_code: Optional[List[str]] = None, screenshot: bool = False,
                       capture_network: bool = False, capture_console: bool = False,
                       mhtml: bool = False, locale: Optional[str] = None,
                       timezone_id: Optional[str] = None, geolocation_lat: Optional[float] = None,
                       geolocation_lng: Optional[float] = None, geolocation_accuracy: Optional[float] = None) -> str:
    """
    Crawl a webpage and return the content in markdown format.
    
    Args:
        url: The URL to crawl
        headless: Run browser in headless mode
        verbose: Enable verbose logging
        cache_mode: Cache mode (ENABLED, DISABLED, BYPASS)
        word_count_threshold: Minimum word count for content
        max_pages: Maximum number of pages to crawl
        depth: Crawl depth limit
        user_agent: Custom user agent string
        proxy: Proxy URL
        timeout: Request timeout in seconds
        wait_for: CSS selector to wait for before extracting content
        js_code: JavaScript code to execute before extraction
        screenshot: Take screenshot during crawl
        capture_network: Capture network traffic
        capture_console: Capture console logs
        mhtml: Generate MHTML snapshot
        locale: Browser locale (e.g., en-US)
        timezone_id: Timezone ID (e.g., America/Los_Angeles)
        geolocation_lat: Geolocation latitude
        geolocation_lng: Geolocation longitude
        geolocation_accuracy: Geolocation accuracy
        ctx: MCP context for logging
    """
    if not CRAWL4AI_AVAILABLE:
        return "Error: Crawl4AI is not available. Please install with: pip install crawl4ai"
    
    try:
        await ctx.info(f"Starting crawl of: {url}")
        
        request = CrawlRequest(
            url=url,
            headless=headless,
            verbose=verbose,
            cache_mode=cache_mode,
            word_count_threshold=word_count_threshold,
            max_pages=max_pages,
            depth=depth,
            user_agent=user_agent,
            proxy=proxy,
            timeout=timeout,
            wait_for=wait_for,
            js_code=js_code,
            screenshot=screenshot,
            capture_network=capture_network,
            capture_console=capture_console,
            mhtml=mhtml,
            locale=locale,
            timezone_id=timezone_id,
            geolocation_lat=geolocation_lat,
            geolocation_lng=geolocation_lng,
            geolocation_accuracy=geolocation_accuracy
        )
        
        browser_config = await create_browser_config(request)
        run_config = await create_run_config(request)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=request.url, config=run_config)
            
            if not result.success:
                await ctx.error(f"Crawl failed: {result.error}")
                return f"Error: {result.error}"
            
            await ctx.info(f"Crawl completed successfully. Content length: {len(result.markdown)}")
            
            # Prepare response
            response = {
                "success": True,
                "url": result.url,
                "markdown": result.markdown,
                "raw_markdown": result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else result.markdown,
                "fit_markdown": result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else None,
                "links": result.links if hasattr(result, 'links') else [],
                "metadata": result.metadata if hasattr(result, 'metadata') else {},
                "screenshot": result.screenshot if hasattr(result, 'screenshot') else None,
                "network_logs": result.network_logs if hasattr(result, 'network_logs') else None,
                "console_logs": result.console_logs if hasattr(result, 'console_logs') else None
            }
            
            return json.dumps(response, indent=2, default=str)
            
    except Exception as e:
        await ctx.error(f"Error during crawl: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error: {str(e)}"


@mcp.tool()
async def extract_structured_data(url: str, extraction_type: str, ctx: Context,
                                schema: Optional[Dict[str, Any]] = None,
                                css_schema: Optional[Dict[str, Any]] = None,
                                llm_provider: Optional[str] = None,
                                llm_api_key: Optional[str] = None,
                                instruction: Optional[str] = None,
                                headless: bool = True, verbose: bool = False) -> str:
    """
    Extract structured data from a webpage using various strategies.
    
    Args:
        url: The URL to extract data from
        extraction_type: Type of extraction ('schema', 'css', or 'llm')
        schema: JSON schema for extraction (for schema/llm types)
        css_schema: CSS-based extraction schema (for css type)
        llm_provider: LLM provider (e.g., openai/gpt-4o)
        llm_api_key: LLM API key
        instruction: Instruction for LLM extraction
        headless: Run browser in headless mode
        verbose: Enable verbose logging
        ctx: MCP context for logging
    """
    if not CRAWL4AI_AVAILABLE:
        return "Error: Crawl4AI is not available. Please install with: pip install crawl4ai"
    
    try:
        await ctx.info(f"Starting structured data extraction from: {url}")
        
        browser_config = BrowserConfig(headless=headless, verbose=verbose)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        
        # Set up extraction strategy
        if extraction_type == "css" and css_schema:
            extraction_strategy = JsonCssExtractionStrategy(css_schema, verbose=verbose)
            run_config.extraction_strategy = extraction_strategy
        elif extraction_type == "llm" and schema and llm_provider and llm_api_key:
            llm_config = LLMConfig(provider=llm_provider, api_token=llm_api_key)
            extraction_strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=schema,
                extraction_type="schema",
                instruction=instruction or "Extract the requested data from the webpage content."
            )
            run_config.extraction_strategy = extraction_strategy
        elif extraction_type == "schema" and schema:
            # For schema-based extraction without LLM
            extraction_strategy = JsonCssExtractionStrategy(schema, verbose=verbose)
            run_config.extraction_strategy = extraction_strategy
        else:
            return "Error: Invalid extraction configuration. Please provide appropriate schema and parameters."
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if not result.success:
                await ctx.error(f"Extraction failed: {result.error}")
                return f"Error: {result.error}"
            
            await ctx.info(f"Extraction completed successfully")
            
            response = {
                "success": True,
                "url": result.url,
                "extracted_content": result.extracted_content,
                "markdown": result.markdown,
                "metadata": result.metadata if hasattr(result, 'metadata') else {}
            }
            
            return json.dumps(response, indent=2, default=str)
            
    except Exception as e:
        await ctx.error(f"Error during extraction: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error: {str(e)}"


@mcp.tool()
async def deep_crawl(url: str, strategy: str = "bfs", max_pages: int = 10, max_depth: int = 3,
                    include_external: bool = False, filters: Optional[List[str]] = None,
                    headless: bool = True, verbose: bool = False, ctx: Context = None) -> str:
    """
    Perform deep crawling of a website using various strategies.
    
    Args:
        url: The starting URL for deep crawling
        strategy: Crawl strategy (bfs, dfs, or bestfirst)
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth
        include_external: Include external links
        filters: URL filters (regex patterns)
        headless: Run browser in headless mode
        verbose: Enable verbose logging
        ctx: MCP context for logging
    """
    if not CRAWL4AI_AVAILABLE:
        return "Error: Crawl4AI is not available. Please install with: pip install crawl4ai"
    
    try:
        await ctx.info(f"Starting deep crawl of: {url} with strategy: {strategy}")
        
        browser_config = BrowserConfig(headless=headless, verbose=verbose)
        
        # Configure deep crawling
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED,
            max_pages=max_pages,
            depth=max_depth,
            include_external=include_external
        )
        
        if filters:
            run_config.filters = filters
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Use the appropriate deep crawl method based on strategy
            if strategy == "bfs":
                results = await crawler.adeep_crawl_bfs(url=url, config=run_config)
            elif strategy == "dfs":
                results = await crawler.adeep_crawl_dfs(url=url, config=run_config)
            elif strategy == "bestfirst":
                results = await crawler.adeep_crawl_bestfirst(url=url, config=run_config)
            else:
                return f"Error: Unknown strategy '{strategy}'. Use 'bfs', 'dfs', or 'bestfirst'"
            
            await ctx.info(f"Deep crawl completed. Found {len(results)} pages")
            
            # Process results
            processed_results = []
            for result in results:
                if result.success:
                    processed_results.append({
                        "url": result.url,
                        "markdown": result.markdown,
                        "links": result.links if hasattr(result, 'links') else [],
                        "metadata": result.metadata if hasattr(result, 'metadata') else {}
                    })
            
            response = {
                "success": True,
                "strategy": strategy,
                "total_pages": len(results),
                "successful_pages": len(processed_results),
                "pages": processed_results
            }
            
            return json.dumps(response, indent=2, default=str)
            
    except Exception as e:
        await ctx.error(f"Error during deep crawl: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error: {str(e)}"


@mcp.tool()
async def extract_tables(url: str, ctx: Context, table_score_threshold: int = 8,
                        headless: bool = True, verbose: bool = False) -> str:
    """
    Extract tables from a webpage and return them as structured data.
    
    Args:
        url: The URL to extract tables from
        table_score_threshold: Threshold for table detection (1-10)
        headless: Run browser in headless mode
        verbose: Enable verbose logging
        ctx: MCP context for logging
    """
    if not CRAWL4AI_AVAILABLE:
        return "Error: Crawl4AI is not available. Please install with: pip install crawl4ai"
    
    try:
        await ctx.info(f"Starting table extraction from: {url}")
        
        browser_config = BrowserConfig(headless=headless, verbose=verbose)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_score_threshold=table_score_threshold
        )
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if not result.success:
                await ctx.error(f"Table extraction failed: {result.error}")
                return f"Error: {result.error}"
            
            tables = result.media.get("tables", []) if hasattr(result, 'media') else []
            await ctx.info(f"Table extraction completed. Found {len(tables)} tables")
            
            response = {
                "success": True,
                "url": result.url,
                "tables_count": len(tables),
                "tables": tables,
                "markdown": result.markdown
            }
            
            return json.dumps(response, indent=2, default=str)
            
    except Exception as e:
        await ctx.error(f"Error during table extraction: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        return f"Error: {str(e)}"


@mcp.tool()
async def check_crawl4ai_status(ctx: Context) -> str:
    """
    Check if Crawl4AI is properly installed and available.
    
    Args:
        ctx: MCP context for logging
    """
    if CRAWL4AI_AVAILABLE:
        try:
            # Try to import and create a basic crawler to verify installation
            from crawl4ai import AsyncWebCrawler, BrowserConfig
            config = BrowserConfig(headless=True)
            await ctx.info("Crawl4AI is available and working correctly")
            return "Crawl4AI is properly installed and ready to use."
        except Exception as e:
            await ctx.error(f"Crawl4AI import error: {str(e)}")
            return f"Crawl4AI is installed but has configuration issues: {str(e)}"
    else:
        return "Crawl4AI is not available. Please install with: pip install crawl4ai"


if __name__ == "__main__":
    print("=== Starting Crawl4AI MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    print(f"Crawl4AI available: {CRAWL4AI_AVAILABLE}")
    
    if not CRAWL4AI_AVAILABLE:
        print("WARNING: Crawl4AI is not installed. Install with: pip install crawl4ai")
    
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== Crawl4AI MCP Server shutting down ===")
