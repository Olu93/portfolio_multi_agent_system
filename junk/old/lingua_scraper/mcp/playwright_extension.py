from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright, Browser, Page
import asyncio
import json
import sys
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8002"))


@dataclass
class PageInfo:
    """Information about the current page"""
    url: str
    title: str
    html_content: str
    text_content: str
    screenshot_path: Optional[str] = None
    timestamp: str = None


class PlaywrightBrowserManager:
    """Manages connection to a running Playwright browser instance"""
    
    def __init__(self, browser_ws_endpoint: Optional[str] = None):
        self.browser_ws_endpoint = browser_ws_endpoint or os.getenv("PLAYWRIGHT_BROWSER_WS_ENDPOINT") or "ws://localhost:9222"
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
    async def connect_to_browser(self, ctx: Context) -> bool:
        """Connect to a running Playwright browser instance"""
        try:
            if not self.browser_ws_endpoint:
                await ctx.error("No browser WebSocket endpoint provided. Set PLAYWRIGHT_BROWSER_WS_ENDPOINT environment variable or pass it to the constructor.")
                return False
                
            await ctx.info(f"Connecting to Playwright browser at: {self.browser_ws_endpoint}")
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(self.browser_ws_endpoint)
            
            # Get the first available page or create a new one
            pages = self.browser.pages
            if pages:
                self.page = pages[0]
                await ctx.info(f"Connected to existing page: {await self.page.title()}")
            else:
                self.page = await self.browser.new_page()
                await ctx.info("Created new page")
                
            return True
            
        except Exception as e:
            await ctx.error(f"Failed to connect to browser: {str(e)}")
            return False
    
    async def get_current_page_info(self, ctx: Context, include_screenshot: bool = False) -> Optional[PageInfo]:
        """Get information about the current page"""
        try:
            if not self.page:
                await ctx.error("No page available. Please connect to a browser first.")
                return None
                
            await ctx.info("Getting current page information...")
            
            # Get basic page information
            url = self.page.url
            title = await self.page.title()
            
            # Get HTML content
            html_content = await self.page.content()
            
            # Get text content (cleaned)
            text_content = await self.page.evaluate("""
                () => {
                    // Remove script and style elements
                    const elementsToRemove = document.querySelectorAll('script, style, nav, header, footer, .ad, .advertisement');
                    elementsToRemove.forEach(el => el.remove());
                    
                    // Get text content
                    return document.body.innerText || document.body.textContent || '';
                }
            """)
            
            # Clean up text content
            text_content = ' '.join(text_content.split())
            
            # Take screenshot if requested
            screenshot_path = None
            if include_screenshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.png"
                await self.page.screenshot(path=screenshot_path)
                await ctx.info(f"Screenshot saved to: {screenshot_path}")
            
            return PageInfo(
                url=url,
                title=title,
                html_content=html_content,
                text_content=text_content,
                screenshot_path=screenshot_path,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            await ctx.error(f"Failed to get page information: {str(e)}")
            return None
    
    async def navigate_to(self, url: str, ctx: Context) -> bool:
        """Navigate to a specific URL"""
        try:
            if not self.page:
                await ctx.error("No page available. Please connect to a browser first.")
                return False
                
            await ctx.info(f"Navigating to: {url}")
            await self.page.goto(url, wait_until="networkidle")
            await ctx.info(f"Successfully navigated to: {url}")
            return True
            
        except Exception as e:
            await ctx.error(f"Failed to navigate to {url}: {str(e)}")
            return False
    
    async def wait_for_element(self, selector: str, timeout: int = 30000, ctx: Context = None) -> bool:
        """Wait for an element to appear on the page"""
        try:
            if not self.page:
                if ctx:
                    await ctx.error("No page available. Please connect to a browser first.")
                return False
                
            if ctx:
                await ctx.info(f"Waiting for element: {selector}")
            await self.page.wait_for_selector(selector, timeout=timeout)
            
            if ctx:
                await ctx.info(f"Element found: {selector}")
            return True
            
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to wait for element {selector}: {str(e)}")
            return False
    
    async def click_element(self, selector: str, ctx: Context) -> bool:
        """Click on an element"""
        try:
            if not self.page:
                await ctx.error("No page available. Please connect to a browser first.")
                return False
                
            await ctx.info(f"Clicking element: {selector}")
            await self.page.click(selector)
            await ctx.info(f"Successfully clicked: {selector}")
            return True
            
        except Exception as e:
            await ctx.error(f"Failed to click element {selector}: {str(e)}")
            return False
    
    async def fill_form(self, form_data: Dict[str, str], ctx: Context) -> bool:
        """Fill form fields with provided data"""
        try:
            if not self.page:
                await ctx.error("No page available. Please connect to a browser first.")
                return False
                
            await ctx.info(f"Filling form with {len(form_data)} fields")
            
            for selector, value in form_data.items():
                await self.page.fill(selector, value)
                await ctx.info(f"Filled {selector} with: {value}")
            
            await ctx.info("Form filled successfully")
            return True
            
        except Exception as e:
            await ctx.error(f"Failed to fill form: {str(e)}")
            return False
    
    async def close(self):
        """Close the browser connection"""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"Error closing browser: {str(e)}")


# Initialize FastMCP server
mcp = FastMCP("playwright-browser", host=MCP_HOST, port=MCP_PORT)
browser_manager = PlaywrightBrowserManager()


@mcp.tool()
async def connect_browser(browser_ws_endpoint: Optional[str] = None, ctx: Context = None) -> str:
    """
    Connect to a running Playwright browser instance.
    
    Args:
        browser_ws_endpoint: WebSocket endpoint of the running browser (optional if set in env)
        ctx: MCP context for logging
    """
    try:
        if browser_ws_endpoint:
            browser_manager.browser_ws_endpoint = browser_ws_endpoint
            
        success = await browser_manager.connect_to_browser(ctx)
        if success:
            return "Successfully connected to Playwright browser"
        else:
            return "Failed to connect to Playwright browser"
            
    except Exception as e:
        error_msg = f"Error connecting to browser: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def get_current_page(include_screenshot: bool = False, ctx: Context = None) -> str:
    """
    Get information about the current page in the browser.
    
    Args:
        include_screenshot: Whether to take a screenshot of the current page
        ctx: MCP context for logging
    """
    try:
        page_info = await browser_manager.get_current_page_info(ctx, include_screenshot)
        if page_info:
            result = {
                "url": page_info.url,
                "title": page_info.title,
                "text_content": page_info.text_content,
                "html_content": page_info.html_content,
                "timestamp": page_info.timestamp
            }
            
            if page_info.screenshot_path:
                result["screenshot_path"] = page_info.screenshot_path
                
            return json.dumps(result, indent=2)
        else:
            return "Failed to get page information"
            
    except Exception as e:
        error_msg = f"Error getting page information: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def navigate_to_url(url: str, ctx: Context = None) -> str:
    """
    Navigate to a specific URL in the browser.
    
    Args:
        url: The URL to navigate to
        ctx: MCP context for logging
    """
    try:
        success = await browser_manager.navigate_to(url, ctx)
        if success:
            return f"Successfully navigated to: {url}"
        else:
            return f"Failed to navigate to: {url}"
            
    except Exception as e:
        error_msg = f"Error navigating to URL: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def wait_for_element(selector: str, timeout_ms: int = 30000, ctx: Context = None) -> str:
    """
    Wait for an element to appear on the page.
    
    Args:
        selector: CSS selector for the element
        timeout_ms: Timeout in milliseconds
        ctx: MCP context for logging
    """
    try:
        success = await browser_manager.wait_for_element(selector, timeout_ms, ctx)
        if success:
            return f"Element found: {selector}"
        else:
            return f"Element not found within timeout: {selector}"
            
    except Exception as e:
        error_msg = f"Error waiting for element: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def click_element(selector: str, ctx: Context = None) -> str:
    """
    Click on an element on the page.
    
    Args:
        selector: CSS selector for the element to click
        ctx: MCP context for logging
    """
    try:
        success = await browser_manager.click_element(selector, ctx)
        if success:
            return f"Successfully clicked: {selector}"
        else:
            return f"Failed to click: {selector}"
            
    except Exception as e:
        error_msg = f"Error clicking element: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def fill_form(form_data: str, ctx: Context = None) -> str:
    """
    Fill form fields with provided data.
    
    Args:
        form_data: JSON string with selector-value pairs
        ctx: MCP context for logging
    """
    try:
        try:
            form_dict = json.loads(form_data)
        except json.JSONDecodeError:
            return "Invalid JSON format for form_data"
            
        success = await browser_manager.fill_form(form_dict, ctx)
        if success:
            return "Form filled successfully"
        else:
            return "Failed to fill form"
            
    except Exception as e:
        error_msg = f"Error filling form: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def get_page_text(ctx: Context = None) -> str:
    """
    Get the text content of the current page (cleaned).
    
    Args:
        ctx: MCP context for logging
    """
    try:
        page_info = await browser_manager.get_current_page_info(ctx, include_screenshot=False)
        if page_info:
            return page_info.text_content
        else:
            return "Failed to get page text"
            
    except Exception as e:
        error_msg = f"Error getting page text: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def get_page_html(ctx: Context = None) -> str:
    """
    Get the HTML content of the current page.
    
    Args:
        ctx: MCP context for logging
    """
    try:
        page_info = await browser_manager.get_current_page_info(ctx, include_screenshot=False)
        if page_info:
            return page_info.html_content
        else:
            return "Failed to get page HTML"
            
    except Exception as e:
        error_msg = f"Error getting page HTML: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return error_msg


if __name__ == "__main__":
    print("=== Starting Playwright Browser MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    try:
        print("Server initialized and ready to handle connections")
        print("Make sure you have a Playwright browser running with remote debugging enabled")
        print("Example: playwright launch --remote-debugging-port=9222")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== Playwright Browser MCP Server shutting down ===")
        asyncio.run(browser_manager.close())
