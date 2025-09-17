import os, json, asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Dict
from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright, Browser, Page
from lxml import html
from lxml_html_clean import Cleaner
import logging
import base64

from pydantic import BaseModel

# --- logging setup ---
log = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8002"))
PW_CHANNEL = os.getenv("PLAYWRIGHT_CHANNEL")  # e.g., "chrome"
PW_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() in {"1","true","yes"}

# Fast reusable cleaner
_LXML_CLEANER = Cleaner(
    scripts=True, javascript=True, style=True, comments=True,
    annoying_tags=True, processing_instructions=True, inline_style=True,
    links=False, forms=False, meta=True, page_structure=False,
)

# CSS buckets for obvious noise (requires cssselect)
_NOISE_CSS = (
    ".ad, .ads, .advertisement, .advertisement-container, "
    ".cookie-banner, .cookie-consent, .popup, .modal, "
    ".sidebar, .social-media, .share-buttons, "
    ".comments, .comment-section, .related-articles, "
    ".newsletter, .subscribe, .promo, .banner"
)

# Attr patterns (case-insensitive); keep class token-aware to avoid matching 'header'
_XPATH_ATTR_NOISE = (
    "//*[@data-ad or @data-advertisement or "
    "contains(translate(@id,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'ad') or "
    "contains(concat(' ', translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), ' '), ' ad ')]"
)

def _prune_noise_nodes(doc) -> None:
    # Structural buckets
    for el in doc.xpath("//nav|//header|//footer|//meta|//link|//title"):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
    # CSS-based buckets
    for el in doc.cssselect(_NOISE_CSS):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
    # Attribute-based buckets
    for el in doc.xpath(_XPATH_ATTR_NOISE):
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)

def clean_html_lxml(html_content: str, reduce_noise: bool = True) -> str:
    """Return cleaned HTML (scripts/styles/comments/etc removed; optional extra noise pruning)."""
    try:
        doc = html.fromstring(html_content)
        doc = _LXML_CLEANER.clean_html(doc)
        if reduce_noise:
            _prune_noise_nodes(doc)
        return html.tostring(doc, encoding="unicode", method="html")
    except Exception:
        # Best-effort fallback
        try:
            return html.tostring(html.fromstring(html_content), encoding="unicode", method="html")
        except Exception:
            return html_content

def extract_text_lxml(html_content: str, reduce_noise: bool = True) -> str:
    """Return normalized visible text from HTML (space-separated, trimmed)."""
    try:
        doc = html.fromstring(html_content)
        doc = _LXML_CLEANER.clean_html(doc)
        if reduce_noise:
            _prune_noise_nodes(doc)
        return " ".join(t.strip() for t in doc.itertext() if t and t.strip())
    except Exception:
        # Minimal fallback: strip tags naïvely via reparse
        try:
            doc = html.fromstring(html_content)
            return " ".join(t.strip() for t in doc.itertext() if t and t.strip())
        except Exception:
            return ""




@dataclass
class PageInfo:
    url: str
    title: str
    html_content: str
    text_content: str
    screenshot: Optional[str] = None
    timestamp: str = None

class MCPResponse(BaseModel):
    status: Literal["OK", "ERR"]
    payload: Optional[str|PageInfo] = None
    error: Optional[str] = None


class PlaywrightBrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    # replace your launch_browser with this more defensive version
    async def launch_browser(self, ctx: Optional[Context], headless: Optional[bool] = None) -> bool:
        headless = PW_HEADLESS if headless is None else headless
        try:
            # Reuse only if truly alive
            if self.browser and self.browser.is_connected():
                if not self.page or self.page.is_closed():
                    self.page = await self.browser.new_page()
                    self.page.set_default_timeout(30_000)
                    self.page.set_default_navigation_timeout(45_000)
                msg = "Browser already running; reusing existing instance."
                log.info(msg)
                if ctx: await ctx.info(msg)
                return True

            # stale handles → full reset
            if ctx: await ctx.info("Starting Playwright…")
            log.info("Starting Playwright…")
            await self.close()  # ensures clean state if anything half-open
            self.playwright = await async_playwright().start()

            launch_kwargs = {"headless": headless, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
            if PW_CHANNEL:
                launch_kwargs["channel"] = PW_CHANNEL

            self.browser = await self.playwright.chromium.launch(**launch_kwargs)
            self._wired_events = False
            self._wire_browser_events(ctx)

            self.page = await self.browser.new_page()
            self.page.set_default_timeout(30_000)
            self.page.set_default_navigation_timeout(45_000)

            msg = f"Browser launched (headless={headless}). New page created."
            log.info(msg)
            if ctx: await ctx.info(msg)
            return True

        except Exception as e:
            log.exception("Failed to launch browser")
            if ctx: await ctx.error(f"Failed to launch browser: {e}")
            await self.close()
            return False


    # hardened navigate_to
    async def navigate_to(self, url: str, ctx: Optional[Context]) -> bool:
        try:
            if not self.browser or not self.browser.is_connected():
                msg = "Browser is not running. Launch the browser first."
                log.warning(msg)
                if ctx: await ctx.error(msg)
                return False

            if not self.page or self.page.is_closed():
                log.info("Page missing/closed; creating a new page.")
                if ctx: await ctx.info("Page was closed; creating a new page.")
                self.page = await self.browser.new_page()
                self.page.set_default_timeout(30_000)
                self.page.set_default_navigation_timeout(45_000)

            if ctx: await ctx.info(f"Navigating: {url}")
            log.info(f"Navigating to {url}")
            await self.page.goto(url, wait_until="networkidle")
            if ctx: await ctx.info(f"Navigation finished: {url}")
            log.info(f"Navigation finished: {url}")
            return True

        except Exception as e:
            log.exception("Goto failed")
            if ctx: await ctx.error(f"Goto failed: {e}")
            return False


    async def wait_for_element(self, selector: str, timeout_ms: int, ctx: Optional[Context]) -> bool:
        try:
            if not self.page:
                msg = "No page. Launch the browser first."
                log.warning(msg)
                if ctx: await ctx.error(msg)
                return False
            log.debug(f"Waiting for selector {selector} (timeout={timeout_ms}ms)")
            if ctx: await ctx.info(f"Waiting for element: {selector}")
            await self.page.wait_for_selector(selector, timeout=timeout_ms)
            msg = f"Element found: {selector}"
            log.info(msg)
            if ctx: await ctx.info(msg)
            return True
        except Exception as e:
            log.exception("wait_for_selector failed")
            if ctx: await ctx.error(f"Failed waiting for {selector}: {e}")
            return False

    async def click_element(self, selector: str, ctx: Optional[Context]) -> bool:
        try:
            if not self.page:
                msg = "No page. Launch the browser first."
                log.warning(msg)
                if ctx: await ctx.error(msg)
                return False
            log.debug(f"Clicking {selector}")
            if ctx: await ctx.info(f"Clicking element: {selector}")
            await self.page.click(selector)
            msg = f"Clicked: {selector}"
            log.info(msg)
            if ctx: await ctx.info(msg)
            return True
        except Exception as e:
            log.exception("Click failed")
            if ctx: await ctx.error(f"Click failed ({selector}): {e}")
            return False

    async def fill_form(self, form_data: Dict[str, str], ctx: Optional[Context]) -> bool:
        try:
            if not self.page:
                msg = "No page. Launch the browser first."
                log.warning(msg)
                if ctx: await ctx.error(msg)
                return False
            log.debug(f"Filling form with {len(form_data)} fields")
            if ctx: await ctx.info(f"Filling form with {len(form_data)} fields")
            for sel, val in form_data.items():
                await self.page.fill(sel, val)
                log.debug(f"Filled {sel}")
                if ctx: await ctx.info(f"Filled {sel}")
            log.info("Form filled successfully")
            if ctx: await ctx.info("Form filled successfully")
            return True
        except Exception as e:
            log.exception("Fill form failed")
            if ctx: await ctx.error(f"Failed to fill form: {e}")
            return False

    async def _ensure_page_available(self, ctx: Optional[Context]) -> tuple[bool, Optional[str]]:
        """Ensure page is available, return (success, error_message)."""
        if not self.page:
            msg = "No page. Launch the browser first."
            log.warning(msg)
            if ctx: await ctx.error(msg)
            return False, msg
        return True, None

    async def get_page_url(self, ctx: Optional[Context]) -> MCPResponse:
        """Get the current page URL."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            url = self.page.url
            log.debug(f"Retrieved page URL: {url}")
            return MCPResponse(status="OK", payload=url)
        except Exception as e:
            error_msg = f"Get page URL failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_page_title(self, ctx: Optional[Context]) -> MCPResponse:
        """Get the current page title."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            title = await self.page.title()
            log.debug(f"Retrieved page title: {title}")
            return MCPResponse(status="OK", payload=title)
        except Exception as e:
            error_msg = f"Get page title failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_page_html_raw(self, ctx: Optional[Context]) -> MCPResponse:
        """Get raw HTML content of the current page."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            html_content = await self.page.content()
            log.debug("Retrieved raw HTML content")
            return MCPResponse(status="OK", payload=html_content)
        except Exception as e:
            error_msg = f"Get page HTML failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_page_text(self, reduce_noise: bool, ctx: Optional[Context]) -> MCPResponse:
        """Get text content of the current page."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            html_content = await self.page.content()
            text_content = extract_text_lxml(html_content, reduce_noise)
            log.debug("Retrieved page text content")
            return MCPResponse(status="OK", payload=text_content)
        except Exception as e:
            error_msg = f"Get page text failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_page_html_cleaned(self, reduce_noise: bool, ctx: Optional[Context]) -> MCPResponse:
        """Get cleaned HTML content of the current page."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            html_content = await self.page.content()
            if reduce_noise:
                html_content = clean_html_lxml(html_content, True)
            log.debug("Retrieved cleaned HTML content")
            return MCPResponse(status="OK", payload=html_content)
        except Exception as e:
            error_msg = f"Get cleaned page HTML failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_page_screenshot(self, ctx: Optional[Context]) -> MCPResponse:
        """Get a screenshot of the current page as base64 string."""
        try:
            available, error_msg = await self._ensure_page_available(ctx)
            if not available:
                return MCPResponse(status="ERR", error=error_msg)
            img_bytes = await self.page.screenshot(full_page=True)
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            log.debug("Captured page screenshot")
            return MCPResponse(status="OK", payload=base64_img)
        except Exception as e:
            error_msg = f"Get page screenshot failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def get_current_page_info(self, include_screenshot: bool, reduce_noise: bool, ctx: Optional[Context]) -> MCPResponse:
        """Get comprehensive page information by calling individual methods."""
        try:
            if ctx: await ctx.info("Collecting page info…")
            
            # Get individual components efficiently
            url_resp = await self.get_page_url(ctx)
            title_resp = await self.get_page_title(ctx)
            html_resp = await self.get_page_html_raw(ctx)
            text_resp = await self.get_page_text(reduce_noise, ctx)
            
            # Check if any individual call failed
            for resp, name in [(url_resp, "URL"), (title_resp, "title"), (html_resp, "HTML"), (text_resp, "text")]:
                if resp.status == "ERR":
                    return MCPResponse(status="ERR", error=f"Failed to get page {name}: {resp.error}")
            
            base64_img = None
            if include_screenshot:
                screenshot_resp = await self.get_page_screenshot(ctx)
                if screenshot_resp.status == "ERR":
                    return MCPResponse(status="ERR", error=f"Failed to get screenshot: {screenshot_resp.error}")
                base64_img = screenshot_resp.payload

            info = PageInfo(
                url=url_resp.payload, title=title_resp.payload, html_content=html_resp.payload, 
                text_content=text_resp.payload, screenshot=base64_img, timestamp=datetime.now().isoformat()
            )
            log.debug(f"Page info collected: title={title_resp.payload}, url={url_resp.payload}")
            if ctx: await ctx.info("Page info collected.")
            return MCPResponse(status="OK", payload=info)
        except Exception as e:
            error_msg = f"Get page info failed: {e}"
            log.exception(error_msg)
            if ctx: await ctx.error(error_msg)
            return MCPResponse(status="ERR", error=str(e))

    async def close(self):
        try:
            log.info("Shutting down Playwright…")
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception:
            log.exception("Error during close()")
        finally:
            self.browser = None
            self.page = None
            self.playwright = None

    def _wire_browser_events(self, ctx: Optional[Context]):
        if self._wired_events or not self.browser:
            return
        def _on_disc(*_):
            log.warning("Browser disconnected; clearing handles.")
            # ctx is optional here; don't await in event
            try:
                # best-effort notify (non-blocking)
                if ctx: asyncio.create_task(ctx.info("Browser disconnected; please relaunch."))
            except Exception:
                pass
            self.page = None
            self.browser = None
        self.browser.on("disconnected", _on_disc)
        self._wired_events = True       

    # make wait/click/form robust too (recreate page if closed)
    def _ensure_page_ok(self) -> bool:
        return self.browser and self.browser.is_connected() and self.page and not self.page.is_closed()


mcp = FastMCP("playwright-browser", host=MCP_HOST, port=MCP_PORT)
manager = PlaywrightBrowserManager()


@mcp.tool()
async def launch_browser_with_page(url: str, headless: bool = False, ctx: Context = None) -> MCPResponse:
    """
    Launch a new browser instance using Playwright and navigate to a specific URL.
    
    This tool starts a browser (Chrome/Chromium by default) that can be used for web automation.
    The browser will be reused if already running and connected.
    
    Args:
        url (str): The URL to navigate to. Must include protocol (http:// or https://).
        headless (bool): Whether to run the browser in headless mode (no GUI).
                        Defaults to False (visible browser).
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        launch_browser_with_page("https://example.com")
        launch_browser_with_page("https://google.com")
    
    Note:
        Requires browser to be launched first using launch_browser().
        This tool is more efficient on first launch than launch_browser() followed by navigate_to_url()
        as it launches the browser and navigates to the URL in one step.
        
    """
    try:
        success = await manager.launch_browser(ctx, headless=headless)
        if success:
            success = await manager.navigate_to(url, ctx)
            if success:
                return MCPResponse(status="OK")
            else:
                return MCPResponse(status="ERR", error="Failed to navigate to URL")
        else:
            return MCPResponse(status="ERR", error="Failed to launch browser")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def launch_browser(headless: bool = False, ctx: Context = None) -> MCPResponse:
    """
    Launch a new browser instance using Playwright.
    
    This tool starts a browser (Chrome/Chromium by default) that can be used for web automation.
    The browser will be reused if already running and connected.
    
    Args:
        headless (bool): Whether to run the browser in headless mode (no GUI).
                        Defaults to False (visible browser).
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        launch_browser(headless=True)  # Start headless browser
        launch_browser()               # Start visible browser
    """
    try:
        success = await manager.launch_browser(ctx, headless=headless)
        if success:
            return MCPResponse(status="OK")
        else:
            return MCPResponse(status="ERR", error="Failed to launch browser")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def navigate_to_url(url: str, ctx: Context = None) -> MCPResponse:
    """
    Navigate the browser to a specific URL.
    
    This tool loads a web page in the current browser tab. The browser must be launched first.
    The tool waits for the page to fully load (network idle) before completing.
    
    Args:
        url (str): The URL to navigate to. Must include protocol (http:// or https://).
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        navigate_to_url("https://example.com")
        navigate_to_url("https://google.com")
    
    Note:
        Requires browser to be launched first using launch_browser().
    """
    try:
        success = await manager.navigate_to(url, ctx)
        if success:
            return MCPResponse(status="OK")
        else:
            return MCPResponse(status="ERR", error=f"Failed to navigate to {url}")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def wait_for_element(selector: str, timeout_ms: int = 30000, ctx: Context = None) -> MCPResponse:
    """
    Wait for an element to appear on the current page.
    
    This tool waits for a specific element to be present in the DOM before proceeding.
    Useful for ensuring elements are loaded before interacting with them.
    
    Args:
        selector (str): CSS selector for the element to wait for.
                       Examples: "#id", ".class", "button", "[data-testid='submit']"
        timeout_ms (int): Maximum time to wait in milliseconds. Defaults to 30000 (30 seconds).
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        wait_for_element("#login-button")
        wait_for_element(".loading-spinner", timeout_ms=5000)
        wait_for_element("[data-testid='submit-form']")
    
    Note:
        Uses CSS selectors. Common patterns:
        - ID: "#myId"
        - Class: ".myClass"  
        - Attribute: "[data-testid='value']"
        - Element: "button", "input", "div"
    """
    try:
        success = await manager.wait_for_element(selector, timeout_ms, ctx)
        if success:
            return MCPResponse(status="OK")
        else:
            return MCPResponse(status="ERR", error=f"Element not found: {selector}")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def click_element(selector: str, ctx: Context = None) -> MCPResponse:
    """
    Click on an element on the current page.
    
    This tool simulates a mouse click on a specific element. The element must be visible and clickable.
    
    Args:
        selector (str): CSS selector for the element to click.
                       Examples: "#submit", ".button", "button[type='submit']"
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        click_element("#login-button")
        click_element(".submit-form")
        click_element("button[data-action='save']")
    
    Note:
        Element must be visible and not covered by other elements.
        Use wait_for_element() first if element might not be immediately available.
    """
    try:
        success = await manager.click_element(selector, ctx)
        if success:
            return MCPResponse(status="OK")
        else:
            return MCPResponse(status="ERR", error=f"Failed to click element: {selector}")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def fill_form(form_data: str, ctx: Context = None) -> MCPResponse:
    """
    Fill form fields on the current page.
    
    This tool fills multiple form fields at once using CSS selectors as keys.
    All fields are filled in sequence.
    
    Args:
        form_data (str): JSON string containing field selectors and values.
                        Format: '{"#username": "john", "#password": "secret", ".email": "john@example.com"}'
    
    Returns:
        MCPResponse: Response object with status and error fields.
    
    Example:
        fill_form('{"#username": "john_doe", "#password": "mypassword123"}')
        fill_form('{"input[name=\"email\"]": "user@example.com", "select[name=\"country\"]": "US"}')
        fill_form('{".form-control": "some value", "[data-testid=\"submit\"]": ""}')
    
    Note:
        - form_data must be valid JSON
        - Keys are CSS selectors for form elements
        - Values are the text to enter
        - Use empty string "" for checkboxes/buttons if needed
        - All specified fields will be filled in the order provided
    """
    try:
        data = json.loads(form_data)
        success = await manager.fill_form(data, ctx)
        if success:
            return MCPResponse(status="OK")
        else:
            return MCPResponse(status="ERR", error="Failed to fill form fields")
    except json.JSONDecodeError as e:
        return MCPResponse(status="ERR", error=f"Invalid JSON format: {e}")
    except Exception as e:
        return MCPResponse(status="ERR", error=str(e))

@mcp.tool()
async def get_current_page(include_screenshot: bool = False, ctx: Context = None) -> MCPResponse:
    """
    Get comprehensive information about the current page.
    
    This tool returns detailed information about the current browser page including URL, title,
    HTML content, and optionally a screenshot. Text content is automatically cleaned of noise.
    
    Args:
        include_screenshot (bool): Whether to capture and save a screenshot of the current page.
                                 Screenshots are saved as PNG files with timestamp names.
                                 Defaults to False.
    
    Returns:
        MCPResponse: Response object with status, payload (JSON string), and error fields.
                    payload contains page information with keys:
                    - url: Current page URL
                    - title: Page title
                    - html_content: Full HTML source
                    - text_content: Cleaned text content (noise removed)
                    - screenshot: Base64 encoded screenshot (if include_screenshot=True)
                    - timestamp: When the data was collected
    
    Example:
        get_current_page()                    # Get basic page info
        get_current_page(include_screenshot=True)  # Get page info + screenshot
    
    Note:
        Text content is automatically cleaned using lxml to remove scripts,
        styles, ads, navigation elements, and other noise for better readability.
    """
    return await manager.get_current_page_info(include_screenshot, True, ctx)


@mcp.tool()
async def get_page_screenshot(ctx: Context = None) -> MCPResponse:
    """
    Get a screenshot of the current page.
    
    This tool captures a full-page screenshot of the current browser page and returns it
    as a base64 encoded string. This is efficient as it only captures the screenshot
    without extracting other page information.
    
    Returns:
        MCPResponse: Response object with status, payload (base64 image data), and error fields.
    
    Example:
        get_page_screenshot()  # Get screenshot as base64 string
    
    Note:
        This tool only captures the screenshot and doesn't extract text or HTML content,
        making it much faster than get_current_page() with include_screenshot=True.
    """
    return await manager.get_page_screenshot(ctx)

@mcp.tool()
async def get_page_text(reduce_noise: bool = True, ctx: Context = None) -> MCPResponse:
    """
    Extract clean text content from the current page.
    
    This tool extracts only the text content from the current page, with optional noise reduction.
    Uses lxml for robust text extraction and cleaning. This is efficient as it only extracts
    text content without getting HTML, screenshots, or other page information.
    
    Args:
        reduce_noise (bool): Whether to remove noise elements like scripts, styles, ads,
                           navigation, headers, footers, etc. Defaults to True.
    
    Returns:
        MCPResponse: Response object with status, payload (text content), and error fields.
    
    Example:
        get_page_text()                # Get clean text (noise removed)
        get_page_text(reduce_noise=False)  # Get raw text (no cleaning)
    
    Note:
        When reduce_noise=True, removes:
        - Script and style elements
        - Navigation, header, footer elements  
        - Advertisement and promotional content
        - Cookie banners, popups, modals
        - Social media elements
        - Comments and related articles
        - Meta tags and head elements
        - Elements with ad-related attributes
    """
    return await manager.get_page_text(reduce_noise, ctx)

@mcp.tool()
async def get_page_html(reduce_noise: bool = True, ctx: Context = None) -> MCPResponse:
    """
    Get HTML source code of the current page.
    
    This tool returns the HTML source code of the current page, with optional cleaning
    to remove noise elements while preserving the HTML structure. This is efficient as
    it only gets HTML content without extracting text, screenshots, or other information.
    
    Args:
        reduce_noise (bool): Whether to clean the HTML by removing noise elements like
                           scripts, styles, ads, navigation, etc. while keeping the
                           HTML structure intact. Defaults to True.
    
    Returns:
        MCPResponse: Response object with status, payload (HTML content), and error fields.
    
    Example:
        get_page_html()                # Get cleaned HTML
        get_page_html(reduce_noise=False)  # Get raw HTML source
    
    Note:
        When reduce_noise=True, removes the same elements as get_page_text() but preserves
        the HTML structure. Useful for:
        - Analyzing page structure
        - Finding specific elements
        - Understanding page layout
        - Debugging web automation issues
    """
    if reduce_noise:
        return await manager.get_page_html_cleaned(True, ctx)
    else:
        return await manager.get_page_html_raw(ctx)

@mcp.tool()
async def get_page_url(ctx: Context = None) -> MCPResponse:
    """
    Get the current page URL.
    
    This tool returns only the URL of the current browser page. This is very efficient
    as it only retrieves the URL without any other page processing.
    
    Returns:
        MCPResponse: Response object with status, payload (URL string), and error fields.
    
    Example:
        get_page_url()  # Returns something like "https://example.com/page"
    
    Note:
        This is the most efficient tool if you only need the current URL.
    """
    return await manager.get_page_url(ctx)

@mcp.tool()
async def get_page_title(ctx: Context = None) -> MCPResponse:
    """
    Get the current page title.
    
    This tool returns only the title of the current browser page. This is efficient
    as it only retrieves the title without processing HTML content or taking screenshots.
    
    Returns:
        MCPResponse: Response object with status, payload (title string), and error fields.
    
    Example:
        get_page_title()  # Returns something like "Example Page Title"
    
    Note:
        This is efficient if you only need the page title for navigation or identification.
    """
    return await manager.get_page_title(ctx)

if __name__ == "__main__":
    print("Playwright MCP on", f"{MCP_HOST}:{MCP_PORT}")
    try:
        mcp.run(transport="streamable-http")
    finally:
        asyncio.run(manager.close())
