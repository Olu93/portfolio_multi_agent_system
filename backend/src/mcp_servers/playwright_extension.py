import os, json, asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
from mcp.server.fastmcp import FastMCP, Context
from playwright.async_api import async_playwright, Browser, Page
from lxml import html
from lxml_html_clean import Cleaner
import logging
import base64

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

    async def get_current_page_info(self, include_screenshot: bool, reduce_noise: bool, ctx: Optional[Context]) -> Optional[PageInfo]:
        try:
            if not self.page:
                msg = "No page. Launch the browser first."
                log.warning(msg)
                if ctx: await ctx.error(msg)
                return None

            if ctx: await ctx.info("Collecting page info…")
            url = self.page.url
            title = await self.page.title()
            html_content = await self.page.content()
            
            # Use Beautiful Soup for text extraction instead of JavaScript
            text_content = extract_text_lxml(html_content, reduce_noise)
            
            base64_img = None
            if include_screenshot:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                # screenshot_path = f"screenshot_{ts}.png"
                img_bytes = await self.page.screenshot(full_page=True)
                base64_img = base64.b64encode(img_bytes).decode("utf-8")
                # log.info(f"Screenshot saved: {screenshot_path}")
                # if ctx: await ctx.info(f"Screenshot saved: {screenshot_path}")

            info = PageInfo(
                url=url, title=title, html_content=html_content, text_content=text_content,
                screenshot=base64_img, timestamp=datetime.now().isoformat()
            )
            log.debug(f"Page info collected: title={title}, url={url}")
            if ctx: await ctx.info("Page info collected.")
            return info
        except Exception as e:
            log.exception("Get page info failed")
            if ctx: await ctx.error(f"Get page info failed: {e}")
            return None

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
async def launch_browser(headless: bool = False, ctx: Context = None) -> str:
    """
    Launch a new browser instance using Playwright.
    
    This tool starts a browser (Chrome/Chromium by default) that can be used for web automation.
    The browser will be reused if already running and connected.
    
    Args:
        headless (bool): Whether to run the browser in headless mode (no GUI).
                        Defaults to False (visible browser).
    
    Returns:
        str: "OK" if browser launched successfully, "ERR" if failed.
    
    Example:
        launch_browser(headless=True)  # Start headless browser
        launch_browser()               # Start visible browser
    """
    return "OK" if await manager.launch_browser(ctx, headless=headless) else "ERR"

@mcp.tool()
async def navigate_to_url(url: str, ctx: Context = None) -> str:
    """
    Navigate the browser to a specific URL.
    
    This tool loads a web page in the current browser tab. The browser must be launched first.
    The tool waits for the page to fully load (network idle) before completing.
    
    Args:
        url (str): The URL to navigate to. Must include protocol (http:// or https://).
    
    Returns:
        str: "OK" if navigation successful, "ERR" if failed.
    
    Example:
        navigate_to_url("https://example.com")
        navigate_to_url("https://google.com")
    
    Note:
        Requires browser to be launched first using launch_browser().
    """
    return "OK" if await manager.navigate_to(url, ctx) else "ERR"

@mcp.tool()
async def wait_for_element(selector: str, timeout_ms: int = 30000, ctx: Context = None) -> str:
    """
    Wait for an element to appear on the current page.
    
    This tool waits for a specific element to be present in the DOM before proceeding.
    Useful for ensuring elements are loaded before interacting with them.
    
    Args:
        selector (str): CSS selector for the element to wait for.
                       Examples: "#id", ".class", "button", "[data-testid='submit']"
        timeout_ms (int): Maximum time to wait in milliseconds. Defaults to 30000 (30 seconds).
    
    Returns:
        str: "OK" if element found within timeout, "ERR" if timeout exceeded or failed.
    
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
    return "OK" if await manager.wait_for_element(selector, timeout_ms, ctx) else "ERR"

@mcp.tool()
async def click_element(selector: str, ctx: Context = None) -> str:
    """
    Click on an element on the current page.
    
    This tool simulates a mouse click on a specific element. The element must be visible and clickable.
    
    Args:
        selector (str): CSS selector for the element to click.
                       Examples: "#submit", ".button", "button[type='submit']"
    
    Returns:
        str: "OK" if click successful, "ERR" if element not found or not clickable.
    
    Example:
        click_element("#login-button")
        click_element(".submit-form")
        click_element("button[data-action='save']")
    
    Note:
        Element must be visible and not covered by other elements.
        Use wait_for_element() first if element might not be immediately available.
    """
    return "OK" if await manager.click_element(selector, ctx) else "ERR"

@mcp.tool()
async def fill_form(form_data: str, ctx: Context = None) -> str:
    """
    Fill form fields on the current page.
    
    This tool fills multiple form fields at once using CSS selectors as keys.
    All fields are filled in sequence.
    
    Args:
        form_data (str): JSON string containing field selectors and values.
                        Format: '{"#username": "john", "#password": "secret", ".email": "john@example.com"}'
    
    Returns:
        str: "OK" if all fields filled successfully, "ERR" if any field failed or invalid JSON.
    
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
    except Exception:
        return "ERR: form_data must be JSON"
    return "OK" if await manager.fill_form(data, ctx) else "ERR"

@mcp.tool()
async def get_current_page(include_screenshot: bool = False, ctx: Context = None) -> str:
    """
    Get comprehensive information about the current page.
    
    This tool returns detailed information about the current browser page including URL, title,
    HTML content, and optionally a screenshot. Text content is automatically cleaned of noise.
    
    Args:
        include_screenshot (bool): Whether to capture and save a screenshot of the current page.
                                 Screenshots are saved as PNG files with timestamp names.
                                 Defaults to False.
    
    Returns:
        str: JSON string containing page information with keys:
             - url: Current page URL
             - title: Page title
             - html_content: Full HTML source
             - text_content: Cleaned text content (noise removed)
             - screenshot_path: Path to screenshot file (if include_screenshot=True)
             - timestamp: When the data was collected
    
    Example:
        get_current_page()                    # Get basic page info
        get_current_page(include_screenshot=True)  # Get page info + screenshot
    
    Note:
        Text content is automatically cleaned using Beautiful Soup to remove scripts,
        styles, ads, navigation elements, and other noise for better readability.
    """
    info = await manager.get_current_page_info(include_screenshot, True, ctx)
    return json.dumps(info.__dict__, indent=2) if info else "ERR"


@mcp.tool()
async def get_page_screenshot(ctx: Context = None) -> str:
    """
    Get a screenshot of the current page.
    
    This tool returns a screenshot of the current page as a base64 encoded string.
    """
    info = await manager.get_current_page_info(True, False, ctx)
    return base64.b64encode(info.screenshot).decode("utf-8") if info else "ERR"

@mcp.tool()
async def get_page_text(reduce_noise: bool = True, ctx: Context = None) -> str:
    """
    Extract clean text content from the current page.
    
    This tool extracts only the text content from the current page, with optional noise reduction.
    Uses Beautiful Soup for robust text extraction and cleaning.
    
    Args:
        reduce_noise (bool): Whether to remove noise elements like scripts, styles, ads,
                           navigation, headers, footers, etc. Defaults to True.
    
    Returns:
        str: Clean text content of the page, or "ERR" if extraction failed.
    
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
    info = await manager.get_current_page_info(False, reduce_noise, ctx)
    return info.text_content if info else "ERR"

@mcp.tool()
async def get_page_html(reduce_noise: bool = True, ctx: Context = None) -> str:
    """
    Get HTML source code of the current page.
    
    This tool returns the HTML source code of the current page, with optional cleaning
    to remove noise elements while preserving the HTML structure.
    
    Args:
        reduce_noise (bool): Whether to clean the HTML by removing noise elements like
                           scripts, styles, ads, navigation, etc. while keeping the
                           HTML structure intact. Defaults to True.
    
    Returns:
        str: HTML source code of the page (cleaned if reduce_noise=True), or "ERR" if failed.
    
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
    try:
        if not manager.page:
            msg = "No page. Launch the browser first."
            log.warning(msg)
            if ctx: await ctx.error(msg)
            return "ERR"
        
        html_content = await manager.page.content()
        
        if reduce_noise:
            html_content = clean_html_lxml(html_content, True)
        
        return html_content
    except Exception as e:
        log.exception("Get page HTML failed")
        if ctx: await ctx.error(f"Get page HTML failed: {e}")
        return "ERR"

if __name__ == "__main__":
    print("Playwright MCP on", f"{MCP_HOST}:{MCP_PORT}")
    try:
        mcp.run(transport="streamable-http")
    finally:
        asyncio.run(manager.close())
