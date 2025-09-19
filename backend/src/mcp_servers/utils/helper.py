
from typing import Literal, Union, Optional, Callable
import logging
import pprint
import asyncio
from fastmcp import Context, FastMCP

LogLevel = Literal["info", "debug", "warning", "error", "exception"]

async def log(
    message: str, 
    level: LogLevel = "info", 
    logger: Optional[logging.Logger] = None, 
    ctx: Optional[Context] = None,
    exception: Optional[Exception] = None
) -> None:
    """
    Unified logging function that logs to both Python logger and MCP context.
    Falls back to pretty print if logger is not provided.
    
    Args:
        message: The message to log
        level: The log level (info, debug, warning, error, exception)
        logger: Python logger instance (optional)
        ctx: MCP context for request-specific logging (optional)
        exception: Optional exception object to include in error/exception logging
    """
    # Handle None cases - use pretty print if neither logger nor ctx is provided
    if logger is None:
        level_prefix = f"[{level.upper()}]"
        if exception:
            pprint.pprint(f"{level_prefix} {message}: {str(exception)}")
        else:
            pprint.pprint(f"{level_prefix} {message}")
        return
    
    # Log to Python logger if provided
    if logger is not None:
        if level == "info":
            logger.info(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            if exception:
                logger.error(f"{message}: {str(exception)}", exc_info=exception)
            else:
                logger.error(message)
        elif level == "exception":
            if exception:
                logger.exception(f"{message}: {str(exception)}", exc_info=exception)
            else:
                logger.exception(message)
    
    # Log to MCP context if provided
    if ctx is not None:
        if level == "info":
            await ctx.info(message)
        elif level == "debug":
            await ctx.debug(message)
        elif level == "warning":
            await ctx.warning(message)
        elif level == "error":
            if exception:
                await ctx.error(f"{message}: {str(exception)}")
            else:
                await ctx.error(message)
        elif level == "exception":
            if exception:
                await ctx.error(f"{message}: {str(exception)}")
            else:
                await ctx.error(message)


async def start_mcp_server(
    mcp: FastMCP,
    host: str,
    port: int,
    logger: Optional[logging.Logger] = None,
    log_info_fun: Optional[Callable[[], None]] = None
) -> None:
    """
    Standardized MCP server startup function.
    
    Args:
        mcp: The FastMCP server instance
        logger: Optional logger instance
        log_info_fun: Optional function to call before starting the server
    """
    await log(f"Server will run on {host}:{port}", "info", logger, None)
    try:
        await log(f"=== Starting {mcp.name} MCP Server ===", "info", logger, None)
        
        if log_info_fun:
            log_info_fun()
        
        await log("Server initialized and ready to handle connections", "info", logger, None)
        mcp.run(transport="streamable-http", host=host, port=port)
        
    except Exception as e:
        await log(f"Server crashed: {str(e)}", "exception", logger, None, exception=e)
        raise
    finally:
        await log(f"=== {mcp.name} MCP Server shutting down ===", "info", logger, None)