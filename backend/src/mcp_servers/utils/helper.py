
from typing import Literal, Union, Optional
import logging
import pprint
from fastmcp import Context

LogLevel = Literal["info", "debug", "warning", "error", "exception"]

def log(
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
            ctx.info(message)
        elif level == "debug":
            ctx.debug(message)
        elif level == "warning":
            ctx.warning(message)
        elif level == "error":
            if exception:
                ctx.error(f"{message}: {str(exception)}")
            else:
                ctx.error(message)
        elif level == "exception":
            if exception:
                ctx.error(f"{message}: {str(exception)}")
            else:
                ctx.error(message)