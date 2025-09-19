from fastmcp import FastMCP, Context
from mcp_servers.utils.models import MCPResponse
from mcp_servers.utils.helper import log, start_mcp_server
from mcp_servers.utils.constants import MCP_HOST, MCP_PORT
import pathlib
import logging
import base64
import asyncio
from typing import Literal

logger = logging.getLogger(__name__)

mcp = FastMCP("file_storage", host=MCP_HOST, port=MCP_PORT)

DEFAULT_STORAGE_PATH = "files/downloads"

# Supported file types
SupportedFileType = Literal[".jpeg", ".jpg", ".png", ".json", ".jsonl", ".txt", ".html", ".xml", ".csv"]


@mcp.tool()
async def bytes_to_base64(data_bytes: bytes, ctx: Context) -> MCPResponse:
    """
    Convert bytes data to base64 encoded string.

    Args:
        data_bytes: The bytes data to convert to base64
        ctx: MCP context for logging
    """
    try:
        await log(f"Converting {len(data_bytes)} bytes to base64", "info", logger, ctx)

        base64_string = base64.b64encode(data_bytes).decode("utf-8")

        await log(
            f"Successfully converted {len(data_bytes)} bytes to base64 string of length {len(base64_string)}",
            "info",
            logger,
            ctx,
        )

        return MCPResponse(
            status="OK",
            payload={
                "message": "Bytes converted to base64 successfully",
                "base64_data": base64_string,
                "data_length": len(data_bytes),
            },
        )
    except Exception as e:
        await log(f"Failed to convert bytes to base64: {str(e)}", "error", logger, ctx, exception=e)
        return MCPResponse(status="ERR", payload={"message": "Failed to convert bytes to base64"}, error=str(e))


@mcp.tool()
async def store_file(base64_data: str, file_name: str, file_type: SupportedFileType, ctx: Context) -> MCPResponse:
    """
    Store base64 encoded data as a file in the file storage.

    Args:
        base64_data: The base64 encoded string data to store
        file_name: The name of the file (without extension)
        file_type: The file type/extension (.jpeg, .jpg, .png, .json, .jsonl, .txt, .html, .xml, .csv)
        ctx: MCP context for logging
    """
    try:
        await log(f"Storing file '{file_name}{file_type}' in {DEFAULT_STORAGE_PATH}", "info", logger, ctx)

        # Ensure storage directory exists
        if not pathlib.Path(DEFAULT_STORAGE_PATH).exists():
            await log(f"Creating storage directory: {DEFAULT_STORAGE_PATH}", "info", logger, ctx)
            pathlib.Path(DEFAULT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)

        # Create full file path with extension
        full_file_name = f"{file_name}{file_type}"
        file_path = pathlib.Path(DEFAULT_STORAGE_PATH, full_file_name)

        await log(f"Full file path: {file_path}", "debug", logger, ctx)

        # Decode base64 data
        await log(f"Decoding base64 data of length {len(base64_data)}", "debug", logger, ctx)
        file_data = base64.b64decode(base64_data)

        await log(f"Decoded {len(file_data)} bytes from base64", "info", logger, ctx)

        # Write file based on type
        if file_type in [".json", ".jsonl", ".txt", ".html", ".xml", ".csv"]:
            # Text-based files
            await log(f"Writing as text file with UTF-8 encoding", "debug", logger, ctx)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_data.decode("utf-8"))
        else:
            # Binary files (images, etc.)
            await log(f"Writing as binary file", "debug", logger, ctx)
            with open(file_path, "wb") as f:
                f.write(file_data)

        await log(
            f"Successfully stored file '{full_file_name}' ({len(file_data)} bytes)",
            "info",
            logger,
            ctx,
        )

        return MCPResponse(
            status="OK",
            payload={
                "message": "File stored successfully",
                "file_name": full_file_name,
                "file_path": str(file_path),
                "file_type": file_type,
                "file_size": len(file_data),
            },
        )
    except Exception as e:
        await log(
            f"Failed to store file '{file_name}{file_type}': {str(e)}",
            "error",
            logger,
            ctx,
            exception=e,
        )
        return MCPResponse(status="ERR", payload={"message": "Failed to store file"}, error=str(e))


@mcp.tool()
async def get_file(file_name: str, file_type: SupportedFileType, ctx: Context) -> MCPResponse:
    """
    Get a file from the file storage and return as base64 encoded string.

    Args:
        file_name: The name of the file (without extension)
        file_type: The file type/extension
        ctx: MCP context for logging
    """
    try:
        await log(
            f"Getting file '{file_name}{file_type}' from {DEFAULT_STORAGE_PATH}",
            "info",
            logger,
            ctx,
        )

        # Create full file path
        full_file_name = f"{file_name}{file_type}"
        file_path = pathlib.Path(DEFAULT_STORAGE_PATH, full_file_name)

        await log(f"Looking for file at: {file_path}", "debug", logger, ctx)

        if not file_path.exists():
            await log(f"File not found: {full_file_name}", "warning", logger, ctx)
            return MCPResponse(
                status="ERR",
                payload={"message": "File not found"},
                error=f"File {full_file_name} not found in {DEFAULT_STORAGE_PATH}",
            )

        await log(f"File found, reading {file_path}", "info", logger, ctx)

        # Read file based on type
        if file_type in [".json", ".jsonl", ".txt", ".html", ".xml", ".csv"]:
            # Text-based files
            await log(f"Reading as text file with UTF-8 encoding", "debug", logger, ctx)
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            # Encode text as base64
            base64_data = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")
            file_size = len(file_content.encode("utf-8"))
        else:
            # Binary files
            await log(f"Reading as binary file", "debug", logger, ctx)
            with open(file_path, "rb") as f:
                file_data = f.read()
            # Encode binary as base64
            base64_data = base64.b64encode(file_data).decode("utf-8")
            file_size = len(file_data)

        await log(f"Successfully read file '{full_file_name}' ({file_size} bytes)", "info", logger, ctx)

        return MCPResponse(
            status="OK",
            payload={
                "message": "File retrieved successfully",
                "file_name": full_file_name,
                "file_path": str(file_path),
                "file_type": file_type,
                "base64_data": base64_data,
                "file_size": file_size,
            },
        )
    except Exception as e:
        await log(
            f"Failed to retrieve file '{file_name}{file_type}': {str(e)}",
            "error",
            logger,
            ctx,
            exception=e,
        )
        return MCPResponse(status="ERR", payload={"message": "Failed to retrieve file"}, error=str(e))


@mcp.tool()
async def delete_file(file_name: str, file_type: SupportedFileType, ctx: Context) -> MCPResponse:
    """
    Delete a file from the file storage.

    Args:
        file_name: The name of the file (without extension)
        file_type: The file type/extension
        ctx: MCP context for logging
    """
    try:
        await log(
            f"Deleting file '{file_name}{file_type}' from {DEFAULT_STORAGE_PATH}",
            "info",
            logger,
            ctx,
        )

        # Create full file path
        full_file_name = f"{file_name}{file_type}"
        file_path = pathlib.Path(DEFAULT_STORAGE_PATH, full_file_name)

        await log(f"Looking for file to delete at: {file_path}", "debug", logger, ctx)

        if not file_path.exists():
            await log(f"File not found for deletion: {full_file_name}", "warning", logger, ctx)
            return MCPResponse(
                status="ERR",
                payload={"message": "File not found"},
                error=f"File {full_file_name} not found in {DEFAULT_STORAGE_PATH}",
            )

        # Get file size before deletion for logging
        file_size = file_path.stat().st_size
        await log(f"File found ({file_size} bytes), proceeding with deletion", "info", logger, ctx)

        # Delete the file
        file_path.unlink()

        await log(f"Successfully deleted file '{full_file_name}' ({file_size} bytes)", "info", logger, ctx)

        return MCPResponse(
            status="OK",
            payload={
                "message": "File deleted successfully",
                "file_name": full_file_name,
                "file_path": str(file_path),
            },
        )
    except Exception as e:
        await log(
            f"Failed to delete file '{file_name}{file_type}': {str(e)}",
            "error",
            logger,
            ctx,
            exception=e,
        )
        return MCPResponse(status="ERR", payload={"message": "Failed to delete file"}, error=str(e))


@mcp.tool()
async def list_files(ctx: Context) -> MCPResponse:
    """
    List all files in the storage directory.

    Args:
        ctx: MCP context for logging
    """
    try:
        await log(f"Listing files in {DEFAULT_STORAGE_PATH}", "info", logger, ctx)

        if not pathlib.Path(DEFAULT_STORAGE_PATH).exists():
            await log(f"Storage directory does not exist: {DEFAULT_STORAGE_PATH}", "info", logger, ctx)
            return MCPResponse(status="OK", payload={"message": "Storage directory does not exist", "files": []})

        storage_path = pathlib.Path(DEFAULT_STORAGE_PATH)
        files = []

        await log(f"Scanning directory: {storage_path}", "debug", logger, ctx)

        for file_path in storage_path.iterdir():
            if file_path.is_file():
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                }
                files.append(file_info)
                await log(
                    f"Found file: {file_path.name} ({file_info['size']} bytes)",
                    "debug",
                    logger,
                    ctx,
                )

        await log(f"Found {len(files)} files in storage directory", "info", logger, ctx)

        return MCPResponse(status="OK", payload={"message": f"Found {len(files)} files", "files": files})
    except Exception as e:
        await log(
            f"Failed to list files in {DEFAULT_STORAGE_PATH}: {str(e)}",
            "error",
            logger,
            ctx,
            exception=e,
        )
        return MCPResponse(status="ERR", payload={"message": "Failed to list files"}, error=str(e))


async def main():
    """Main function to start the File Storage MCP server"""

    def log_info():
        log(f"Storage directory: {DEFAULT_STORAGE_PATH}", "info", logger, None)

    await start_mcp_server(mcp, MCP_HOST, MCP_PORT, logger, log_info)


if __name__ == "__main__":
    asyncio.run(main())
