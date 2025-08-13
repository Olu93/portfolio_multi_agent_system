from fastmcp import FastMCP, Context
import pathlib
import json
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8003"))

mcp = FastMCP("file_storage", host=MCP_HOST, port=MCP_PORT)

DEFAULT_STORAGE_PATH = "files/downloads"

@mcp.tool()
async def store_dict(data:dict, ctx:Context, file_name:str) -> str:
    """
    Store a dictionary in the file storage.

    Args:
        data: The dictionary to store in the file.
        ctx: MCP context for logging
        file_name: The name of the file to store the data in. The file ending is .json
    """
    ctx.info(f"Storing data in {DEFAULT_STORAGE_PATH}")
    if not pathlib.Path(DEFAULT_STORAGE_PATH).exists():
        pathlib.Path(DEFAULT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(DEFAULT_STORAGE_PATH, file_name), "w") as f:
        json.dump(data, f, indent=4)
    return "File stored successfully"


if __name__ == "__main__":
    print("=== Starting File Storage MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}", exc_info=True)
        raise
    finally:
        print("=== File Storage MCP Server shutting down ===") 







