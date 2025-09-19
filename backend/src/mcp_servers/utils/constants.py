"""
Centralized constants and environment configuration for MCP servers.
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# MCP Server Configuration
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
