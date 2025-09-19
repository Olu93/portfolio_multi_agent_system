import os
from pathlib import Path


from a2a_servers import configure_logging

configure_logging()

CURRENT_DIR = Path(__file__).parent
AGENT_CONFIG_DIR = Path(os.getenv("AGENT_CONFIG_DIR", (CURRENT_DIR / ".." / ".." / "config").expanduser().resolve()))

AGENT_CONFIG_AGENTS_DIR = AGENT_CONFIG_DIR / "agents"
AGENT_CONFIG_MODELS_DIR = AGENT_CONFIG_DIR / "models"
AGENT_CONFIG_PROMPTS_DIR = AGENT_CONFIG_DIR / "prompts"
