from a2a_servers import configure_logging
import os
import yaml
from pathlib import Path

configure_logging()

AGENTS_CONFIG_PATH = os.getenv("AGENTS_CONFIG_PATH", "agents.yaml")

# load the agents.yaml file
with open(AGENTS_CONFIG_PATH, "r") as f:
    AGENT_CONFIG = yaml.safe_load(f)

CURRENT_DIR = Path(__file__).parent
AGENT_CONFIG_DIR = Path(os.getenv("AGENT_CONFIG_DIR", (CURRENT_DIR / ".." / ".." / "config").expanduser().resolve()))

AGENT_CONFIG_AGENTS_DIR = AGENT_CONFIG_DIR / "agents"
AGENT_CONFIG_MODELS_DIR = AGENT_CONFIG_DIR / "models"
AGENT_CONFIG_PROMPTS_DIR = AGENT_CONFIG_DIR / "prompts"


