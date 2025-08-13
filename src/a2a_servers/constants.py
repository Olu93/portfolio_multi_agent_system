from a2a_servers import configure_logging
import os
import yaml

configure_logging()

AGENTS_CONFIG_PATH = os.getenv("AGENTS_CONFIG_PATH", "agents.yaml")

# load the agents.yaml file
with open(AGENTS_CONFIG_PATH, "r") as f:
    AGENT_CONFIG = yaml.safe_load(f)






