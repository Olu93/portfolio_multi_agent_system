import logging
from a2a_servers.constants import AGENT_CONFIG_AGENTS_DIR, AGENT_CONFIG_MODELS_DIR, AGENT_CONFIG_PROMPTS_DIR
import yaml

logger = logging.getLogger(__name__)

def load_agent_config(agent_name: str):
    file_to_open = AGENT_CONFIG_AGENTS_DIR / f"{agent_name}.yml"
    if not file_to_open.exists():
        logger.error(f"Agent config file {file_to_open} not found")
        raise FileNotFoundError(f"Agent config file {file_to_open} not found")
    with open(file_to_open, "r") as f:
        logger.info(f"Loading agent config for {agent_name}")
        return yaml.safe_load(f)


def load_model_config(model_name: str):
    file_to_open = AGENT_CONFIG_MODELS_DIR / f"{model_name}.yml"
    if not file_to_open.exists():
        logger.error(f"Model config file {file_to_open} not found")
        raise FileNotFoundError(f"Model config file {file_to_open} not found")
    with open(file_to_open, "r") as f:
        logger.info(f"Loading model config for {model_name}")
        return yaml.safe_load(f)


def load_prompt_config(prompt_file_name: str):
    file_to_open = AGENT_CONFIG_PROMPTS_DIR / f"{prompt_file_name}"
    if not file_to_open.exists():
        logger.warning(f"Prompt file {file_to_open} not found, using empty prompt")
        return ""
    with open(file_to_open, "r") as f:
        logger.info(f"Loading prompt config for {prompt_file_name}")
        return f.read()

