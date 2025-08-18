from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource
from typing import List
from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from constants import AGENTS_CONFIG_PATH
import logging

logger = logging.getLogger(__name__)

def kebab_alias(name: str) -> str:
    return name.replace("_", "-")

class HyphenModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=kebab_alias)


class MCPServer(HyphenModel):
    url: HttpUrl
    transport: str

class Tool(HyphenModel):
    name: str
    description: str
    mcp_server: MCPServer  # maps to "mcp-server" in YAML

class SubAgent(HyphenModel):
    name: str
    description: str
    tools: List[Tool]
    meta_prompt: str  # "meta-prompt" in YAML
    prompt_file: str  # "prompt-file" in YAML


class Settings(BaseSettings):
    sub_agents: List[SubAgent]
    model_config = ConfigDict(populate_by_name=True, alias_generator=kebab_alias)


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        yaml_file = AGENTS_CONFIG_PATH
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file),
            file_secret_settings,
        )


if __name__ == "__main__":
    settings = Settings()
    logging.info(settings.sub_agents[0].tools[0].mcp_server.url)
