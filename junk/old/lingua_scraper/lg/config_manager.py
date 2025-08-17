"""
Configuration Manager for LangGraph Supervisor Multi-Agent System
Uses Pydantic Settings with YAML support for type-safe configuration
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


# ============================================================================
# Configuration Models
# ============================================================================

class LLMConfig(BaseModel):
    """LLM configuration for agents and supervisor"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4000


class MCPServerConfig(BaseModel):
    """MCP server configuration"""
    url: str
    transport: str = "streamable_http"


class AgentConfig(BaseModel):
    """Individual agent configuration"""
    name: str
    description: str
    llm: LLMConfig
    system_prompt: str
    mcp_servers: Dict[str, MCPServerConfig]
    response_format: Optional[str] = None


class SupervisorConfig(BaseModel):
    """Supervisor configuration"""
    llm: LLMConfig
    prompt_template: str


# ============================================================================
# Custom YAML Settings Source
# ============================================================================

class YAMLConfigSettingsSource(PydanticBaseSettingsSource):
    """
    Custom settings source that loads configuration from a YAML file.
    Based on the Pydantic Settings documentation pattern.
    """
    
    def __init__(self, settings_cls: type[BaseSettings], yaml_file_path: str = "config/agents.yaml"):
        super().__init__(settings_cls)
        self.yaml_file_path = Path(yaml_file_path)
    
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from YAML file"""
        if not self.yaml_file_path.exists():
            return None, field_name, False
        
        try:
            import yaml
            with open(self.yaml_file_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)
            
            # Handle nested field access (e.g., "agents.research_agent")
            field_value = yaml_content
            for part in field_name.split('.'):
                if isinstance(field_value, dict) and part in field_value:
                    field_value = field_value[part]
                else:
                    return None, field_name, False
            
            return field_value, field_name, False
            
        except Exception as e:
            print(f"Warning: Could not load YAML config from {self.yaml_file_path}: {e}")
            return None, field_name, False
    
    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        """Prepare field value for validation"""
        return value
    
    def __call__(self) -> dict[str, Any]:
        """Load all fields from YAML file"""
        d: dict[str, Any] = {}
        
        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value
        
        return d


# ============================================================================
# Main Configuration Class
# ============================================================================

class SystemConfig(BaseSettings):
    """Main configuration class for the multi-agent system"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # Supervisor configuration
    supervisor: SupervisorConfig = Field(default_factory=lambda: SupervisorConfig(
        llm=LLMConfig(),
        prompt_template="You manage a multi-agent system."
    ))
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML file"""
        return (
            init_settings,
            YAMLConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
    
    def get_agent_descriptions(self) -> str:
        """Generate agent descriptions for supervisor prompt"""
        descriptions = []
        for i, (agent_name, agent_config) in enumerate(self.agents.items(), 1):
            descriptions.append(f"{i}. **{agent_name}**: {agent_config.description}")
        return "\n    ".join(descriptions)
    
    def get_supervisor_prompt(self) -> str:
        """Get the supervisor prompt with agent descriptions injected"""
        agent_descriptions = self.get_agent_descriptions()
        return self.supervisor.prompt_template.format(
            agent_descriptions=agent_descriptions
        )


# ============================================================================
# Configuration Manager Singleton
# ============================================================================

class ConfigManager:
    """Singleton configuration manager"""
    
    _instance: Optional[SystemConfig] = None
    
    @classmethod
    def get_config(cls, yaml_file_path: str = "config/agents.yaml") -> SystemConfig:
        """Get or create the configuration instance"""
        if cls._instance is None:
            # Create a custom source with the specified YAML file path
            class CustomYAMLSource(YAMLConfigSettingsSource):
                def __init__(self, settings_cls: type[BaseSettings]):
                    super().__init__(settings_cls, yaml_file_path)
            
            class CustomSystemConfig(SystemConfig):
                @classmethod
                def settings_customise_sources(
                    cls,
                    settings_cls: type[BaseSettings],
                    init_settings: PydanticBaseSettingsSource,
                    env_settings: PydanticBaseSettingsSource,
                    dotenv_settings: PydanticBaseSettingsSource,
                    file_secret_settings: PydanticBaseSettingsSource,
                ) -> tuple[PydanticBaseSettingsSource, ...]:
                    return (
                        init_settings,
                        CustomYAMLSource(settings_cls),
                        env_settings,
                        dotenv_settings,
                        file_secret_settings,
                    )
            
            cls._instance = CustomSystemConfig()
        
        return cls._instance
    
    @classmethod
    def reload_config(cls, yaml_file_path: str = "config/agents.yaml") -> SystemConfig:
        """Reload configuration from file"""
        cls._instance = None
        return cls.get_config(yaml_file_path) 