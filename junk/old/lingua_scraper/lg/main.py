"""
LangGraph Supervisor Multi-Agent System
Configuration-driven version using YAML and Pydantic Settings
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool

from .agent_factory import create_all_agents, create_supervisor_llm, get_supervisor_prompt

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Supervisor Setup
# ============================================================================

async def create_supervisor_system(config_path: str = "config/agents.yaml"):
    """Create the supervisor multi-agent system using configuration"""
    
    @tool
    def explain_step(explanation: str) -> str:
        """Explain what you did and why you did it."""
        return f"{explanation}"

    print("Creating individual agents...")
    
    # Create all agents from configuration
    agents = await create_all_agents(config_path)
    agent_list = list(agents.values())
    
    print(f"Created {len(agent_list)} agents: {list(agents.keys())}")
    
    print("Creating supervisor...")
    
    # Create supervisor LLM and prompt from configuration
    supervisor_llm = await create_supervisor_llm(config_path)
    supervisor_prompt = get_supervisor_prompt(config_path)
    
    # Create supervisor
    supervisor = create_supervisor(
        tools=[explain_step],
        agents=agent_list,
        model=supervisor_llm,
        prompt=supervisor_prompt
    ).compile()
    
    return supervisor


# ============================================================================
# Main Execution
# ============================================================================

async def run_query(supervisor: CompiledStateGraph, query: str):
    """Run a query through the supervisor system"""
    print(f"\n=== Processing Query: {query} ===")
    
    try:
        async for chunk in supervisor.astream({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }):
            print(chunk)
            print("\n")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        logger.error(f"Error during execution: {str(e)}", exc_info=True)


async def interactive_mode(supervisor):
    """Run the system in interactive mode"""
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'help' for example queries")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nExample queries:")
                print("- 'Find real estate agents in Amsterdam'")
                print("- 'Scrape content from https://example.com'")
                print("- 'Find 30 real estate agents and collect their contact information'")
                print("- 'Research sustainable energy companies in Rotterdam'")
                continue
            elif not query:
                continue
            
            await run_query(supervisor, query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


async def main():
    """Main function to run the supervisor system"""
    
    print("=== Initializing LangGraph Supervisor Multi-Agent System ===")
    
    # Get config path from command line or use default
    config_path = "config/agents.yaml"
    if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
        config_path = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # Remove config path from args
    
    # Create the supervisor system
    supervisor = await create_supervisor_system(config_path)
    with open("supervisor.png", "wb") as f:
        f.write(supervisor.get_graph().draw_mermaid_png())
    
    print("Supervisor system created successfully!")
    
    # Get agent information from configuration
    from .config_manager import ConfigManager
    config = ConfigManager.get_config(config_path)
    print(f"Available agents: {', '.join(config.agents.keys())}")
    print("\nAgent capabilities:")
    for agent_name, agent_config in config.agents.items():
        print(f"- {agent_name}: {agent_config.description}")
    
    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        await run_query(supervisor, query)
    else:
        # Run in interactive mode
        await interactive_mode(supervisor)


if __name__ == "__main__":
    asyncio.run(main())
