import asyncio
import logging
import os
from pathlib import Path
import click
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langgraph.errors import GraphRecursionError
from langchain.chat_models.base import BaseChatModel
from python_a2a import AgentSkill
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.utils.errors import ServerError
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from langchain_mcp_adapters.client import MultiServerMCPClient
import uvicorn

from a2a_servers.constants import AGENT_CONFIG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: BaseChatModel,
        status_message: str = "Processing request...",
        artifact_name: str = "response",
    ):
        """Initialize a generic LangGraph agent executor.

        Args:
            agent: The LangGraph agent instance that implements AgentInterface
            status_message: Message to display while processing
            artifact_name: Name for the response artifact
        """
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Update status with custom message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(self.status_message, task.contextId, task.id),
            )

            # Process with agent using the common interface
            response_text = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            # Take messages and convert to text parts

            if isinstance(response_text, str):
                text_parts = [Part(root=TextPart(text=response_text))]
            else:
                messages = response_text.get("messages")
                text_parts = (
                    [Part(root=TextPart(text=message.content)) for message in messages]
                    if messages
                    else []
                )

                # Take structured response and convert to data part
                structured_response: BaseModel = response_text.get(
                    "structured_response"
                )
                data_parts = (
                    [
                        Part(
                            root=DataPart(
                                data=structured_response.model_dump(mode="python")
                            )
                        )
                    ]
                    if structured_response
                    else []
                )

            # Add response as artifact with custom name
            await updater.add_artifact(
                text_parts + data_parts,
                name=self.artifact_name,
            )

            await updater.complete()

        except GraphRecursionError as e:
            logger.error(f"Error: {e!s}")
            all_parts = []
            for msg in task.history:
                logger.error(f"Message: {msg!s}")
                all_parts.extend(msg.parts)
            all_parts.append(Part(root=TextPart(text=f"Error: {e!s}")))

            await updater.update_status(
                TaskState.failed,
                new_agent_parts_message(all_parts, task.contextId, task.id),
                final=True,
            )
        except Exception as e:
            logger.error(f"Error: {e!s}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.contextId, task.id),
                final=True,
            )

    async def execute_streaming(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute with streaming - not supported by this agent."""
        # Since streaming is disabled, we'll just call the regular execute method
        # and let the framework handle the streaming response
        await self.execute(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Implementation for cancelling tasks
        pass


async def create_sub_agent(agent_name: str, host: str, port: int):
    """
    Create a sub agent for the main agent based on the agents.yml file in which all the agents are registered.
    """
    try:
        # TODO: Add skills from the agent config
        skills = [
            AgentSkill(
                id="search",
                name="Web Research Tool",
                description="Searches the web for information and provides relevant URLs",
                tags=["research", "web search", "information gathering"],
                examples=["Find me websites with real estate listings in Rotterdam"],
            )
        ]

        capabilities = AgentCapabilities(streaming=False, pushNotifications=False)
        agent_config = AGENT_CONFIG["sub_agents"][agent_name]
        tool_config = agent_config.get("tools", [])
        meta_prompt = agent_config.get("meta_prompt", "You are a helpful assistant that can use multiple tools")
        prompt_file = Path(agent_config.get("prompt_file", "default.txt"))

        tool_config_dict = {tool["name"]: tool["mcp_server"] for tool in tool_config}

        agent_card = AgentCard(
            name=agent_config["name"],
            description=agent_config["description"],
            # url=f"http://{host}:{port}/",
            url=f"http://host.docker.internal:{port}/",
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        client = MultiServerMCPClient(tool_config_dict)
        tools = await client.get_tools()

        prompt = f"{meta_prompt}\n\n{prompt_file.read_text()}" if prompt_file.exists() else meta_prompt

        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=prompt
        )

        # Create executor with custom parameters
        executor = BaseAgentExecutor(
            agent=agent,
            status_message="Processing request...",
            artifact_name="response",
        )
        request_handler = DefaultRequestHandler(  
            agent_executor=executor,  
            task_store=InMemoryTaskStore()  
        ) 
        return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
    except Exception as e:
        logger.error(f"Error: {e!s}")
        raise e


@click.command()
@click.option("--agent-name", default="web_search_agent", help="Name of the agent to run")
@click.option("--host", default="localhost", help="Host to run the agent on")
@click.option("--port", default=10020, help="Port to run the agent on")
@click.option("--log-level", default="info", help="Log level to run the agent on")
def run_agent_server(agent_name: str, host: str, port: int, log_level: str):
    app = asyncio.run(create_sub_agent(agent_name, host, port))
    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_agent_server()
