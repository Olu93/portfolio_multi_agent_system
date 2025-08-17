import asyncio
import logging
import os
import sys
from typing import Any, List, Optional, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel
import uvicorn

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.utils.errors import ServerError
from dotenv import load_dotenv, find_dotenv
from langgraph.errors import GraphRecursionError

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


class AgentInterface(Protocol):
    """Common interface that all agents must implement."""
    
    async def process_query(self, query: str) -> str:
        """Process a query and return the result as a string."""
        ...


class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: AgentInterface,
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

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Update status with custom message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    self.status_message, task.contextId, task.id
                ),
            )

            # Process with agent using the common interface
            response_text = await self.agent.process_query(query)
            # Take messages and convert to text parts

            if isinstance(response_text, str):
                text_parts = [Part(root=TextPart(text=response_text))]
            else:
                messages = response_text.get("messages")
                text_parts = [Part(root=TextPart(text=message.content)) for message in messages] if messages else []

                # Take structured response and convert to data part
                structured_response:BaseModel = response_text.get("structured_response")
                data_parts = [Part(root=DataPart(data=structured_response.model_dump(mode="python")))] if structured_response else []

            # Add response as artifact with custom name
            await updater.add_artifact(
                text_parts+data_parts,
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

    @staticmethod
    def create_agent_a2a_server(
        agent: AgentInterface,
        name: str,
        description: str,
        skills: List[AgentSkill],
        host: str = "localhost",
        port: int = 10020,
        status_message: str = "Processing request...",
        artifact_name: str = "response",
    ):
        """Create an A2A server for any LangGraph agent.

        Args:
            agent: The LangGraph agent instance that implements AgentInterface
            name: Display name for the agent
            description: Agent description
            skills: List of AgentSkill objects
            host: Server host
            port: Server port
            status_message: Message shown while processing
            artifact_name: Name for response artifacts

        Returns:
            A2AStarletteApplication instance
        """
        # Agent capabilities - explicitly disable streaming and push notifications
        capabilities = AgentCapabilities(streaming=False, pushNotifications=False)

        # Agent card (metadata)
        agent_card = AgentCard(
            name=name,
            description=description,
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        # Create executor with custom parameters
        executor = BaseAgentExecutor(
            agent=agent,
            status_message=status_message,
            artifact_name=artifact_name,
        )

        # Create a custom request handler that explicitly disables streaming
        class NonStreamingRequestHandler(DefaultRequestHandler):
            def __init__(self, agent_executor, task_store):
                super().__init__(agent_executor=agent_executor, task_store=task_store)
            
            async def on_message_send_stream(self, request, context):
                """Override to disable streaming and use regular message handling."""
                # Redirect streaming requests to regular message handling
                return await self.on_message_send(request, context)

        request_handler = NonStreamingRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        # Create A2A application
        return A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )


def run_agent_server(
    create_agent_function,
    host: str = "localhost",
    port: int = 10020,
    log_level: str = "info",
):
    """Run an agent server with proper error handling."""
    try:
        # Check for required environment variables
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError(
                "OPENAI_API_KEY environment variable not set."
            )

        print(f"🚀 Starting agent on {host}:{port}...")
        app = create_agent_function()
        uvicorn.run(
            app.build(),
            host=host,
            port=port,
            log_level=log_level,
        )

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1) 