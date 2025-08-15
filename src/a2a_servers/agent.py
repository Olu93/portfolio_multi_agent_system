import asyncio
import logging
from typing import Any, AsyncIterable, Literal
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
    # DataPart,
    Part,
    TaskState,
    TextPart,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, ToolMessage
import uvicorn
from langchain.chat_models import init_chat_model

from a2a_servers.config_loader import load_agent_config, load_model_config, load_prompt_config
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

memory = InMemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class Agent:
    """CurrencyAgent - a specialized assistant for currency convesions."""


    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, model, tools, prompt):
        self.model = model
        self.tools = tools
        self.prompt = prompt

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.prompt,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the request...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Executing the tool...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']



class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: Agent,
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

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            async for item in self.agent.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
                    break

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


async def create_sub_agent(agent_name: str):
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
        agent_config = load_agent_config(agent_name)
        tool_config = agent_config.get("tools", [])
        model_config = load_model_config(agent_config.get("model", "default"))	
        meta_prompt = agent_config.get("meta_prompt", "You are a helpful assistant that can use multiple tools")
        prompt_config = load_prompt_config(agent_config.get("prompt_file", f"{agent_name}.txt"))

        tool_config_dict = {tool["name"]: tool["mcp_server"] for tool in tool_config}

        agent_card = AgentCard(
            name=agent_config["name"],
            description=agent_config["description"],
            # url=f"http://{host}:{port}/",
            url=agent_config["agent_url"],
            version="1.0.0",
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=capabilities,
            skills=skills,
        )

        model = init_chat_model(model_config["name"], **model_config["parameters"], model_provider=model_config["provider"])


        for tool in tool_config_dict:
            if not ("/mcp" in tool_config_dict[tool]["url"] or "/sse" in tool_config_dict[tool]["url"]):
                logger.warning(f"Tool {tool} is not using the correct URL format. Please use the correct URL format. The URL is {tool_config_dict[tool]['url']}")

        client = MultiServerMCPClient(tool_config_dict)
        tools = await client.get_tools()

        prompt = f"{meta_prompt}\n\n{prompt_config}"

        agent = Agent(model, tools, prompt)

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
@click.option("--host", default="0.0.0.0", help="Host to run the agent on")
@click.option("--port", default=10020, help="Port to run the agent on")
@click.option("--log-level", default="info", help="Log level to run the agent on")
def run_agent_server(agent_name: str, host: str, port: int, log_level: str):
    app = asyncio.run(create_sub_agent(agent_name))
    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_agent_server()
