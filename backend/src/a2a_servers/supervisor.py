# supervisor.py — LLM-routed supervisor using LangGraph + A2A Registry discovery (a2a package)
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Annotated, Any, Dict, Optional, List, Callable, AsyncIterable, TypedDict

import click
import httpx
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
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
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
import uvicorn
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig

from a2a_servers.a2a_client import A2ASubAgentClient
from a2a_servers.config_loader import (
    load_agent_config,
    load_model_config,
    load_prompt_config,
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)

# Initialize memory saver for LangGraph
memory = InMemorySaver()
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)

    messages: Annotated[list, add_messages]   
# --- Pydantic Models ---------------------------------------------------------

class ChunkResponse(BaseModel):
    """Response structure for each chunk in the agent stream."""
    
    is_task_complete: bool = Field(
        default=False,
        description="Whether the task has been completed"
    )
    require_user_input: bool = Field(
        default=False,
        description="Whether the agent requires additional user input to proceed"
    )
    content: str = Field(
        description="The content/message for this chunk"
    )
    step_number: Optional[int] = Field(
        default=None,
        description="The step number in the execution sequence"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool being executed (if applicable)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for this chunk"
    )

# --- Config helper -----------------------------------------------------------

# --- Registry Client ---------------------------------------------------------
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://registry:8000")
REFRESH_SECS = int(os.getenv("DISCOVERY_REFRESH_SECS", "30"))


async def fetch_agents() -> List[AgentCard]:
    logger.info(f"Fetching agents from registry at {REGISTRY_URL}")
    async with httpx.AsyncClient(timeout=10) as s:
        r = await s.get(f"{REGISTRY_URL}/registry/agents")
        r.raise_for_status()
        agents = [AgentCard(**a) for a in r.json()]
        logger.info(f"Discovered {len(agents)} agents from registry")
        return agents


@tool
async def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    # TODO: Turn interrupt output into a json -> https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/#2-update-the-state-inside-the-tool
    human_response = interrupt(f"Human assistance requested: {query}")
    return human_response["data"]

# --- Tool factory from AgentCards -------------------------------------------


async def build_tools_from_registry(
    allow_urls: set, allow_caps: set
) -> List[StructuredTool]:
    logger.info(
        f"Building tools from registry with allow_urls={allow_urls}, allow_caps={allow_caps}"
    )

    def _safe_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()

    def _create_tool_for_card(card: AgentCard) -> StructuredTool:
        """Create a tool function for a specific agent card"""

        async def tool_impl(content: str, config: RunnableConfig):
            """
            content: str - the content of the message to send to the agent
            config: RunnableConfig - the config of the message

            The context is a dictionary of key-value pairs that are passed to the agent.
            It is used to pass information to the agent, such as the task_id, message_id, session_id, conversation_id, and other metadata.
            The task_id is a unique identifier for the task, the message_id is a unique identifier for the message, and the session_id is a unique identifier for the session.
            The task_id, message_id, and session_id are used to identify the task, message, and session in the A2A protocol.
            The conversation_id is a unique identifier for the conversation. The conversation_id is used to identify the conversation in the A2A protocol.
            """

            logger.debug(f"Tool {card.name} called with content: {content[:100]}...")
            cfg = config.get("configurable", {})
            context_id = cfg.get("thread_id")
            task_id = cfg.get("task_id")
            # For now, return a placeholder response
            # TODO: Implement proper A2A client communication
            client = A2ASubAgentClient()

            writer = get_stream_writer()
            response = client.async_send_message_streaming(card.url, content, context_id, task_id)
            async for chunk in response:
                logger.info(f"Tool {card.name} returned response: {chunk}")
                writer(chunk)
            return chunk
            # return response


        tool_name = _safe_name(card.name) or _safe_name(card.url)
        capabilities = list(card.capabilities.model_dump(mode="python").items())
        desc_caps = ", ".join([f"{k} = {v}" for k,v in sorted(capabilities, key=lambda x: x[0])])
        summary = (
            f"{card.description or 'A2A agent'}. Caps: {desc_caps or 'unspecified'}"
        )

        logger.debug(
            f"Creating tool '{tool_name}' for agent '{card.name}' at {card.url}"
        )
        return StructuredTool.from_function(
            coroutine=tool_impl,
            name=tool_name,
            description=summary,
        )

    cards = await fetch_agents()
    logger.info(f"Processing {len(cards)} agent cards")

    if allow_urls:
        original_count = len(cards)
        cards = [c for c in cards if c.url in allow_urls]
        logger.info(f"Filtered by allow_urls: {original_count} -> {len(cards)} agents")

    if allow_caps:
        original_count = len(cards)
        cards = [c for c in cards if allow_caps & set((c.capabilities or {}).keys())]
        logger.info(f"Filtered by allow_caps: {original_count} -> {len(cards)} agents")

    tools: List[StructuredTool] = []
    for card in cards:
        tools.append(_create_tool_for_card(card))

    logger.info(f"Successfully created {len(tools)} tools from registry")
    return tools



# --- Supervisor Agent Class -------------------------------------------------


class SupervisorAgent:
    """Supervisor agent that orchestrates other agents using LangGraph."""

    def __init__(self, name, model: BaseChatModel, tools, prompt):
        self.name = name
        self.tools = tools
        self.prompt = prompt
        self.model = model.bind_tools(self.tools)
        self.graph = self.build_graph()
        # self.graph = create_react_agent(
        #     name=name,
        #     model=self.model,
        #     tools=self.tools,
        #     checkpointer=memory,
        #     prompt=prompt,
        # )

    def build_graph(self):
     

        def model_execution(state: State) -> State:
            # TODO: Need to send all messages
            message:BaseMessage = self.model.invoke([state["messages"][-1]])
            assert len(message.tool_calls) <= 1
            return {"messages": [message]}


        
       



        graph_builder = StateGraph(State)
        graph_builder.add_node("model", model_execution)
        # for tool in self.tools:
        #     graph_builder.add_node(tool.name, ToolNode(tool))
        #     graph_builder.add_edge("model", tool.name)
        #     graph_builder.add_edge(tool.name, "model")

        graph_builder.add_node("tools", ToolNode(self.tools))
        # graph_builder.add_edge("model", "tools")
        # graph_builder.add_edge("tools", END)

        graph_builder.add_conditional_edges(
            "model",
            tools_condition,
        )

        graph_builder.add_edge("tools", "model")

        graph_builder.add_edge(START, "model")
        graph_builder.add_edge("model", END)
        graph = graph_builder.compile(checkpointer=memory, name=self.name)

        return graph

        



    async def stream(self, query, context_id: str, task_id: str) -> AsyncIterable[ChunkResponse]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id, 'task_id': task_id, "stream_id": uuid.uuid4()}}

        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield ChunkResponse(
                    is_task_complete=False,
                    require_user_input=False,
                    content='Processing the request...',
                    step_number=len(item.get('messages', [])),
                    tool_name=message.tool_calls[0].get('name') if message.tool_calls else None,
                    metadata={'message_type': 'tool_call'}
                )
            elif isinstance(message, ToolMessage):
                yield ChunkResponse(
                    is_task_complete=False,
                    require_user_input=False,
                    content='Executing the tool...',
                    step_number=len(item.get('messages', [])),
                    metadata={'message_type': 'tool_execution'}
                )

        yield self.get_agent_response(config)

    def get_agent_response(self, config) -> ChunkResponse:
        current_state = self.graph.get_state(config)
        messages = current_state.values.get('messages', [])
        
        if messages:
            final_message = messages[-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            return ChunkResponse(
                is_task_complete=True,
                require_user_input=False,
                content=response_text,
                step_number=len(messages),
                metadata={'message_type': 'final_response'}
            )

        return ChunkResponse(
            is_task_complete=False,
            require_user_input=True,
            content=(
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
            metadata={'message_type': 'error', 'error': 'no_messages'}
        )


# --- Base Agent Executor ---------------------------------------------------


class BaseAgentExecutor(AgentExecutor):
    """Base AgentExecutor for LangGraph/LangChain agents."""

    def __init__(
        self,
        agent: SupervisorAgent,
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

        updater = TaskUpdater(event_queue, context_id=task.context_id, task_id=task.id)
        try:
            async for item in self.agent.stream(query, context_id=task.context_id, task_id=task.id):
                # item is now a ChunkResponse model
                is_task_complete = item.is_task_complete
                require_user_input = item.require_user_input

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item.content,
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item.content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item.content))],
                        name=self.artifact_name,
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f"Error: {e!s}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.context_id, task.id),
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


# --- Agent Creation ---------------------------------------------------------


async def create_supervisor_agent(agent_name: str):
    """
    Create a supervisor agent based on the agents.yml file.
    """
    try:
        agent_config = load_agent_config(agent_name)
        model_config = load_model_config(agent_config.get("model", "default"))
        meta_prompt = agent_config.get("meta_prompt", "You are a helpful supervisor that can orchestrate multiple agents")
        prompt_config = load_prompt_config(agent_config.get("prompt_file", f"{agent_name}.txt"))

        # Build tools from registry
        allow_urls = set(agent_config.get("allow_urls", []) or [])
        allow_caps = set(agent_config.get("allow_caps", []) or [])
        tools = await build_tools_from_registry(allow_urls, allow_caps) + [human_assistance]

        # Initialize the model
        model = init_chat_model(
            model_config["name"], 
            **model_config["parameters"], 
            model_provider=model_config["provider"]
        )

        # Create the supervisor agent
        prompt = f"{meta_prompt}\n\n{prompt_config}"
        agent = SupervisorAgent(agent_name, model, tools, prompt)

        # Create agent card
        skills = [
            AgentSkill(
                id=skill.get("name", ""),
                name=skill.get("name", ""),
                description=skill.get("description", ""),
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
            )
            for skill in agent_config.get("skills", [])
        ]
        description = agent_config.get("description", "")
        capabilities = agent_config.get("capabilities", {})
        agent_card = AgentCard(
            version="1.0.0",
            name=agent_name,
            description=description,
            url=agent_config["agent_url"],
            capabilities=capabilities,
            skills=skills,
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=["application/json"],
        )

        # Create executor with custom parameters
        executor = BaseAgentExecutor(
            agent=agent,
            status_message="Processing request...",
            artifact_name="supervisor_response",
        )
        
        request_handler = DefaultRequestHandler(  
            agent_executor=executor,  
            task_store=InMemoryTaskStore()  
        ) 
        
        return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
        
    except Exception as e:
        logger.error(f"Error creating supervisor agent {agent_name}: {e}")
        raise e


# --- CLI entrypoint -----------------------------------------------------------


@click.command()
@click.option(
    "--agent-name",
    default="supervisor",
    help="Name of the supervisor agent config to load",
)
def run_supervisor(agent_name: str):
    """Run the Supervisor agent server."""
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 10020))

    async def run_with_background_tasks():
        app = await create_supervisor_agent(agent_name)
        fastapi_app = app.build()
        return fastapi_app
    
    # Run the async function and get the FastAPI app
    fastapi_app = asyncio.run(run_with_background_tasks())
    
    uvicorn.run(
        fastapi_app,
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    run_supervisor()
