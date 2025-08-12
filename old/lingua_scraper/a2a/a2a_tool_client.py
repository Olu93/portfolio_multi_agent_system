import json
import uuid
from typing import Any

import httpx
import requests

from a2a.client import A2AClient
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest


class A2AToolClient:
    """A2A client for agent-to-agent communication."""

    def __init__(self, default_timeout: float = 120.0, httpx_client: httpx.AsyncClient | None = None):
        # Cache for agent metadata - also serves as the list of registered agents
        # None value indicates agent is registered but metadata not yet fetched
        self._agent_info_cache: dict[str, dict[str, Any] | None] = {}
        # Default timeout for requests (in seconds)
        self.default_timeout = default_timeout
        # External httpx client to use (if provided)
        self._httpx_client = httpx_client

    def add_remote_agent(self, agent_url: str):
        """Add agent to the list of available remote agents."""
        normalized_url = agent_url.rstrip('/')
        if normalized_url not in self._agent_info_cache:
            # Initialize with None to indicate metadata not yet fetched
            self._agent_info_cache[normalized_url] = None

    def list_remote_agents(self) -> list[dict[str, Any]]:
        """List available remote agents with caching."""
        if not self._agent_info_cache:
            return []

        remote_agents_info = []
        for remote_connection in self._agent_info_cache:
            # Use cached data if available
            if self._agent_info_cache[remote_connection] is not None:
                remote_agents_info.append(self._agent_info_cache[remote_connection])
            else:
                try:
                    # Fetch and cache agent info
                    agent_info = requests.get(f"{remote_connection}/.well-known/agent.json")
                    agent_data = agent_info.json()
                    self._agent_info_cache[remote_connection] = agent_data
                    remote_agents_info.append(agent_data)
                except Exception as e:
                    print(f"Failed to fetch agent info from {remote_connection}: {e}")

        return self._agent_info_cache

    async def create_task(self, agent_url: str, message: str) -> str:
        """Send a message following the official A2A SDK pattern."""
        # Normalize URL by removing trailing slashes
        normalized_url = agent_url.rstrip('/')
        
        # Configure timeout
        timeout_config = httpx.Timeout(
            timeout=self.default_timeout,
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0
        )

        # Use external client if provided, otherwise create a new one
        if self._httpx_client is not None:
            httpx_client = self._httpx_client
            # Update timeout if needed
            httpx_client.timeout = timeout_config
        else:
            # Create a new client with context manager
            async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
                return await self._create_task_with_client(httpx_client, normalized_url, message)
        
        # Use the external client directly
        return await self._create_task_with_client(httpx_client, normalized_url, message)

    async def _create_task_with_client(self, httpx_client: httpx.AsyncClient, normalized_url: str, message: str) -> str:
        """Internal method to create task using the provided httpx client."""
        # Check if we have cached agent card data
        if normalized_url in self._agent_info_cache and self._agent_info_cache[normalized_url] is not None:
            agent_card_data = self._agent_info_cache[normalized_url]
        else:
            # Fetch the agent card
            agent_card_response = await httpx_client.get(f"{normalized_url}/.well-known/agent.json")
            agent_card_data = agent_card_response.json()

        # Create AgentCard from data
        agent_card = AgentCard(**agent_card_data)

        # Create A2A client with the agent card
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=agent_card
        )

        # Build the message parameters following official structure
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': message}
                ],
                'messageId': uuid.uuid4().hex,
            }
        }

        # Create the request
        request = SendMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(**send_message_payload)
        )

        # Send the message with timeout configuration
        response = await client.send_message(request)

        # Extract text from response
        try:
            response_dict = response.model_dump(mode='json', exclude_none=True)
            if 'result' in response_dict and 'artifacts' in response_dict['result']:
                artifacts = response_dict['result']['artifacts']
                for artifact in artifacts:
                    if 'parts' in artifact:
                        # if the last part is a data part, return the data
                        if artifact['parts'][-1]['kind'] == 'data':
                            return artifact['parts'][-1]['data']
                        # if the last part is a text part, return the text
                        elif artifact['parts'][-1]['kind'] == 'text':
                            return "\n".join([f"Part: {d!s}" for d in artifact['parts'][:-5] if d]) 

            # If we couldn't extract text, return the full response as formatted JSON
            return json.dumps(response_dict, indent=2)

        except Exception as e:
            # Log the error and return string representation
            print(f"Error parsing response: {e}")
            return str(response)

    def remove_remote_agent(self, agent_url: str):
        """Remove an agent from the list of available remote agents."""
        normalized_url = agent_url.rstrip('/')
        if normalized_url in self._agent_info_cache:
            del self._agent_info_cache[normalized_url] 