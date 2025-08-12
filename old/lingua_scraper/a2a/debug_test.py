#!/usr/bin/env python3
"""
Debug test to understand the streaming issue.
"""

import asyncio
import logging
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def debug_orchestrator_agent():
    """Debug the orchestrator agent to understand the streaming issue"""
    base_url = 'http://localhost:5000'
    
    async with httpx.AsyncClient() as httpx_client:
        try:
            # Get agent card
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            print("Agent Card:")
            print(f"  Name: {agent_card.name}")
            print(f"  Capabilities: {agent_card.capabilities}")
            print(f"  Streaming: {agent_card.capabilities.streaming}")
            print(f"  Push Notifications: {agent_card.capabilities.pushNotifications}")
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Simple test message
            simple_task = "Hello, can you help me?"
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': simple_task}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            print(f"\nSending request: {request}")
            print(f"Request params: {request.params}")
            
            # Try to send message
            response = await client.send_message(request)
            
            print(f"\nResponse received: {response}")
            if hasattr(response, 'root') and response.root:
                print(f"Response root: {response.root}")
                if hasattr(response.root, 'result'):
                    print(f"Response result: {response.root.result}")
                if hasattr(response.root, 'error'):
                    print(f"Response error: {response.root.error}")
            
        except Exception as e:
            logger.error(f"Error in debug test: {e}", exc_info=True)
            print(f"Exception type: {type(e)}")
            print(f"Exception message: {str(e)}")


if __name__ == "__main__":
    asyncio.run(debug_orchestrator_agent()) 