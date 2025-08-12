import logging
import asyncio
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PUBLIC_AGENT_CARD_PATH = '/.well-known/agent.json'
EXTENDED_AGENT_CARD_PATH = '/agent/authenticatedExtendedCard'


async def test_orchestrator_complex_task(task):
    """Test the orchestrator agent with a complex task: finding 8 realtors in Rotterdam and extracting their contacts"""
    base_url = 'http://localhost:5000'
    logger = logging.getLogger(__name__)
    # Configure timeout for the httpx client
    timeout_config = httpx.Timeout(
        timeout=300.0,
        connect=10.0,
        read=300.0,
        write=10.0,
        pool=5.0
    )    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {base_url}{PUBLIC_AGENT_CARD_PATH}'
            )
            _public_card = await resolver.get_agent_card()
            logger.info('Successfully fetched public agent card:')
            logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
            final_agent_card_to_use = _public_card
            logger.info('\nUsing PUBLIC agent card for client initialization (default).')

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        # Initialize client and send message
        client = A2AClient(
            httpx_client=httpx_client, agent_card=final_agent_card_to_use
        )
        logger.info('A2AClient initialized for orchestrator agent.')

        # Complex task: Find 30 realtors in Rotterdam and extract their contacts
        complex_task = f"""
        {task}
        
        Requirements:
        1. Focus on individual real estate agencies, not platforms like Funda or Pararius
        2. Find real estate agents that are based in Rotterdam
        3. Extract the following contact information for each realtor:
           - Company/agency name
           - Contact person name (if available)
           - Phone number
           - Email address
           - Website URL
           - Physical address
           - Any additional contact details
        
        4. Ensure you get at least the amount of real estate agents specified in the task
        5. Provide a comprehensive list with all the extracted information
        
        This is a complex task that requires:
        - Research to find real estate agent websites
        - Scraping individual websites to extract contact details
        - Processing and organizing the information
        """

        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': complex_task}
                ],
                'messageId': uuid4().hex,
            },
        }
        
        logger.info("Sending complex task to orchestrator agent...")
        logger.info(f"Task: {complex_task[:200]}...")
        
        # Use non-streaming request
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        logger.info("Sending non-streaming request...")
        
        print("\n" + "="*80)
        print("ORCHESTRATOR AGENT - COMPLEX TASK EXECUTION")
        print("="*80)
        print("Task: Find realtors in Rotterdam and extract their contacts")
        print("="*80)
        
        try:
            response = await client.send_message(request)
            
            # Extract and display the response
            if hasattr(response, 'root') and response.root:
                if hasattr(response.root, 'result') and response.root.result:
                    result = response.root.result
                    if hasattr(result, 'artifacts') and result.artifacts:
                        for artifact in result.artifacts:
                            if hasattr(artifact, 'parts') and artifact.parts:
                                for part in artifact.parts:
                                    if hasattr(part, 'root') and part.root:
                                        if hasattr(part.root, 'text'):
                                            print(f"Result: {part.root.text}")
                                        elif hasattr(part.root, 'data'):
                                            print(f"Data: {part.root.data}")
                    else:
                        print(f"Response: {result}")
                elif hasattr(response.root, 'error') and response.root.error:
                    print(f"Error: {response.root.error}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            print(f"Error: {e}")
        
        print("="*80)
        print("Complex task completed!")
        print("="*80)


async def test_orchestrator_simple_task():
    """Test the orchestrator agent with a simple task for comparison"""
    base_url = 'http://localhost:5000'
    logger = logging.getLogger(__name__)
    
    # Configure timeout for the httpx client
    timeout_config = httpx.Timeout(
        timeout=120.0,
        connect=10.0,
        read=120.0,
        write=10.0,
        pool=5.0
    )
    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )

        try:
            _public_card = await resolver.get_agent_card()
            
            # Check agent capabilities
            logger.info(f"Agent capabilities: {_public_card.capabilities}")
            logger.info(f"Streaming supported: {_public_card.capabilities.streaming}")
            logger.info(f"Push notifications supported: {_public_card.capabilities.pushNotifications}")
            
            client = A2AClient(
                httpx_client=httpx_client, agent_card=_public_card
            )
            logger.info('A2AClient initialized for simple task test.')

            simple_task = "Find 5 real estate websites in Rotterdam"

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': simple_task}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            logger.info("Sending simple task to orchestrator agent...")
            
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            print("\n" + "="*60)
            print("ORCHESTRATOR AGENT - SIMPLE TASK TEST")
            print("="*60)
            print(f"Task: {simple_task}")
            print("="*60)
            
            try:
                response = await client.send_message(request)
                
                # Extract and display the response
                if hasattr(response, 'root') and response.root:
                    if hasattr(response.root, 'result') and response.root.result:
                        result = response.root.result
                        if hasattr(result, 'artifacts') and result.artifacts:
                            for artifact in result.artifacts:
                                if hasattr(artifact, 'parts') and artifact.parts:
                                    for part in artifact.parts:
                                        if hasattr(part, 'root') and part.root:
                                            if hasattr(part.root, 'text'):
                                                print(f"Result: {part.root.text}")
                                            elif hasattr(part.root, 'data'):
                                                print(f"Data: {part.root.data}")
                        else:
                            print(f"Response: {result}")
                    elif hasattr(response.root, 'error') and response.root.error:
                        print(f"Error: {response.root.error}")
                
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                print(f"Error: {e}")
                # Print more details about the error
                if "streaming" in str(e).lower():
                    print("This appears to be a streaming-related error.")
                    print("The agent is configured with streaming=False but the client is still trying to use streaming.")
            
            print("="*60)
            print("Simple task completed!")
            print("="*60)

        except Exception as e:
            logger.error(f'Error in simple task test: {e}', exc_info=True)


async def main() -> None:
    """Main function to test the orchestrator agent with complex realtor task"""
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting Orchestrator Agent Complex Task Test...")
    
    try:
        # Test simple task first (optional)
        # logger.info("\n=== Testing Simple Task (Optional) ===")
        # await test_orchestrator_simple_task()
        
        # Test complex task - main focus
        logger.info("\n=== Testing Complex Task: Find Realtors in Rotterdam. Extract their contact information. ===")
        await test_orchestrator_complex_task("Find 3 real estate agents (realtors) in Rotterdam, Netherlands and extract their contact information.")
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f'Error during testing: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    asyncio.run(main())