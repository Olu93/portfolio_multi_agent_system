#!/usr/bin/env python3
"""
Multi-turn A2A Client Example Script

This script demonstrates how to use the A2A client to have a multi-turn conversation
with the communication agent to send an email. The flow is:
1. Send initial request to send email with "hello world"
2. Agent asks for more information (recipient)
3. Provide recipient email address
4. Agent completes the email sending task
"""

import asyncio
import logging
import uuid
from typing import AsyncIterable

from a2a_servers.a2a_client import A2ASubAgentClient, A2AClientResponse
from a2a.types import TaskState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTurnEmailClient:
    """Client for multi-turn email conversations with A2A communication agent."""

    def __init__(self, agent_url: str = "http://localhost:10003"):
        self.agent_url = agent_url
        self.client = A2ASubAgentClient(default_timeout=60.0)
        self.context_id = str(uuid.uuid4())
        self.task_id = str(uuid.uuid4())

    async def send_message(self, message: str) -> AsyncIterable[A2AClientResponse]:
        """Send a message to the communication agent and yield responses."""
        logger.info(f"Sending message: {message}")
        logger.info(f"Agent URL: {self.agent_url}")
        async for response in self.client.async_send_message_streaming(
            agent_url=self.agent_url,
            message=message,
            context_id=self.context_id,
            task_id=self.task_id,
        ):
            yield response

    async def handle_response(self, response: A2AClientResponse) -> str | None:
        """Handle a response from the agent and return user input if needed."""
        if hasattr(response, "state"):
            state = response.state
        elif hasattr(response, "status"):
            state = response.status
        else:
            logger.warning(f"Unknown response type: {type(response)}")
            return None

        logger.info(f"Response state: {state}")

        if state == TaskState.input_required:
            # Agent needs more information
            if hasattr(response, "message") and response.message:
                logger.info(f"Agent message: {response.message}")
                return response.message
            elif hasattr(response, "content") and response.content:
                logger.info(f"Agent content: {response.content}")
                return response.content
        elif state == TaskState.completed:
            logger.info("Task completed successfully!")
            if hasattr(response, "message") and response.message:
                logger.info(f"Final message: {response.message}")
            elif hasattr(response, "content") and response.content:
                logger.info(f"Final content: {response.content}")
        elif state == TaskState.working:
            logger.info("Agent is working...")
            if hasattr(response, "message") and response.message:
                logger.info(f"Working message: {response.message}")
            elif hasattr(response, "content") and response.content:
                logger.info(f"Working content: {response.content}")
        elif state == TaskState.failed:
            logger.error("Task failed!")
            if hasattr(response, "message") and response.message:
                logger.error(f"Error message: {response.message}")
            elif hasattr(response, "content") and response.content:
                logger.error(f"Error content: {response.content}")

        return None


async def main():
    """Main function demonstrating multi-turn email conversation."""
    logger.info("Starting multi-turn email example...")

    # Initialize the client
    email_client = MultiTurnEmailClient()

    # Step 1: Send initial request
    logger.info("=== Step 1: Sending initial email request ===")
    initial_message = "Send an email with 'hello world'"

    user_input_needed = None
    async for response in email_client.send_message(initial_message):
        user_input_needed = await email_client.handle_response(response)
        if user_input_needed:
            break

    # Step 2: Provide recipient information if agent asks for it
    if user_input_needed:
        logger.info("=== Step 2: Agent asked for more information ===")
        logger.info(f"Agent said: {user_input_needed}")

        # Provide recipient email
        recipient_message = "I want to send it to o.hundogan@ohcm.nl"
        logger.info(f"Providing recipient: {recipient_message}")

        async for response in email_client.send_message(recipient_message):
            await email_client.handle_response(response)

    logger.info("=== Multi-turn conversation completed ===")


if __name__ == "__main__":
    asyncio.run(main())
