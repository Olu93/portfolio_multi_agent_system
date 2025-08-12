#!/usr/bin/env python3
"""
Test script for the redesigned agents.
This script tests the basic functionality of the new agent structure.
"""

import asyncio
import logging
from typing import List

from .a2a_tool_client import A2AToolClient
from .research_agent import ResearchAgent
from .scraper_agent import ScraperAgent
from .orchestrator_agent import OrchestratorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_research_agent():
    """Test the research agent directly."""
    print("Testing Research Agent...")
    
    agent = ResearchAgent()
    try:
        result = await agent.process_query("Find information about Python programming")
        print(f"Research Agent Result: {result}")
        return True
    except Exception as e:
        print(f"Research Agent Error: {e}")
        return False


async def test_scraper_agent():
    """Test the scraper agent directly."""
    print("Testing Scraper Agent...")
    
    agent = ScraperAgent()
    try:
        result = await agent.process_query("https://example.com")
        print(f"Scraper Agent Result: {result}")
        return True
    except Exception as e:
        print(f"Scraper Agent Error: {e}")
        return False


async def test_orchestrator_agent():
    """Test the orchestrator agent directly."""
    print("Testing Orchestrator Agent...")
    
    # Create orchestrator with no remote agents for testing
    agent = OrchestratorAgent(remote_agent_addresses=[])
    try:
        # Test that it can be initialized (even without remote agents)
        await agent.ensure_initialized()
        print("Orchestrator Agent initialized successfully")
        
        # Test that it has the correct tools
        expected_tools = ['list_remote_agents', 'create_task']
        actual_tools = [tool.name for tool in agent.tools]
        print(f"Expected tools: {expected_tools}")
        print(f"Actual tools: {actual_tools}")
        
        # Test the common interface
        result = await agent.process_query("Find information about AI")
        print(f"Orchestrator Agent Result: {result}")
        
        if set(expected_tools) == set(actual_tools):
            print("Orchestrator Agent has correct tools")
            return True
        else:
            print("Orchestrator Agent has incorrect tools")
            return False
    except Exception as e:
        print(f"Orchestrator Agent Error: {e}")
        return False


async def test_a2a_client():
    """Test the A2A client."""
    print("Testing A2A Client...")
    
    client = A2AToolClient()
    client.add_remote_agent("http://localhost:5001")
    client.add_remote_agent("http://localhost:5002")
    
    try:
        agents = client.list_remote_agents()
        print(f"A2A Client Agents: {agents}")
        return True
    except Exception as e:
        print(f"A2A Client Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("Starting agent tests...\n")
    
    tests = [
        # ("Research Agent", test_research_agent),
        ("Scraper Agent", test_scraper_agent),
        # ("Orchestrator Agent", test_orchestrator_agent),
        # ("A2A Client", test_a2a_client),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"{test_name} test: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            print(f"{test_name} test: FAILED with exception {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("Test Summary:")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main()) 