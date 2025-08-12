import asyncio

from pydantic import BaseModel
from lingua_scraper.core_agent import CoreAgent
from lingua_scraper.mcp.duckduckgo import DuckDuckGoSearcher
from lingua_scraper.destination import KafkaDestination
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from lingua_scraper.schemas.realestate import RealEstate
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# async def download_file(url:str):
#     browser = await async_playwright().chromium.launch(headless=False)
#     page = await browser.new_page()
#     await page.goto(url)
#     await page.wait_for_load_state("networkidle")
#     await page.screenshot(path="screenshot.png")
#     await browser.close()


async def main():
    core_agent = CoreAgent()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can search the web and store data in a destination."),
        ("user", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")

    mcp_client = MultiServerMCPClient(
        {
            "duckduckgo": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            },
            # "playwright": {
            #     "url": "http://localhost:8001/mcp",
            #     "transport": "streamable_http"
            # },
            "file_storage": {
                "url": "http://localhost:8002/mcp",
                "transport": "streamable_http"
            }
        }
    )


    core_agent.set_llm(llm)
    core_agent.set_mcp_client(mcp_client)
    core_agent.set_prompt(prompt)
    core_agent.set_destination(KafkaDestination(None, None))
    core_agent.set_checkpointer(InMemorySaver())
    core_agent.set_schema(RealEstate)
    core_agent = await core_agent.build()

    result = await core_agent.ainvoke(input="Download information from https://www.tamparealtors.org/index.php?src=reso&srctype=detail&query=1&key=a76aa393cd02b317ad34776c04a732d7 then structure it and store it in a file storage.")
    print(result)

    # async for event in core_agent.astream(input="Download information from https://www.zillow.com/homedetails/8940-SW-20th-St-Miami-FL-33165/44180500_zpid/"):
    #     if event is not None:
    #         print(f"Event: {event}")


    return core_agent

if __name__ == "__main__":
    asyncio.run(main())





