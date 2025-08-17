from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_community.storage import MongoDBStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from lingua_scraper.destination import Destination
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel
import uuid
from langchain_core.output_parsers import JsonOutputParser

class CoreAgent:
    def __init__(self):
        self.llm = None
        
        self.tools = []
        self.agent = None

    def set_llm(self, llm:BaseChatModel):
        self.llm = llm
        return self
    
    def set_checkpointer(self, checkpointer:BaseCheckpointSaver):
        self.checkpointer = checkpointer
        return self

    def set_mcp_client(self, mcp_client:MultiServerMCPClient):
        self.mcp_client = mcp_client
        return self
    
    def set_destination(self, destination:Destination):
        self.destination = destination
        return self

    def set_scrape_tools(self, tools:list[Tool]):
        self.tools += tools
        return self
    
    
    def set_store_tools(self, tools:list[Tool]):
        self.tools += tools
        return self
    
    def set_prompt(self, prompt:ChatPromptTemplate):
        self.prompt = prompt
        return self
    
    def set_schema(self, schema:BaseModel):
        
        parser = JsonOutputParser(pydantic_object=schema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You return a JSON object matching the schema."),
            ("system", "{format_instructions}"),
            ("human", "{input}"),
        ]).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.llm | parser

        chain_tool = chain.as_tool(
            name="structure_output", description=
            "Generate a structured output in JSON format." 
            
        )

        self.schema = schema
        self.chain_tool = chain_tool
        return self

    async def build(self):
        if self.llm is None:
            raise ValueError("LLM is not set")

        self.tools = await self.mcp_client.get_tools()
        self.tools.append(self.chain_tool)

        # self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.agent:CompiledGraph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            # prompt=self.prompt,
            checkpointer=self.checkpointer,
            # response_format=self.schema
        )
        return self


    async def ainvoke(self, input:str, *args, **kwargs):
        messages = {"messages":[
            {"role": "system", "content": 
             "You are a helpful scraper that can search the web, download files, and store data in a destination."
             "You will receive a URL which you will download"
             "You will use structure_output to extract the data the user is interested in."
             "Store the extracted data that you received as a result of the structure_output tool in a file storage."
             "Use the file name of the structured output to store the data in the file storage."
             },
            {"role": "user", "content": input}]}
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        return await self.agent.ainvoke(messages, config, *args, **kwargs)
    

    async def astream(self, input:str, *args, **kwargs):
        messages = {"messages":[
            {"role": "system", "content": "You are a helpful scraper that can search the web, download files, and store data in a destination."},
            {"role": "user", "content": input}]}
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        async for event in self.agent.astream(messages, stream_mode=["messages"], config=config):
            print(event)
            yield event
        yield None






