
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


class LangChainAgent:
    tools = []
    memory = MemorySaver()
    def __init__(self,model="llama3.1-local",db_path="sqlite:///memory.db",
                 retrievers=None,base_url="http://192.168.1.7:11434/v1/"):
        
        self.db_path = db_path
        self.llm =ChatOpenAI(
            # model="llama3.1",
            model=model,
            # model="phi3.5:3.8b-mini-instruct-fp16",
            # model="llama3.1-local",
            openai_api_key="121212",
            base_url=base_url,
        )

        if db_path is not None:
            toolkit = SQLDatabaseToolkit(db=db_path, llm=self.llm)
            self.tools.extend(toolkit.get_tools())

        if retrievers is not None:
            retriever_tool = create_retriever_tool(
                retrievers,
                "retrieve_blog_posts",
                "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
            )
            self.tools.append(retriever_tool)

        self.agent_executor = create_react_agent(
            self.llm, self.tools,checkpointer=self.memory,debug=True
        )

    def chat(self,input: str,language="chinese",user_id="",conversation_id="",stream=True):
        if stream:
            events = self.agent_executor.stream(
                {"messages": [HumanMessage(content=input)]},
                stream_mode="values",
            )
        else:
            events = self.agent_executor.invoke(
                {"messages": [("user", input)]},
            )

        for event in events:
            print(event)
            event["messages"][-1].pretty_print()
                
