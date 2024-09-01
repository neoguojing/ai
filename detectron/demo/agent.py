
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
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit,SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults


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
            db = SQLDatabase.from_uri(db_path)
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            self.tools.extend(toolkit.get_tools())

        if retrievers is not None:
            retriever_tool = create_retriever_tool(
                retrievers,
                "retrieve_blog_posts",
                "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
            )
            self.tools.append(retriever_tool)


        search = DuckDuckGoSearchResults()
        self.tools.append(search)

        self.agent_executor = create_react_agent(
            self.llm, self.tools,checkpointer=self.memory,debug=True
        )

    def chat(self,input: str,language="chinese",user_id="",conversation_id="",stream=False):
        config = {"configurable": {"thread_id": "abc123"}}
        if stream:
            events = self.agent_executor.stream(
                {"messages": [HumanMessage(content=input)]},config,
                stream_mode="values",
            )
            for event in events:
                yield event["messages"][-1]
        else:
            events = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=input)]},config
            )

            yield events["messages"][-1]
            # event["messages"][-1].pretty_print()
    
    def __call__(self,input: str,user_id="",conversation_id=""):
        response = self.chat(input=input,user_id=user_id,conversation_id=conversation_id)
        for item in response:
            # 从每个 item 中提取 'content'
            content = item.content
            # 使用 yield 生成提取的 content
            yield content

# if __name__ == "__main__":
#     app = LangChainAgent()
#     stream_generator = app.chat("俄乌战争进展",stream=False)
#     # 遍历生成器
#     for response in stream_generator:
#         print("---",response.content)  # 或者进行其他操作，如解析、保存等