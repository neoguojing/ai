from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.base import Runnable

class LangchainApp:
    system_prompt: str = "You are a helpful assistant. Answer all questions to the best of your ability.Please use simple chinese as default language."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    llm: BaseChatModel
    runnable: Runnable
    with_message_history: RunnableWithMessageHistory

    db_path: str

    def __init__(self,model="qwen2",db_path="sqlite:///memory.db"):
        self.db_path = db_path
        self.llm =ChatOpenAI(
            # model="llama3.1",
            model=model,
            # model="phi3.5:3.8b-mini-instruct-fp16",
            # model="llama3.1-local",
            openai_api_key="121212",
            base_url="http://192.168.1.7:11434/v1/",
        )

        self.runnable = self.prompt | self.llm

        self.with_message_history = RunnableWithMessageHistory(
            self.runnable ,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

    def get_session_history(self,user_id: str, conversation_id: str):
        return SQLChatMessageHistory(f"{user_id}--{conversation_id}", self.db_path)
    
    def chat(self,input: str,language="chinese",user_id="",conversation_id="",stream=True):
        input_template = {"language": language, "input": input},
        config = {"configurable": {"user_id": user_id, "conversation_id": conversation_id}}

        response = None
        if stream:
            response = self.with_message_history.stream(input_template,config)
        else:
            response = self.with_message_history.invoke(input_template,config)
        

        for item in response:
            # 从每个 item 中提取 'content'
            content = item.content
            # 使用 yield 生成提取的 content
            yield content
