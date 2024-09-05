import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
import asyncio
import subprocess
import pandas as pd
import tiktoken
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch


from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)

# from llm import CustomLLM
# from embedding import Embedding
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2

RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
ROOT = "./knowledge_bases"
ENV_DIR = f"{ROOT}/.env"
SETTING_DIR = f"{ROOT}/settings.yaml"
class GraphRag:
    
    def __init__(self,base_name: str):
        # parquet files generated from indexing pipeline
        self.BASE_DIR = f"{ROOT}/{base_name}"
        self.INPUT_DIR = f"{self.BASE_DIR}/operation dulce"
        self.LANCEDB_URI = f"{self.INPUT_DIR}/lancedb"
        self.prepare_env()
        # 显式指定 .env 文件路径
        load_dotenv(dotenv_path=f"{self.BASE_DIR}/.env")
        self.api_key = os.environ["GRAPHRAG_API_KEY"]
        self.llm_model = os.environ["GRAPHRAG_LLM_MODEL"]
        self.embedding_model = os.environ["GRAPHRAG_EMBEDDING_MODEL"]
        self.api_base = os.environ["GRAPHRAG_API_BASE"]
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.llm_model,
            api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
            max_retries=20,
            api_base=self.api_base,
        )
        
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        # self.text_embedder = Embedding()
        self.text_embedder = OpenAIEmbedding(
            api_key=self.api_key,
            api_base=self.api_base,
            api_type=OpenaiApiType.OpenAI,
            model=self.embedding_model,
            deployment_name=self.embedding_model,
            max_retries=20,
        )

        self.load_graph()

    def load_graph(self):
        if not any(Path(self.INPUT_DIR).glob("*.parquet")):
            print(f"no .parquet files  in  {self.INPUT_DIR}")
            return 

         # read nodes table to get community and degree data
        self.entity_df = pd.read_parquet(f"{self.INPUT_DIR}/{ENTITY_TABLE}.parquet")
        self.entity_embedding_df = pd.read_parquet(f"{self.INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        self.entities = read_indexer_entities(self.entity_df, self.entity_embedding_df, COMMUNITY_LEVEL)

        # load description embeddings to an in-memory lancedb vectorstore
        # to connect to a remote db, specify url and port values.
        self.description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings",
        )
        self.description_embedding_store.connect(db_uri=self.LANCEDB_URI)
        self.entity_description_embeddings = store_entity_semantic_embeddings(
            entities=self.entities, vectorstore=self.description_embedding_store
        )
        print(f"Entity count: {len(self.entity_df)}")
        self.entity_df.head()

        self.relationship_df = pd.read_parquet(f"{self.INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        self.relationships = read_indexer_relationships(self.relationship_df)
        print(f"Relationship count: {len(self.relationship_df)}")
        self.relationship_df.head()

        self.covariate_df = pd.read_parquet(f"{self.INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        self.claims = read_indexer_covariates(self.covariate_df)
        print(f"Claim records: {len(self.claims)}")
        self.covariates = {"claims": self.claims}
        
        self.report_df = pd.read_parquet(f"{self.INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        self.reports = read_indexer_reports(self.report_df, self.entity_df, COMMUNITY_LEVEL)
        print(f"Report records: {len(self.report_df)}")
        self.report_df.head()

        text_unit_df = pd.read_parquet(f"{self.INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        self.text_units = read_indexer_text_units(text_unit_df)
        print(f"Text unit records: {len(self.text_unit_df)}")
        self.text_unit_df.head()

        self.init_global()
        self.init_local()

    def init_local(self):
        self.llm_params = {
            "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
            "temperature": 0.0,
        }
        
        self.local_context_builder = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates=self.covariates,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        self.local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        }

        self.local_search_engine = LocalSearch(
            llm=self.llm,
            context_builder=self.local_context_builder,
            token_encoder=self.token_encoder,
            llm_params=self.llm_params,
            context_builder_params=self.local_context_params,
            response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        self.question_generator = LocalQuestionGen(
            llm=self.llm,
            context_builder=self.context_builder,
            token_encoder=self.token_encoder,
            llm_params=self.llm_params,
            context_builder_params=self.local_context_params,
        )
        
    def init_global(self):
        self.context_builder = GlobalCommunityContext(
            community_reports=self.reports,
            entities=self.entities,  # default to None if you don't want to use community weights for ranking
            token_encoder=self.token_encoder,
        )

        self.context_builder_params = {
            "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        self.map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        self.reduce_llm_params = {
            "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
            "temperature": 0.0,
        }

        self.search_engine = GlobalSearch(
            llm=self.llm,
            context_builder=self.context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
            map_llm_params=self.map_llm_params,
            reduce_llm_params=self.reduce_llm_params,
            allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=self.context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

    async def g_search(self,input):
        result = await self.search_engine.asearch(
            input
        )
        result.context_data["reports"]
        print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
        return (result.response)
    
    async def search(self,input):
        result = await self.local_search_engine.asearch(input)
        result.context_data["entities"].head()

        result.context_data["relationships"].head()

        result.context_data["reports"].head()

        result.context_data["sources"].head()

        if "claims" in result.context_data:
            print(result.context_data["claims"].head())
        print(result.response)

    
    async def q_gene(self):
        question_history = [
            "Tell me about Agent Mercer",
            "What happens in Dulce military base?",
        ]
        candidate_questions = await self.question_generator.agenerate(
            question_history=question_history, context_data=None, question_count=5
        )

        return candidate_questions.response
    
    def prepare_env(self):
         # 创建目录，如果目录不存在则创建
        Path(self.BASE_DIR).mkdir(parents=True, exist_ok=True)

        command = ['python', '-m', 'graphrag.index','--init', '--root', self.BASE_DIR]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Command executed successfully: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")

        # 复制文件到目标目录
        shutil.copy(ENV_DIR, self.BASE_DIR)
        print(f"Copied {ENV_DIR} to {self.BASE_DIR}")
        shutil.copy(SETTING_DIR, self.BASE_DIR)
        print(f"Copied {SETTING_DIR} to {self.BASE_DIR}")

    async def do_index(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        # 定义要执行的命令
        command = ['python', '-m', 'graphrag.index', '--root', self.BASE_DIR]
        print(f"Command start: {command}")
        # 使用 asyncio.create_subprocess_exec() 异步执行命令
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # 等待命令完成，并获取输出和错误信息
        # stdout, stderr = await process.communicate()

        # 定义读取流的异步函数
        async def read_stream(stream, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                # 处理每行输出
                decoded_line = line.decode('utf-8').rstrip()
                logger.info(f"{stream_name}: {decoded_line}")

        # 创建任务来实时读取标准输出和标准错误
        stdout_task = asyncio.create_task(read_stream(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(read_stream(process.stderr, "stderr"))

        # 等待子进程结束并确保所有输出都已处理
        await process.wait()
        await stdout_task
        await stderr_task

        # 打印命令的输出
        if process.returncode == 0:
            print(f"Command executed successfully")
            self.load_graph()
        else:
            print(f"Command failed with error: { {process.returncode}}")

if __name__ == "__main__":
    rag = GraphRag("test")
    asyncio.run(rag.do_index())
    # ret = rag.text_embedder.embed("我爱北京天安门")
    # print(ret)