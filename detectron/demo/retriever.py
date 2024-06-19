from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import os
from typing import Any,List,Dict
from embedding import Embedding


class KnowledgeBaseManager:
    def __init__(self, base_path="./knowledge_bases", embedding_dim=512, batch_size=16):
        self.base_path = base_path
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embeddings = Embedding()
        self.knowledge_bases: Dict[str, FAISS] = {}
        os.makedirs(self.base_path, exist_ok=True)

    def create_knowledge_base(self, name: str):
        index = faiss.IndexFlatL2(self.embedding_dim)
        kb = FAISS(self.embeddings, index, InMemoryDocstore(), {})
        self.knowledge_bases[name] = kb
        self.save_knowledge_base(name)
        print(f"Knowledge base '{name}' created.")

    def delete_knowledge_base(self, name: str):
        if name in self.knowledge_bases:
            del self.knowledge_bases[name]
            os.remove(os.path.join(self.base_path, f"{name}.faiss"))
            print(f"Knowledge base '{name}' deleted.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    def load_knowledge_base(self, name: str):
        kb_path = os.path.join(self.base_path, f"{name}.faiss")
        if os.path.exists(kb_path):
            self.knowledge_bases[name] = FAISS.load_local(self.base_path, self.embeddings, name, allow_dangerous_deserialization=True)
            print(f"Knowledge base '{name}' loaded.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    def save_knowledge_base(self, name: str):
        if name in self.knowledge_bases:
            self.knowledge_bases[name].save_local(self.base_path, name)
            print(f"Knowledge base '{name}' saved.")
        else:
            print(f"Knowledge base '{name}' does not exist.")

    def add_documents_to_kb(self, name: str, file_paths: List[str]):
        if name not in self.knowledge_bases:
            print(f"Knowledge base '{name}' does not exist.")
            self.create_knowledge_base(name)
        
        kb = self.knowledge_bases[name]
        documents = self.load_documents(file_paths)
        print(f"Loaded {len(documents)} documents.")
        
        pages = self.split_documents(documents)
        print(f"Split documents into {len(pages)} pages.")
        
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i:i+self.batch_size]
            kb.add_documents(batch)
        
        self.save_knowledge_base(name)

    def load_documents(self, file_paths: List[str]):
        documents = []
        for file_path in file_paths:
            loader = self.get_loader(file_path)
            documents.extend(loader.load())
        return documents

    def get_loader(self, file_path: str):
        if file_path.endswith('.txt'):
            return TextLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path)
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def retrieve_documents(self, names: List[str], query: str):
        results = []
        for name in names:
            if name not in self.knowledge_bases:
                print(f"Knowledge base '{name}' does not exist.")
                continue
            
            retriever = self.knowledge_bases[name].as_retriever(
                search_type="mmr",
                search_kwargs={"score_threshold": 0.5, "k": 1}
            )
            docs = retriever.get_relevant_documents(query)
            results.extend([{"name": name, "content": doc.page_content} for doc in docs])
        
        return results

class Retriever():
    index_path = "./"
    index_name = "default"
    batch_size = 16
    def __init__(self):
        self.embeddings = Embedding()
        if os.path.exists(self.index_path+self.index_name+".faiss"):
             print("load faiss from local index ")
             self.vector_store = FAISS.load_local(self.index_path, self.embeddings,self.index_name,allow_dangerous_deserialization=True)
             
        else:
            index = faiss.IndexFlatL2(512)
            self.vector_store = FAISS(self.embeddings,index,InMemoryDocstore(),{})
            self.vector_store.save_local(self.index_path,self.index_name)
        
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"score_threshold": 0.5,"k": 1}
            )

    def load_documents(self, file_paths):
        documents = []

        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        for file_path in file_paths:
            if file_path.endswith('.txt'):
                self.loader = TextLoader(file_path)
            elif file_path.endswith('.json'):
                self.loader = JSONLoader(file_path)
            elif file_path.endswith('.pdf'):
                self.loader = PyPDFLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            documents.extend(self.loader.load())
        return documents

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def build_vector_store(self, docs):
        self.vector_store.add_documents(docs)

    def retrieve_documents(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]
    
    def run(self, input: Any,**kwargs):
        if input is None or input == "":
            return 
        
        docs = self.load_documents(input)
        print("load_documents:",len(docs))
        pages = self.split_documents(docs)
        print("split_documents:",len(pages))
        groups = []
        if len(pages) > self.batch_size:
            groups = [docs[i:i+self.batch_size] for i in range(0, len(docs), self.batch_size)]
        else:
            groups = [pages]
        print("groups:",len(groups))
        for g in groups:
            self.build_vector_store(g)

        self.vector_store.save_local(self.index_path,self.index_name)

    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    

