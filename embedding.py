from transformers import AutoModel, AutoTokenizer
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from langchain_core.embeddings import Embeddings
import torch
from graphrag.query.llm.base import BaseTextEmbedding


class Embedding(Embeddings,BaseTextEmbedding):

    def __init__(self,**kwargs):
        self.model=AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5')
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
        self.model.eval()
       
    @property
    def _llm_type(self) -> str:
        return "BAAI/bge-small-zh-v1.5"
    
    @property
    def model_name(self) -> str:
        return "embedding"
    
    def _call(
        self,
        prompt: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        encoded_input = self.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            print(sentence_embeddings.shape)
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.numpy()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
    def embed_documents(self, texts) -> List[List[float]]:
        # Embed a list of documents
        embeddings = []
        print("embed_documents:",len(texts),type(texts))
        embedding = self._call(texts)
        for row in embedding:
            embeddings.append(row)
        # print("embed_documents: shape",embeddings.shape)
        return embeddings
    
    def embed_query(self, text) -> List[float]:
        # Embed a single query
        embedding = self._call([text])
        return embedding[0]
    

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a text string."""
        emb = self._call([text])

        if emb is not None and len(emb) > 0:
            list_data = emb[0].tolist()
            print("embed:",emb[0].shape,type(emb[0]),type(list_data))
            return list_data

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a text string asynchronously."""
        return self.embed(text)


# if __name__ == "__main__":
#     emb = Embedding()
#     # demo.launch(server_name="10.151.124.137")
#     emb.embed("你好！")