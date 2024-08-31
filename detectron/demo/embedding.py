from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from langchain_core.embeddings import Embeddings
import torch

class Embedding(Embeddings):

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
    

# if __name__ == '__main__':
#     sd = Embedding()
#     v1 = sd.embed_query("他是一个人")
#     v2 = sd.embed_query("他是一个好人")
#     v3 = sd.embed_documents(["她是一条狗","他是一个人"])
#     print(v1 @ v2.T)
