from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from pydantic import  Field
from langchain_core.embeddings import Embeddings

class Embedding(Embeddings):

    def __init__(self,**kwargs):
        self.model=AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5')
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
       
    @property
    def _llm_type(self) -> str:
        return "BAAI/bge-small-zh-v1.5"
    
    @property
    def model_name(self) -> str:
        return "embedding"
    
    def _call(
        self,
        prompt: Union[str,List[str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        batch_data = self.tokenizer(
                text=prompt,
                padding="longest",
                return_tensors="pt",
                max_length=1024,
                truncation=True,
        )

        attention_mask = batch_data["attention_mask"]
        # batch_data.to('cuda')
        model_output = self.model(**batch_data)
        # model_output = model_output.cpu()
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        vectors = vectors.detach().numpy()
        # 对每行的向量进行归一化
        vectors = normalize(vectors, norm="l2", axis=1)
        print("_call",vectors.shape) 
        return vectors

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
        return embeddings
    
    def embed_query(self, text) -> List[float]:
        # Embed a single query
        embedding = self._call(text)
        return embedding[0]
    

if __name__ == '__main__':
    sd = Embedding()
    v1 = sd.embed_query("他是一个人")
    v2 = sd.embed_query("她是一条狗")
    print(v1 @ v2.T)
