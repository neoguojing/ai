from transformers import AutoModel, AutoTokenizer
from typing import Any, List, Mapping, Optional,Union
import torch
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .custom_llm_config import CustomConfiguration

class Embedding(BaseLLM[EmbeddingInput, EmbeddingOutput]):

    def __init__(self,configuration: CustomConfiguration):
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
    
    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        
        embedding = self._call(input)
        return [row.tolist() for row in embedding]
