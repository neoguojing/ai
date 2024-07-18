import requests
import json
from http import HTTPStatus
from dashscope import Application
from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)
from typing_extensions import Unpack

class CustomLLM(BaseLLM[CompletionInput, CompletionOutput]):
    def __init__(self,ak,sk,token=None):
        self.ak = ak
        self.sk = sk
        self.token = token

    def baidu_client(self, input):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k?access_token=" + self.get_access_token()
        
        payload = json.dumps({
            "temperature": 0.95,
            "top_p": 0.7,
            "penalty_score": 1,
            "messages": [
                {
                    "role": "user",
                    "content": input
                }
            ],
            "system": ""
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        
        print("baidu_client", response.text)
        return response.json()["result"]
    
    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.ak, "client_secret": self.sk}
        return str(requests.post(url, params=params).json().get("access_token"))

    def qwen_agent_app(self, input):
        response = Application.call(app_id=self.ak, prompt=input, api_key=self.sk)

        if response.status_code != HTTPStatus.OK:
            print('request_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message))
            return ""
        else:
            print('request_id=%s\n output=%s\n usage=%s\n' % (response.request_id, response.output, response.usage))
            return response.output["text"]
    
    def hg_client(self, input):
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {self.token}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        output = query({"inputs": input})
        print(output)
        if len(output) > 0:
            return output[0]['generated_text']
        return ""
    
    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        return self.baidu_client(input)