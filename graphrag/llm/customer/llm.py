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
from .custom_llm_config import CustomConfiguration

class CustomLLM(BaseLLM[CompletionInput, CompletionOutput]):
    _configuration: CustomConfiguration
    def __init__(self,configuration: CustomConfiguration):
        self._configuration = configuration

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
        
        
        print("baidu_client response：", response.json()["result"])
        return response.json()["result"]
    
    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self._configuration.access_key, "client_secret": self._configuration.secret_key}
        return str(requests.post(url, params=params).json().get("access_token"))

    def qwen_agent_app(self, input):
        response = Application.call(app_id=self._configuration.access_key, prompt=input, api_key=self._configuration.secret_key)

        if response.status_code != HTTPStatus.OK:
            print('request_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message))
            return ""
        else:
            print('request_id=%s\n output=%s\n usage=%s\n' % (response.request_id, response.output, response.usage))
            return response.output["text"]
    
    def hg_client(self, input):
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {self._configuration.token}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        output = query({"inputs": input})
        print(output)
        if len(output) > 0:
            return output[0]['generated_text']
        return ""
    
    def perform_variable_replacements(self,
        input: str, history: list[dict], variables: dict | None
    ) -> str:
        """Perform variable replacements on the input string and in a chat log."""
        result = input

        def replace_all(input: str) -> str:
            result = input
            if variables:
                for entry in variables:
                    result = result.replace(f"{{{entry}}}", variables[entry])
            return result

        result = replace_all(result)
        for i in range(len(history)):
            entry = history[i]
            if entry.get("role") == "system":
                history[i]["content"] = replace_all(entry.get("content") or "")

        return result

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        
        # print("baidu_client args:**********", kwargs)
        variables = kwargs.get("variables")
        history = kwargs.get("history") or []
        format_input = self.perform_variable_replacements(input, history, variables)
        # print("baidu_client input:**********", format_input)
        return self.baidu_client(format_input)