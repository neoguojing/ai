import requests
import json
from http import HTTPStatus
from dashscope import Application

ak = ""
sk = ""

def init_param(access_key,secret_key):
    global ak, sk
    ak = access_key
    sk = secret_key


def baidu_client(input,conversation_id=""):
    global ak, sk   
    if ak == "" or sk == "":
        return ""
    
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k?access_token=" + get_access_token()
    
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
    
    print("baidu_client",response.text)
    response = response.json()["result"].removeprefix(input)
    for item in response:
        yield item
    

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": ak, "client_secret": sk}
    return str(requests.post(url, params=params).json().get("access_token"))


def qwen_agent_app(input,conversation_id=""):
    global ak, sk   
    if ak == "" or sk == "":
        return ""
    response = Application.call(app_id=ak,
                                prompt=input,
                                api_key=sk,
                                )

    if response.status_code != HTTPStatus.OK:
        print('request_id=%s, code=%s, message=%s\n' % (response.request_id, response.status_code, response.message))
        return ""
    else:
        print('request_id=%s\n output=%s\n usage=%s\n' % (response.request_id, response.output, response.usage))
        response = response.output["text"].removeprefix(input)
        for item in response:
            yield item
    

def hg_client(input,conversation_id=""):
    global ak, sk   
    if sk == "":
        return ""
    import requests
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {sk}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": input,
    })
    print(output)
    if len(output) >0:
        response = output[0]['generated_text'].removeprefix(input)
        for item in response:
            yield item
    
    return ""

# if __name__ == "__main__":
#     stream_generator = openai_client("介绍下上海")
    
#     # 遍历生成器
#     for response in stream_generator:
#         # 假设 response 是一个可迭代的流对象
#         for part in response:
#             # 处理每一部分流
#             print(part)  # 或者进行其他操作，如解析、保存等