import json
from functools import partial
from pathlib import Path
import time
import gradio as gr
from PIL import Image
import torch
import numpy as np
from gradio_image_prompter import ImagePrompter
import sys
sys.path.append("..")

from inference import ModelFactory
from face import FaceAlgo
from sam_everything import SamAnything


components = {}

params = {
    "algo_type": None,
    "input_image":None
}


def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [components[k] for k in keys]


algo_map = {
    "目标检测":"detect",
    "单阶段目标检测":"onestep_detect",
    "分类":"classification",
    "特征提取":"feature",
    "语义分割":"semantic",
    "实例分割":"instance",
    "关键点检测":"keypoint",
    "全景分割":"panoptic",
    "YOLO":"yolo",
}

face_algo_map = {
    "人脸检测":"detect",
    "人脸识别":"recognize",
    "人脸比对":"compare",
    "特征提取":"feature",
    "属性分析":"attr",
}

def create_ui():
    with gr.Blocks() as demo:
        with gr.Tab("基础算法"):
            with gr.Row():
                with gr.Column(scale=2):
                    components["algo_type"] = gr.Dropdown(
                                    ["目标检测","单阶段目标检测", "分类", "特征提取","语义分割","实例分割","关键点检测","全景分割","YOLO"],value="全景分割",
                                    label="算法类别",interactive=True
                            )
                with gr.Column(scale=2):
                    components["submit_btn"] = gr.Button(value="解析")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id='audio-container'):
                        with gr.Group():
                            components["image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入')
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False)

            with gr.Row():
                with gr.Group():
                    components["result_output"] = gr.JSON(label="推理结果")

        with gr.Tab("人脸算法"):   
            with gr.Row():
                with gr.Column(scale=2):
                    components["face_type"] = gr.Dropdown(
                                    ["人脸检测","人脸识别", "人脸比对", "特征提取","属性分析"],value="人脸检测",
                                    label="算法类别",interactive=True
                            )
                with gr.Column(scale=2):
                    components["face_submit_btn"] = gr.Button(value="解析")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id=''):
                        with gr.Group():
                            components["face_input"] = gr.Gallery(elem_id='face-input',label='输入',columns=2,type="pil")
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["face_image_output"] = gr.Gallery(elem_id='face_image_output',label='输出',columns=2,interactive=False)

            with gr.Row():
                with gr.Group():
                    components["face_output"] = gr.JSON(label="推理结果")
        with gr.Tab("SAM everything"): 
            with gr.Row():
                with gr.Column(scale=2):
                    components["sam_submit_btn"] = gr.Button(value="解析")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        # components["sam_input"] = gr.ImageEditor(elem_id='sam-input',label='输入',type="pil")
                        components["sam_input"] = ImagePrompter(elem_id='sam-input',label='输入',type="pil")
                with gr.Column(scale=2):
                    with gr.Group():
                        components["sam_output"] = gr.Gallery(elem_id='sam_output',label='输出',columns=1,interactive=False)

        with gr.Tab("RAG"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        components["file_upload"] = gr.File(elem_id='doc-input',label='文档上传',file_types=[".pdf",".doc",'.docx','.json','.csv'])
                        components["db_view"] = gr.Dataframe(
                                                    headers=["name", "id"],
                                                    datatype=["str", "str"],
                                                    row_count=2,
                                                    col_count=(2, "fixed"),
                                                    interactive=False
                                                )
                with gr.Column(scale=3):
                    with gr.Group():
                        components["chatbot"] = gr.Chatbot(
                                            [(None,"What can I help you?")],
                                            elem_id="chatbot",
                                            bubble_full_width=False,
                                            height=800
                        )
                        components["chat_input"] = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)


        create_event_handlers()
    return demo


def create_event_handlers():
    params["algo_type"] = gr.State("全景分割")
    params["input_image"] = gr.State()
    params["face_type"] = gr.State("人脸检测")
    

    components["image_input"].upload(
        lambda x: x, gradio('image_input'), params["input_image"]
    )
    
    components["algo_type"].change(
        lambda x: x, gradio('algo_type'), params["algo_type"]
    )

    components["submit_btn"].click(
        do_refernce,gradio('algo_type','image_input'),gradio("result_output",'image_output')
    )

    components["face_type"].change(
        ui_by_facetype, gradio('face_type'), params["face_type"]
    )

    components["face_submit_btn"].click(
        do_face_refernce,gradio('face_type','face_input'),gradio("face_output",'face_image_output')
    )

    # components["sam_input"].upload(
    #     do_sam_everything,gradio('sam_input'),gradio("sam_output")
    # )

    # components["sam_input"].change(
    #     do_sam_everything,gradio('sam_input'),gradio("sam_output")
    # )

    components["sam_submit_btn"].click(
        do_sam_everything,gradio('sam_input'),gradio("sam_output")
    )

    components["chat_input"].submit(
        do_llm_request, gradio("chatbot", "chat_input"), gradio("chatbot", "chat_input")
    ).then(
        do_llm_response, gradio("chatbot"), gradio("chatbot"), api_name="bot_response"
    ).then(
        lambda: gr.MultimodalTextbox(interactive=True), None, gradio('chat_input')
    )

    components["chatbot"].like(print_like_dislike, None, None)

    components['file_upload'].upload(
        file_handler, gradio('file_upload'),  gradio('db_view'), show_progress=False
    )

def do_refernce(algo_type,input_image):
# def do_refernce():
    print("input image",input_image)
    print(algo_type)

    if input_image is None:
        gr.Warning('请上传图片')
        return None
    algo_type = algo_map[algo_type]
    factory = ModelFactory()
    output,output_image = factory.predict(pil_image=input_image,task_type=algo_type)
    if output_image is None or len(output_image) == 0:
        return output,None
    print("output image",output_image[0])
    return output,output_image[0]

def ui_by_facetype(face_type):
    print("ui_by_facetype",face_type)


def do_face_refernce(algo_type,input_images):
    print("input image",input_images)
    print(algo_type)

    if input_images is None:
        gr.Warning('请上传图片')
        return None,None
    
    input1 = input_images[0][0]
    input2 = None
    algo_type = face_algo_map[algo_type]
    if algo_type == "compare" and len(input_images) >=2:
        input2 = input_images[1][0]
    elif algo_type == "compare" and len(input_images) < 2:
        gr.Warning('请上传两张图片')    
        return None,None

    m = FaceAlgo()  # pragma: no cover
    out,faces = m.predict(pil_image=input1,pil_image1=input2,algo_type=algo_type)

    return out,faces

def do_sam_everything(im):
    sam_anything = SamAnything()
    print(im)
    image_pil = im['image']
    points = im['points']
    images = None
    if points is None or len(points) == 0:
        _, images = sam_anything.seg_all(image_pil)
    else:
        point_coords = []
        box = None
        for item in points:
            if item[2] == 1:
                # 点类型
                point_coords.append([item[0],item[1]])
            else:
                # box类型,只使用最后一个box
                box = [item[0],item[1],item[3],item[4]]
                box = np.array(box)
        
        if box is not None:
            _, images = sam_anything.seg_with_promp(image_pil,box=box)
        else:
            coords = np.array(point_coords)
            print("point_coords:",coords.shape)
            _, images = sam_anything.seg_with_promp(image_pil,point_coords=coords)
        
    return images

def point_to_mask(pil_image):
    # 遍历每个像素
    width, height = pil_image.size
    print(width, height)
    points_list = []
    for x in range(width):
        for y in range(height):
            # 获取像素的RGB值
            pix_val = pil_image.getpixel((x, y))
            if pix_val[0] != 0 and pix_val[1] != 0 and pix_val[2] != 0:
                points_list.append((x, y))
    points_array = np.array(points_list)
    points_array_reshaped = points_array.reshape(-1, 2)
    return points_array_reshaped

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def do_llm_request(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def do_llm_response(history):
    response = llm(history[-1][0])
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

def llm(input):
    import requests
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": input,
    })
    print(output)
    if len(output) >0:
        return output[0]['generated_text']
    return ""

def file_handler(file_objs):
    import shutil
    import os
    from retriever import Retriever
    print("file_obj:",type(file_objs))
    task = Retriever()

    os.makedirs(os.path.dirname("./files/input/"), exist_ok=True)
    for idx, file in enumerate(file_objs):
        shutil.move(file.name,"./files/input/")
        file_path = "./files/input/" +  os.path.basename(file.name)
        task.run(file_path)

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()