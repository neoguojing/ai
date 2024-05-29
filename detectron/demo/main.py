import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image
import torch

import sys
sys.path.append("..")

from inference import ModelFactory

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
}

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                components["algo_type"] = gr.Dropdown(
                                ["目标检测","单阶段目标检测", "分类", "特征提取","语义分割","实例分割","关键点检测","全景分割"],value="全景分割",
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
                

        create_event_handlers()
    return demo


def create_event_handlers():
    params["algo_type"] = gr.State("全景分割")
    params["input_image"] = gr.State()
    

    components["image_input"].upload(
        lambda x: x, gradio('image_input'), params["input_image"]
    )
    
    components["algo_type"].change(
        lambda x: x, gradio('algo_type'), params["algo_type"]
    )

    components["submit_btn"].click(
        do_refernce,gradio('algo_type','image_input'),gradio("result_output",'image_output')
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
    if len(output_image) == 0:
        return output,None
    print("output image",output_image[0])
    return output,output_image[0]

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()