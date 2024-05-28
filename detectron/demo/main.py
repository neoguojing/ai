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
    "algo_type": "全景分割",
    "input_image":None
}

def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [params[k] for k in keys]


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
            components["algo_type"] = gr.Dropdown(
                            ["目标检测","单阶段目标检测", "分类", "特征提取","语义分割","实例分割","关键点检测","全景分割"],value="全景分割",
                            label="算法类别",interactive=True
                        )
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
    components["image_input"].upload(
        update_input_image, components['image_input'], None).then(
        do_refernce,None,[components["result_output"],components["image_output"]]
        )
    
    components["algo_type"].change(
        update_algo_type, components['algo_type'],None
    )

def update_algo_type(input):
    params["algo_type"] = input
    print(params["algo_type"])
    return input

def update_input_image(input):
    params["input_image"] = input
    print(params["input_image"])
    return input

# def do_refernce(algo_type,input_image):
def do_refernce():
    print(params["input_image"])
    print(params["algo_type"])
    input_image = params["input_image"]
    algo_type = algo_map[params["algo_type"]]
    factory = ModelFactory()
    output,output_image = factory.predict(pil_image=input_image,task_type=algo_type)
    print(output)
    print(output_image)
    return output,output_image[0]

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()