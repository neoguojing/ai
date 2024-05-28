import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

import sys
sys.path.append("..")

from inference import ModelFactory

components = {}

params = {
    "algo_type": "",
    "input_image":None
}

def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [params[k] for k in keys]



def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            components["algo_type"] = gr.Dropdown(
                            ["目标检测", "分类", "特征提取","语义分割","实例分割","关键点检测","全景分割"],value="全景分割",
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
        do_refernce,gradio("algo_type","input_image"),None
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

def do_refernce(algo_type,input_image):
    print(algo_type)
    print(input_image)
    # factory = ModelFactory()
    # factory.predict(pil_image=input_image,task_type=algo_type)



if __name__ == "__main__":
    demo = create_ui()
    demo.launch()