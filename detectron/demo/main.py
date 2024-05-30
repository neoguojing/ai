import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image
import torch

import sys
sys.path.append("..")

from inference import ModelFactory
from face import FaceAlgo

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
                            components["face_input"] = gr.Image(type="pil",elem_id='face-input',label='输入')
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["face_image_output"] = gr.Gallery(elem_id='face_image_output',label='输出',columns=5,interactive=False)

            with gr.Row():
                with gr.Group():
                    components["face_output"] = gr.JSON(label="推理结果")

        # with gr.Tab("OCR"):  

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

    components["face_type"].change(
        ui_by_facetype, gradio('face_type'), params["face_type"]
    )

    components["face_submit_btn"].click(
        do_face_refernce,gradio('face_type','face_input'),gradio("face_output",'face_image_output')
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

def ui_by_facetype(face_type):
    if face_type == "人脸比对":
    else:
        components["face_image_output"].update()


def do_face_refernce(algo_type,input_image,input_image1):
# def do_refernce():
    print("input image",input_image)
    print(algo_type)

    if input_image is None:
        gr.Warning('请上传图片')
        return None
    algo_type = face_algo_map[algo_type]
    m = FaceAlgo()  # pragma: no cover
    out,faces = m.predict(pil_image=input_image,pil_image1=input_image1,algo_type=algo_type)
    # TODO 防止人脸过多的处理
    return out,faces

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()