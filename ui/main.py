import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

element = {}

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            element["algo_type"] = gr.Dropdown(
                            ["目标检测", "分类", "特征提取","语义分割","实例分割","关键点检测","全景分割"],value="全景分割",
                            label="算法类别"
                        )
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row(elem_id='audio-container'):
                    with gr.Group():
                        
                        element["image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入')
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Group():
                        element["image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False)

        with gr.Row():
            with gr.Group():
                element["result_output"] = gr.JSON(label="推理结果")
    
    return demo


def create_event_handlers():
    element["image_input"].upload(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, None), gradio('image'), gradio('Image input', 'image'), show_progress=False).then(
        chat.image_input_wrapper, gradio('Image input','interface_state'), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, js=f'() => {{{ui.audio_notification_js}}}')
    return


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()