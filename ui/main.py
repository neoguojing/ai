import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

import pdb


def create_ui():
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(elem_id='audio-container'):
                with gr.Group():
                    shared.gradio['audio'] = gr.Audio(sources="microphone",elem_id='audio-input',label='Speech input')
                    shared.gradio['speech_output'] = gr.Checkbox(value=shared.settings['speech_output'], label='Speech output', elem_id='speech-output')
                    shared.gradio['tone_record'] = gr.Audio(sources="microphone",elem_id='tone-record',label='Tone record')
                    shared.gradio['tone_upload'] = gr.Audio(sources="upload",elem_id='tone-upload',label='Tone upload')
                    
            # with gr.Row(elem_id='image-container'):
            #     shared.gradio['image'] = gr.Image(type="pil",elem_id='image-input',label='Image input')

            with gr.Row(elem_id='file-container'):
                shared.gradio['file'] = gr.File(file_count="multiple", file_types=["text", ".json", ".csv",".docx",".doc",".pdf"])
            
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(elem_id='chat-col'):
                    shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat'))

                    with gr.Row(elem_id="chat-input-row"):
                        with gr.Column(scale=1, elem_id='gr-hover-container'):
                            gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                        with gr.Column(scale=10, elem_id='chat-input-container'):
                            shared.gradio['textbox'] = gr.MultimodalTextbox(interactive=True, file_types=["image","audio","video"], placeholder="Enter message or upload file...", show_label=False)
                            # shared.gradio['textbox'] = gr.Textbox(label='', placeholder='Send a message', elem_id='chat-input', elem_classes=['add_scrollbar'])
                            shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='Show controls (Ctrl+S)', elem_id='show-controls')
                            shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')
                        
                        with gr.Column(scale=1, elem_id='generate-stop-container'):
                            with gr.Row():
                                shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop', visible=False)
                                # shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')

def create_event_handlers():
    # shared.gradio['textbox'].submit(
    #     ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
    #     lambda x: (x, None), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
    #     chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
    #     ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
    #     chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
    #     lambda: None, None, None, js=f'() => {{{ui.audio_notification_js}}}')
