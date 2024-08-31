
import gradio as gr
from gradio_image_prompter import ImagePrompter
from retriever import knowledgeBase
import sys
import os
sys.path.append("..")
from event_handler import components,gradio
from event_handler import create_event_handlers


from pathlib import Path

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()

# 获取当前文件所在的目录
current_directory = current_file_path.parent

width = 713
height = int(width / 1.618)

def create_ui():
    with gr.Blocks() as demo:
        with gr.Tab("问答"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        components["llm_client"] =gr.Dropdown(["llama3.1", "Wenxin", "Tongyi","Huggingface"],value="llama3.1", label="请选择大语言模型")
                        components["ak"] = gr.Textbox(label="appid",visible=False)
                        components["sk"] = gr.Textbox(label="secret",visible=False)
                        components["llm_setting_btn"] =  gr.Button(value="设置",visible=False)
                        components["db_select"] = gr.CheckboxGroup(knowledgeBase.get_bases(),label="知识库", info="可选择1个或多个知识库")
                with gr.Column(scale=3):
                    with gr.Group():
                        components["chatbot"] = gr.Chatbot(
                                            [(None,"你好，有什么需要帮助的？")],
                                            elem_id="chatbot",
                                            bubble_full_width=False,
                                            height=600
                            )
                        components["chat_input"] = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

        with gr.Tab("知识库"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        components["db_view"] = gr.Dataframe(
                                                    headers=["列表"],
                                                    datatype=["str"],
                                                    row_count=2,
                                                    col_count=(1, "fixed"),
                                                    interactive=False
                        )
                with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column(scale=2):
                                components["db_name"] = gr.Textbox(label="名称", info="请输入库名称", lines=1, value="")
                            with gr.Column(scale=2):
                                components["db_submit_btn"] = gr.Button(value="提交")
                        components["file_upload"] = gr.File(elem_id='file_upload',file_count='multiple',label='文档上传', file_types=[".pdf", ".doc", '.docx', '.json', '.csv'])
            with gr.Row():
                with gr.Column(scale=2):
                    components["db_input"] = gr.Textbox(label="关键词", lines=1, value="")
                with gr.Column(scale=1):
                    components["db_test_select"] = gr.Dropdown(knowledgeBase.get_bases(),multiselect=True, label="知识库选择")
                with gr.Column(scale=1):
                    components["dbtest_submit_btn"] = gr.Button(value="检索")
            with gr.Row():
                with gr.Group():
                    components["db_search_result"] = gr.JSON(label="检索结果")

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
                            components["image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入',width=width,height=height)
                            components["base_examples"] = gr.Examples(
                                examples=[
                                    [os.path.join(current_directory,"examples/horse.png")],
                                    [os.path.join(current_directory,"examples/view.jpeg")],
                                    [os.path.join(current_directory,"examples/cat.jpg")],
                                    [os.path.join(current_directory,"examples/coco.jpg")],
                                    [os.path.join(current_directory,"examples/seg2.jpeg")],
                                ],
                                inputs=components["image_input"],
                                examples_per_page=6,
                                label="图片示例"
                            )
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False,width=width,height=height)

            with gr.Row():
                with gr.Group():
                    components["result_output"] = gr.JSON(label="推理结果")

        with gr.Tab("YOLO"):
            with gr.Row():
                with gr.Column(scale=2):
                    components["yolo_algo_type"] = gr.Dropdown(
                                    ["目标检测","分类","实例分割","姿态","OBB","跟踪","计数","体操","热力图","视界","速度","距离","队列"],value="目标检测",
                                    label="算法类别",interactive=True
                            )
                with gr.Column(scale=2):
                    components["yolo_submit_btn"] = gr.Button(value="解析")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id='audio-container'):
                        with gr.Column(scale=2):
                            components["yolo_dist_a"] = gr.Number(value=0,label="A",visible=False)
                        with gr.Column(scale=2):
                            components["yolo_dist_b"] = gr.Number(value=0,label="B",visible=False)
                        with gr.Group():
                            components["yolo_image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入',width=width,height=height)
                            components["yolo_video_input"] = gr.Video(label='输入',visible=False,interactive=True,width=width,height=height)
                            components["yolo_image_examples"] = gr.Examples(
                                examples=[
                                    [os.path.join(current_directory,"examples/horse.png")],
                                    [os.path.join(current_directory,"examples/view.jpeg")],
                                    [os.path.join(current_directory,"examples/cat.jpg")],
                                    [os.path.join(current_directory,"examples/coco.jpg")],
                                    [os.path.join(current_directory,"examples/seg2.jpeg")],
                                ],
                                inputs=[components["yolo_image_input"]],
                                examples_per_page=6,
                                label="图片示例"
                            )
                            components["yolo_video_examples"] = gr.Examples(
                                examples=[
                                    [os.path.join(current_directory,"examples/gym.mp4")],
                                    [os.path.join(current_directory,"examples/traffic.mp4")],
                                ],
                                inputs=[components["yolo_video_input"]],
                                examples_per_page=4,
                                label="视频示例"
                            )
        
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["yolo_image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False,width=width,height=height)
                            components["yolo_video_output"] = gr.PlayableVideo(label='输出',visible=False,width=width,height=height)
                            components["yolo_crop_output"] = gr.Gallery(type="pil",elem_id='crop-output',label='裁剪',interactive=False,visible=True)

            with gr.Row():
                with gr.Group():
                    components["yolo_result_output"] = gr.JSON(label="推理结果")

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
                            components["face_input"] = gr.Image(elem_id='face-input',label='输入',type="pil",width=width,height=height)
                            components["face_input2"] = gr.Image(elem_id='face_input2',label='输入-比对',type="pil",width=width,height=height)
                            components["face_examples"] = gr.Examples(
                                examples=[
                                    [os.path.join(current_directory,"examples/face1.jpeg")],
                                    [os.path.join(current_directory,"examples/face2.jpeg")],
                                ],
                                inputs=[components["face_input"]],
                                examples_per_page=6,
                                label="图片示例"
                            )
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["face_image_output"] = gr.Gallery(elem_id='face_image_output',label='输出',columns=2,interactive=False)

            with gr.Row():
                with gr.Group():
                    components["face_output"] = gr.JSON(label="推理结果")
        with gr.Tab("OCR"):
            with gr.Row():
                with gr.Column(scale=2):
                        components["ocr_type"] = gr.Dropdown(
                                        ["OCR","Easy"],value="Easy",
                                        label="算法类别",interactive=True
                                )
                with gr.Column(scale=2):
                    components["submit_ocr_btn"] = gr.Button(value="解析")

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row(elem_id=''):
                        with gr.Group():
                            components["ocr_input"] = gr.Image(elem_id='ocr-input',label='输入',type="pil",width=width,height=height)
                            components["ocr_examples"] = gr.Examples(
                                examples=[
                                    [os.path.join(current_directory,"examples/json.png")],
                                    [os.path.join(current_directory,"examples/text.jpeg")],
                                ],
                                inputs=[components["ocr_input"]],
                                examples_per_page=6,
                                label="图片示例"
                            )
                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group():
                            components["ocr_output"] = gr.Image(elem_id='ocr_output',label='输出',interactive=False,type="pil",width=width,height=height)
            with gr.Row():
                with gr.Group():
                    components["ocr_json_output"] = gr.JSON(label="推理结果")
        with gr.Tab("SAM everything"): 
            with gr.Row():
                with gr.Column(scale=2):
                    components["sam_version"] = gr.Dropdown(
                                    ["1","2"],value="1",
                                    label="sam版本",interactive=True
                            )
                with gr.Column(scale=2):
                    components["sam_submit_btn"] = gr.Button(value="解析")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        components["sam_input"] = ImagePrompter(elem_id='sam-input',label='输入',type="pil",width=width,height=height)
                        components["sam_video_input"] = gr.Video(label='视频输入',visible=False,interactive=True,width=width,height=height)
                        components["sam_image_examples"] = gr.Examples(
                            examples=[
                                [{'image': os.path.join(current_directory,"examples/horse.png"), 'points': []}],
                                [{'image': os.path.join(current_directory,"examples/view.jpeg"), 'points': []}],
                                [{'image': os.path.join(current_directory,"examples/coco.jpg"), 'points': []}],
                                [{'image': os.path.join(current_directory,"examples/seg2.jpeg"), 'points': []}],
                            ],
                            inputs=[components["sam_input"]],
                            examples_per_page=6,
                            label="图片示例"
                        )
                        components["sam_video_examples"] = gr.Examples(
                            examples=[
                                [os.path.join(current_directory,"examples/traffic.mp4")],
                                [os.path.join(current_directory,"examples/gym.mp4")],
                            ],
                            inputs=[components["sam_video_input"]],
                            examples_per_page=4,
                            label="视频示例"
                        )
                with gr.Column(scale=2):
                    with gr.Group():
                        components["sam_output"] = gr.Gallery(elem_id='sam_output',label='输出',columns=1,interactive=False)
                        components["sam_video_output"] = gr.PlayableVideo(label='输出',visible=False,width=width,height=height)


        create_event_handlers()
        demo.load(init,None,gradio("db_view","db_select","db_test_select"))
    return demo



def init():
    db_list = knowledgeBase.get_bases()
    db_df_list = knowledgeBase.get_df_bases()
    return db_df_list,gr.CheckboxGroup(db_list,label="知识库", info="可选择1个或多个知识库"),gr.Dropdown(db_list,multiselect=True, label="知识库选择")


