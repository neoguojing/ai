import gradio as gr
from gradio_image_prompter import ImagePrompter
import time
import sys
import numpy as np

sys.path.append("..")
from detectron2.config import get_cfg
from inference import ModelFactory
from face import FaceAlgo
from sam_everything import SamAnything
from retriever import knowledgeBase
from yolo_predictor import YOLOPredictor
from ocr import do_ocr
params = {
    "algo_type": None,
    "input_image":None
}



algo_map = {
    "目标检测":"detect",
    "单阶段目标检测":"onestep_detect",
    "分类":"classification",
    "特征提取":"feature",
    "语义分割":"semantic",
    "实例分割":"instance",
    "关键点检测":"keypoint",
    "全景分割":"panoptic",
    "姿态":"pose",
    "OBB":"OBB",
    "跟踪":"instance",
    "计数":"detect",
    "体操":"pose",
    "热力图":"detect",
    "视界":"detect",
    "速度":"detect",
    "队列":"detect",
    "距离":"detect",
}

face_algo_map = {
    "人脸检测":"detect",
    "人脸识别":"recognize",
    "人脸比对":"compare",
    "特征提取":"feature",
    "属性分析":"attr",
}


components = {}
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [components[k] for k in keys]

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

    components["yolo_submit_btn"].click(
        do_yolo_refernce,
        gradio('yolo_algo_type','yolo_image_input','yolo_video_input'),
        gradio("yolo_result_output",'yolo_image_output','yolo_video_output',"yolo_crop_output")
    )

    components["yolo_algo_type"].change(
        do_yolo_algo_type_chage, 
        gradio('yolo_algo_type'),
        gradio('yolo_image_input','yolo_image_output','yolo_video_input','yolo_video_output','yolo_dist_a','yolo_dist_b',"yolo_crop_output")
    )

    components["yolo_dist_b"].change(
        do_yolo_dist_change,
        gradio('yolo_dist_a','yolo_dist_b'),
        None,
    )


    components["face_type"].change(
        ui_by_facetype, gradio('face_type'), params["face_type"]
    )

    components["face_submit_btn"].click(
        do_face_refernce,gradio('face_type','face_input'),gradio("face_output",'face_image_output')
    )

    components["sam_version"].change(
        do_sam_version_chage, 
        gradio('sam_version'),
        gradio('sam_input','sam_video_input','sam_output','sam_video_output')
    )

    components["sam_submit_btn"].click(
        do_sam_everything,gradio('sam_input','sam_video_input','sam_version'),gradio("sam_output","sam_video_output")
    )

    components["submit_ocr_btn"].click(
        do_ocr,gradio('ocr_type','ocr_input'),gradio("ocr_output","ocr_json_output")
    )

    components["db_submit_btn"].click(
        file_handler,gradio('file_upload','db_name'),gradio("db_view",'db_select','db_test_select')
    )

    components["chat_input"].submit(
        do_llm_request, gradio("chatbot", "chat_input"), gradio("chatbot", "chat_input")
    ).then(
        do_llm_response, gradio("chatbot","db_select"), gradio("chatbot"), api_name="bot_response"
    ).then(
        lambda: gr.MultimodalTextbox(interactive=True), None, gradio('chat_input')
    )

    # components["chatbot"].like(print_like_dislike, None, None)

    components['dbtest_submit_btn'].click(
        do_search, gradio('db_test_select','db_input'), gradio('db_search_result')
    )

    # components['db_select'].select(
    #     db_select_handler, gradio('db_select'), None, show_progress=False
    # )

    components['llm_setting_btn'].click(
        llm, gradio('ak','sk','llm_client'), None
    )

def do_refernce(algo_type,input_image):
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

def do_yolo_dist_change(a,b):
     if YOLOPredictor.get_instance("detect") is not None:
         YOLOPredictor.get_instance("detect").set_dis_obj(a,b)

def do_yolo_algo_type_chage(value):
    print("do_yolo_algo_type_chage:",value)
    if value.strip() == "跟踪" or value.strip() == "计数" or value.strip() == "体操" or "热力图" == value.strip() or "视界" == value.strip() or "速度" == value.strip() or "距离" == value.strip() or "队列" == value.strip():
        components["yolo_video_input"] = gr.Video(label='输入',visible=True,interactive=True)
        components["yolo_video_output"] = gr.PlayableVideo(label='输出',visible=True)
        components["yolo_image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入',visible=False)
        components["yolo_image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False,visible=True)
        components["yolo_crop_output"] = gr.Gallery(type="pil",elem_id='crop-output',label='裁剪',interactive=False,visible=False)
        if "距离" == value.strip():
            components["yolo_dist_a"] = gr.Number(value=0,label="A",visible=True,info="跟踪id-A")
            components["yolo_dist_b"] = gr.Number(value=0,label="B",visible=True,info="跟踪id-B")
    else:
        components["yolo_image_input"] = gr.Image(type="pil",elem_id='image-input',label='输入',visible=True)
        components["yolo_image_output"] = gr.Image(type="pil",elem_id='image-output',label='输出',interactive=False,visible=True)
        components["yolo_video_input"] = gr.Video(label='输入',visible=False,interactive=True)
        components["yolo_video_output"] = gr.PlayableVideo(label='输出',visible=False)
        components["yolo_dist_a"] = gr.Number(value=0,label="A",visible=False)
        components["yolo_dist_b"] = gr.Number(value=0,label="B",visible=False)
        components["yolo_crop_output"] = gr.Gallery(type="pil",elem_id='crop-output',label='裁剪',interactive=False,visible=True)

    return components["yolo_image_input"],components["yolo_image_output"],components["yolo_video_input"],components["yolo_video_output"], components["yolo_dist_a"], components["yolo_dist_b"],components["yolo_crop_output"]

def do_yolo_refernce(algo_type,input_image,input_video):
    print("input image",input_image)
    print(algo_type)

    if input_image is None and input_video is None:
        gr.Warning('请上传图片或视频')
        return None
    cfg = get_cfg()
    cfg.TASK_TYPE = algo_map[algo_type]
    yolo = YOLOPredictor(cfg)

    if algo_type.strip() == "跟踪":
        yield from yolo.track_with_seg(input_video)
    elif algo_type.strip() == "计数":
        yield from yolo.counting(input_video)
    elif algo_type.strip() == "体操":
        yield from yolo.gym_monitor(input_video)
    elif algo_type.strip() == "热力图":
        yield from yolo.heatmap(input_video)
    elif algo_type.strip() == "视界":
        yield from yolo.vision_eye(input_video)
    elif algo_type.strip() == "速度":
        yield from yolo.speed(input_video)
    elif algo_type.strip() == "距离":
        yield from yolo.distance(input_video)
    elif algo_type.strip() == "队列":
        yield from yolo.queue_manager(input_video)
    else:
        output,output_image = yolo(input_image)
        if output_image is None or len(output_image) == 0:
            yield output,None,None,None
        print("output image:",len(output_image))
        yield output,output_image[0],None,output_image[1:]

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

def do_sam_version_chage(value):
    print("do_sam_version_chage:",value)
    if value.strip() == "1":
        components["sam_input"] = ImagePrompter(elem_id='sam-input',label='输入',type="pil")
        components["sam_video_input"] = gr.Video(label='视频输入',visible=False,interactive=True)
        components["sam_output"] = gr.Gallery(elem_id='sam_output',label='输出',columns=1,interactive=False)
        components["sam_video_output"] = gr.PlayableVideo(label='输出',visible=False)
    elif value.strip() == "2":
        components["sam_input"] = ImagePrompter(elem_id='sam-input',label='输入',type="pil")
        components["sam_video_input"] = gr.Video(label='视频输入',visible=True,interactive=True)
        components["sam_output"] = gr.Gallery(elem_id='sam_output',label='输出',columns=1,interactive=False)
        components["sam_video_output"] = gr.PlayableVideo(label='输出',visible=True)

    return components["sam_input"],components["sam_video_input"],components["sam_output"],components["sam_video_output"]


def do_sam_everything(im,video,version):
    print(im)
    sam_anything = None
    if version == "1":
        sam_anything = SamAnything()
    elif version == "2":
        sam_anything = None
    
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
    print("do_llm_request:",history,message)
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def do_llm_response(history,selected_dbs):
    print("do_llm_response:",history,selected_dbs)
    user_input = history[-1][0]
    prompt = ""
    quote = ""
    if len(selected_dbs) > 0:
        knowledge = knowledgeBase.retrieve_documents(selected_dbs,user_input)
        print("do_llm_response context:",knowledge)
        prompt = f'''
背景1：{knowledge[0]["content"]}
背景2：{knowledge[1]["content"]}
背景3：{knowledge[2]["content"]}
基于以上事实回答问题：{user_input}
        '''

        quote = f'''
> 文档：{knowledge[0]["meta"]["source"]}，页码：{knowledge[0]["meta"]["page"]}
> 文档：{knowledge[1]["meta"]["source"]}，页码：{knowledge[1]["meta"]["page"]}
> 文档：{knowledge[2]["meta"]["source"]}，页码：{knowledge[2]["meta"]["page"]}
'''
    else:
        prompt = user_input
    
    history[-1][1] = ""
    if llm_client is None:
        gr.Warning("请先设置大模型")
        response = "模型参数未设置"
    else:
        print("do_llm_response prompt:",prompt)
        response = llm_client(prompt)
        response = response.removeprefix(prompt)
        response += quote

    for character in response:
        history[-1][1] += character
        time.sleep(0.01)
        yield history

llm_client = None
def llm(ak,sk,client):
    global llm_client
    import llm
    llm.init_param(ak,sk)
    if client == "Wenxin":
        llm_client = llm.baidu_client
    elif client == "Tongyi":
        llm_client = llm.qwen_agent_app
    elif client == "Huggingface":
        llm_client = llm.hg_client
    
    if ak == "" and sk == "":
        gr.Info("重置成功")
    else:
        gr.Info("设置成功")

    return llm_client

def file_handler(file_objs,name):
    import shutil
    import os
    
    print("file_obj:",file_objs)
    
    os.makedirs(os.path.dirname("./files/input/"), exist_ok=True)

    for idx, file in enumerate(file_objs):
        print(file)
        file_path = "./files/input/" +  os.path.basename(file.name)
        if not os.path.exists(file_path):
            shutil.move(file.name,"./files/input/")
        
        knowledgeBase.add_documents_to_kb(name,[file_path])

    dbs = knowledgeBase.get_bases()
    dfs = knowledgeBase.get_df_bases()
    return dfs,gr.CheckboxGroup(dbs,label="知识库", info="可选择1个或多个知识库"),gr.Dropdown(dbs,multiselect=True, label="知识库选择")

def do_search(selected_dbs,user_input):
    print("do_search:",selected_dbs,user_input)
    context = knowledgeBase.retrieve_documents(selected_dbs,user_input)
    return context
