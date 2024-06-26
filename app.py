
import gradio as gr
import numpy as np
import time
from pathlib import Path
from retriever import knowledgeBase
import llm
import os


current_file_path = Path(__file__).resolve()
absolute_path = (current_file_path.parent / "files" / "input").resolve()

components = {}

params = {
    "algo_type": None,
    "input_image":None
}


def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [components[k] for k in keys]

def create_ui():
    with gr.Blocks() as demo:
        with gr.Tab("问答"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        components["chatbot"] = gr.Chatbot(
                                            [(None,"你好，有什么需要帮助的？")],
                                            elem_id="chatbot",
                                            bubble_full_width=False,
                                            height=600,
                                            show_label=False
                            )
                        components["chat_input"] = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)
                        components["db_select"] = gr.CheckboxGroup(knowledgeBase.get_bases(),label="知识库", info="可选择1个或多个知识库")
        
        with gr.Tab("知识库"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                                """
                                ### 知识库列表
                                """)
                    with gr.Group():
                        components["db_view"] = gr.Dataframe(
                                                    headers=["库名"],
                                                    datatype=["str"],
                                                    row_count=1,
                                                    col_count=1,
                                                    interactive=False,
                                                    height=200,
                                                    column_widths=["50px"],
                                                    show_label=False,
                                                    wrap=False
                        )
                        components["file_expr"] = gr.FileExplorer(
                            scale=1,
                            value=[],
                            file_count="single",
                            root_dir=absolute_path,
                            elem_id="file_expr",
                            height=200,
                            show_label=False
                        )
                with gr.Column(scale=2):
                    gr.Markdown(
                            """
                            ### 新建知识库
                            """)
                    with gr.Row():
                        with gr.Column(scale=2):
                            components["db_name"] = gr.Textbox(placeholder="请输入知识库名称",show_label=False, lines=1, value="")
                        with gr.Column(scale=2):
                            components["db_submit_btn"] = gr.Button(value="提交")
                    components["file_upload"] = gr.File(elem_id='file_upload',file_count='multiple',show_label=False,
                                                        label='文档上传', file_types=[".pdf", ".doc", '.docx', '.json', '.csv'])
            gr.Markdown(
                        """
                        ### 知识库检索
                        """)
            with gr.Row():
                with gr.Column(scale=2):
                    components["db_input"] = gr.Textbox(placeholder="关键词", lines=1, value="",show_label=False)
                with gr.Column(scale=1):
                    components["db_test_select"] = gr.Dropdown(knowledgeBase.get_bases(),multiselect=True,show_label=False)
                with gr.Column(scale=1):
                    components["dbtest_submit_btn"] = gr.Button(value="检索")
            with gr.Row():
                with gr.Group():
                    components["db_search_result"] = gr.JSON(label="检索结果")

        create_event_handlers()
        demo.load(init,None,gradio("db_view","db_select","db_test_select"))
    return demo

def init():
    db_list = knowledgeBase.get_bases()
    db_df_list = knowledgeBase.get_df_bases()
    return db_df_list,gr.CheckboxGroup(db_list,label="知识库", info="可选择1个或多个知识库"),gr.Dropdown(db_list,multiselect=True, label="知识库选择")

def create_event_handlers():

    components["db_submit_btn"].click(
        file_handler,gradio('file_upload','db_name'),gradio("db_view",'db_select',"db_test_select")
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

    components['db_view'].select(
        db_expr, gradio('db_view'), gradio('file_expr')
    )

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def do_llm_request(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
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
> 文档：{os.path.basename(knowledge[0]["meta"]["source"])}，页码：{knowledge[0]["meta"]["page"]}
> 文档：{os.path.basename(knowledge[1]["meta"]["source"])}，页码：{knowledge[1]["meta"]["page"]}
> 文档：{os.path.basename(knowledge[2]["meta"]["source"])}，页码：{knowledge[2]["meta"]["page"]}
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


llm_client = llm.baidu_client


def file_handler(file_objs,name):
    import shutil
    import os
    
    print("file_obj:",file_objs)
    
    if name == "":
        gr.Warning("请输入知识库名称！")
        return
    
    if len(file_objs) == 0:
        gr.Warning("请上传文档！")
        return
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

def db_expr(selected_index: gr.SelectData, dataframe_origin):
    print("db_expr",selected_index.index)
    
    dbname = dataframe_origin.iloc[selected_index.index[0],selected_index.index[1]]
    print("db_expr",dbname)
    
    return knowledgeBase.get_db_files(dbname)

def do_search(selected_dbs,user_input):
    if len(selected_dbs) == 0:
        gr.Warning("请选择知识库！")
        return
    if user_input == "":
        gr.Warning("请输入检索关键词！")
        return
    print("do_search:",selected_dbs,user_input)
    context = knowledgeBase.retrieve_documents(selected_dbs,user_input)
    return context

if __name__ == "__main__":
    demo = create_ui()
    # demo.launch(server_name="10.151.124.137")
    demo.launch(root_path="/chat")