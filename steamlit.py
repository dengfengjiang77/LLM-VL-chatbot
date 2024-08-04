

# 导入操作系统模块，用于设置环境变量
import os
# 设置环境变量 OMP_NUM_THREADS 为 8，用于控制 OpenMP 线程数
os.environ["OMP_NUM_THREADS"] = "8"

# 导入时间模块
import time
# 导入 Streamlit 模块，用于创建 Web 应用
import streamlit as st
# 从 transformers 库中导入 AutoTokenizer 类
from transformers import AutoTokenizer
# 从 transformers 库中导入 TextStreamer 类
from transformers import TextStreamer, TextIteratorStreamer
# 从 ipex_llm.transformers 库中导入 AutoModelForCausalLM 类
from ipex_llm.transformers import AutoModelForCausalLM
# 导入 PyTorch 库
import torch
from threading import Thread

# 导入RAG相关模块和方法
from run_rag import Config, main_rag


# 指定模型路径
load_path = "qwen2chat_int4"
# 加载低比特率模型
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
# 从预训练模型中加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

# 定义生成响应函数
def generate_response(messages, message_placeholder):
    # 将用户的提示转换为消息格式
    # messages = [{"role": "user", "content": prompt}]
    # 应用聊天模板并进行 token 化
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    
    # 创建 TextStreamer 对象，跳过提示和特殊标记
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 使用 zip 函数同时遍历 model_inputs.input_ids 和 generated_ids
    generation_kwargs = dict(inputs=model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer



# Streamlit 应用部分

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []


# 设置应用标题
st.title("鲁大师上云了")

# 普通模式
st.header("普通模式")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("你想说点什么?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        streamer = generate_response(st.session_state.messages, message_placeholder)
        for text in streamer:
            response += text
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# 专家模式：文件上传和问题输入
st.header("专家模式")

uploaded_file = st.file_uploader("上传你的文件进行RAG处理", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("处理中..."):
        # 保存上传的文件
        file_path = os.path.join("./data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 更新Config中的data_path
        config = Config()
        config.data_path = file_path

        # 反馈上传成功
        st.success("语料上传成功！")

rag_input = st.text_input("输入你的问题来进行RAG处理")

if st.button("进行RAG处理"):
    if uploaded_file is not None:
        with st.spinner("处理中..."):
            # 调用RAG处理函数
            rag_response = main_rag(rag_input)
            st.write("RAG Response:")
            st.write(rag_response)
    else:
        st.warning("请先上传语料文件。")