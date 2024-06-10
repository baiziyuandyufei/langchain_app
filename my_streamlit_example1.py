import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# 部署到streamlit时，请在streamlit中配置环境变量
load_dotenv()
# 初始化语言模型
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct",
                    temperature=0.3,
                    top_p=0.3)
# 系统提示
system_message_prompt = SystemMessagePromptTemplate.from_template("你是一个求职助手，用汉语交流。")
# 用户提示
human_message_prompt = HumanMessagePromptTemplate.from_template("HR问或说：“{input}”，你用汉语回答：")
# 对话提示
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt,
     human_message_prompt]
)
# 对话链
chain = chat_prompt|llm|StrOutputParser()
# 页面大标题
st.title("个人求职助手")
st.title("💬 聊天机器人")
# 页面描述
st.caption("🚀 一个Streamlit个人求职助手聊天机器人，基于FireWorks的llama-v3-70b-instruct模型")
# 侧边栏
with st.sidebar:
    # 密码框
    st.text_input("密码框", key="chatbot_api_key", type="password")
    "[API申请](#)"
    "[查看源码](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![在GitHub Codespaces打开](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/streamlit/llm-examples?quickstart=1)"
# 初始化聊天消息会话
if "messages" not in st.session_state:
    #  添加助手消息
    st.session_state["messages"] = [{"role": "assistant", "content": "我是你的个人求职助手，帮你回答HR提出的问题，你可以将HR的问题输入给我！"}]
# 显示会话中的所有聊天消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 聊天输入表格
# 这句代码使用了海象运算符，将用户在聊天输入框中输入的内容赋值给变量prompt，并检查这个输入内容是否为真（即是否有输入内容）。
if prompt := st.chat_input("HR的问题"):
    # 向会话消息中添加用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 显示用户输入
    st.chat_message("user").write(prompt)
    # 调用链获取响应
    response = chain.invoke({'input':prompt})
    # 向会话消息中添加助手输入
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 显示助手消息
    st.chat_message("assistant").write(response)
