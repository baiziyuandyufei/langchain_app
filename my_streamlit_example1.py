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
load_dotenv()

st.title("个人求职助手")
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct",
                    temperature=0.3,
                    top_p=0.3)
system_message_prompt = SystemMessagePromptTemplate.from_template("你是一个求职助手，用汉语交流。")
human_message_prompt = HumanMessagePromptTemplate.from_template("HR问或说：“{input}”，你用汉语回答：")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt,
     human_message_prompt]
)
chain = chat_prompt|llm|StrOutputParser()

# Title and caption
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by llama-v3-70b-instruct")

# Sidebar for OpenAI API Key
with st.sidebar:
    st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "我是你的求职助手，帮你回答HR提出的问题，你可以将HR问题告诉我！"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input form
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = chain.invoke({'input':prompt})  # Placeholder for your language model response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
