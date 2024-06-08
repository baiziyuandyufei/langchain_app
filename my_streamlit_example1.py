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
import os

fireworks_api_key = os.getenv('FIREWORKS_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
langchain_hub_api_key = os.getenv('LANGCHAIN_HUB_API_KEY')

# Now you can use these variables in your application
print(f"Fireworks API Key: {fireworks_api_key}")
print(f"Anthropic API Key: {anthropic_api_key}")
print(f"Hugging Face Hub API Token: {huggingfacehub_api_token}")
print(f"Langchain Hub API Key: {langchain_hub_api_key}")

st.title("个人求职助手")
llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct",temperature=0.3,top_p=0.3)
system_message_prompt = SystemMessagePromptTemplate.from_template("你是一个求职助手，用汉语交流。")
human_message_prompt = HumanMessagePromptTemplate.from_template("HR问或说：“{input}”，你用汉语回答：")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt,
     human_message_prompt]
)
chain = chat_prompt|llm|StrOutputParser()
with st.form('my_form'):
    text = st.text_area('输入HR提出的问题','请问，最近有换工作的意愿么？我们正在寻找一位团队伙伴。')
    submitted = st.form_submit_button('提交')
    if submitted:
        st.info(chain.invoke({'input':text}))
