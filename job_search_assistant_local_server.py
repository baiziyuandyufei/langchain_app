
from fastapi import FastAPI
from langserve import add_routes
from job_search_assistant_server import JobSearchAssistant
from dotenv import load_dotenv
import os

# 部署到streamlit时，请在streamlit中配置环境变量
load_dotenv()

# 直接在info括号内获取并输出环境变量的值
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "job-search-assistant-local"

# 部署服务
url = "https://raw.githubusercontent.com/baiziyuandyufei/langchain-self-study-tutorial/main/jl.txt"
# embedding_model_name = "BAAI/bge-large-zh-v1.5"
embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
chat_model_name = "accounts/fireworks/models/llama-v3-70b-instruct"
assistant = JobSearchAssistant(url, embedding_model_name, chat_model_name)
print(assistant.final_chain.invoke({"question":"你好"}))

app = FastAPI(
    title="求职助手本地服务版",
    description="基于LangChain构建的求职助手"
)
add_routes(app, assistant.final_chain,path='/job')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
