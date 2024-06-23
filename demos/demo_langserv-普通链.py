"""
请求客户端
curl -X POST http://localhost:8000/chat/invoke \   
     -H "Content-Type: application/json" \
     -d '{"input":{"question":"你好"},"config":{}}'

{"output":"你好！很高兴能够为您提供帮助。如果您有任何关于中文语言或其他主题的问题，请随时告诉我，我会尽力提供有用的回答。","metadata":{"run_id":"755e38bd-4aea-4291-a1fc-14115337afe3","feedback_tokens":[]}}
"""
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks import ChatFireworks

# 语言模型
llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",temperature=0)
# 提示
prompt = ChatPromptTemplate.from_template(
    """用汉语回答问题。
    问题: {question}"""
)
# 集成检索器到链中
chain = prompt| llm | StrOutputParser()

from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="测试LangServe",
              description="测试LangServe")

add_routes(app,chain,path='/chat')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)


