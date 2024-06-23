from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# 检索器
retriever = TavilySearchAPIRetriever(k=3)
# 语言模型
llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",temperature=0)
# 提示
prompt = ChatPromptTemplate.from_template(
    """仅基于提供的上下文用汉语回答问题
上下文: {context}

问题: {question}"""
)
# 集成检索器到链中
chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever) # 传递question和context
    | prompt
    | llm
    | StrOutputParser()
)

from fastapi import FastAPI
from langserve import add_routes

def create_app(chain):
    app = FastAPI(title="测试LangServe", description="测试LangServe")
    
    # Add routes to the app
    add_routes(app, chain, path="/chat")
    
    return app


if __name__ == "__main__":
    # Create FastAPI app
    app = create_app(chain)
    
    # Run the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
