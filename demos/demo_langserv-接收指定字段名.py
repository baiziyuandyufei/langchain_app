"""
curl -X POST http://127.0.0.1:8000/chat/invoke \
    -H "Content-Type: application/json" \
    -d '{"input":{"job_title":"中级会计"},"config":{}}'
"""
from langchain_core.prompts import PromptTemplate,format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks import ChatFireworks
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

chat = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",temperature=0,max_tokens=3000)
receive_prompt = PromptTemplate.from_template("{job_title}")
system_prompt = PromptTemplate.from_template("""
提取职位信息，每条职位信息可以包括以下字段: 

- 职位名
- 薪资范围
- 公司名
- 职位地点
- HR名字
- 公司性质
- 学历要求
- 经验要求
- 职位性质
- 发布时间

没有找到的字段不输出。
                                             
输出格式举例如下：
                                             
1. 职位名: nlp自然语言处理工程师
- 薪资范围: 1万-2万
- 公司名: 四川智服人力资源有限公司
- 职位地点: 北京·海淀·西北旺
- HR名字: 罗镇坤
- 公司性质: 民营

2. 职位名: 自然语言处理工程师
- 薪资范围: 2.9万-3.5万
- 公司名: 北京友安丰廷创新科技有限公司
- 职位地点: 北京·西城·展览路
- HR名字: 宋女士
- 公司性质: 民营

...

以下是待提取职位信息的文本内容: 
                             
{content}

输出普通字符串即可。        
""")

def crawl_page(job_title):
    job_title = job_title.to_string()
    urls = [f"http://www.rcxx.com/search/searjobok.aspx?keyword={job_title}"]
    loader = AsyncChromiumLoader(urls,user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    doc_prompt = PromptTemplate.from_template("{page_content}")
    content = "\n\n".join(format_document(doc,doc_prompt) for doc in docs_transformed)
    return {"content":content}

chain =  receive_prompt | RunnableLambda(crawl_page) | system_prompt | chat | StrOutputParser()

from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="测试LangServe",
              description="测试LangServe")

add_routes(app,chain,path='/chat')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)

