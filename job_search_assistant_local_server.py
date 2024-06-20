
from fastapi import FastAPI, Request
from job_search_assistant_server import JobSearchAssistant

url = "https://raw.githubusercontent.com/baiziyuandyufei/langchain-self-study-tutorial/main/jl.txt"
embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
chat_model_name = "accounts/fireworks/models/llama-v3-70b-instruct"

assistant = JobSearchAssistant(url, embedding_model_name, chat_model_name)

app = FastAPI(
    title="求职助手本地服务版",
    description="基于LangChain构建的求职助手"
)

# 原生添加路由方式 
# curl -X POST http://localhost:8006/job -H "Content-Type: application/json" -d '{"question": "What jobs are available?"}'
@app.post("/job")
async def handle_job(request: Request):
    body = await request.json()
    result = assistant.get_response(body["question"])
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
