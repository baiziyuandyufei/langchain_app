from dotenv import load_dotenv
import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_fireworks import ChatFireworks
from langserve import add_routes

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class ChatService:
    def __init__(self, model_name: str, temperature: float):
        # Initialize language model
        self.llm = ChatFireworks(model=model_name, temperature=temperature)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """用汉语回答问题。
            问题: {question}"""
        )
        
        # Create the processing chain
        self.chain = self.prompt | self.llm | StrOutputParser()

def create_app(chain):
    app = FastAPI(title="测试LangServe", description="测试LangServe")
    
    # Add routes to the app
    add_routes(app, chain, path="/chat")
    
    return app

if __name__ == "__main__":
    # Initialize the chat service with desired configuration
    chat_service = ChatService(
        model_name="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0,
    )
    
    # Create FastAPI app
    app = create_app(chat_service.chain)
    
    # Run the FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
