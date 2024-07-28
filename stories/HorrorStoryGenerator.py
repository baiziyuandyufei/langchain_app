import sqlalchemy as sa
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import re
import os
import shutil
from PIL import Image
from gradio_client import Client
from docx import Document
from docx.shared import Inches
from tqdm import tqdm
import json

class HorrorStoryGenerator:
    def __init__(self, db_uri, model_name, image_model_name, interface_name):
        load_dotenv()
        self.db = SQLDatabase.from_uri(db_uri)
        self.metadata = sa.MetaData()
        self.lingyizhi = sa.Table(
            "lingyizhi",
            self.metadata,
            sa.Column("id", sa.INTEGER, primary_key=True),
            sa.Column("title", sa.TEXT),
            sa.Column("content", sa.TEXT),
        )
        
        if interface_name == "ChatGroq":
            self.chat = ChatGroq(
                temperature=0.3,
                model=model_name,
                # model_kwargs={
                #     "frequency_penalty": 0.5,
                #     "presence_penalty": 0.5,
                #     "top_p": 0.6
                # },
            )
        else:
            raise ValueError(f"Unsupported interface name: {interface_name}")

        self.prompt = PromptTemplate.from_template(
"""
你是一名汉语故事创作者，请你润色用户输入的故事素材，使其行文流畅逻辑通顺，用汉语交流。

用户输入的故事素材:

{ref_text}

将故事素材汇总为一篇故事并输出。

严格遵守以下输出格式:

```
标题:xxx
内容:
xxx
```
""")
        self.chain = self.prompt | self.chat | StrOutputParser()
        self.client = Client(image_model_name)
    
    def query_stories(self, keywords, top_n):
        conditions = [self.lingyizhi.c.content.like(f'%{keyword}%') for keyword in keywords]
        length_condition = sa.func.length(self.lingyizhi.c.content) >= 1000
        query = sa.select(self.lingyizhi).where(sa.and_(sa.or_(*conditions), length_condition)).limit(top_n)
        result = self.db.run(query, fetch="cursor")
        stories = list(result.mappings())
        return stories

    
    def generate_story(self, input_text, ref_text):
        content = self.chain.invoke({"ref_text":ref_text})
        content = re.sub('[\r\n]+','\n',content)
        return content
    
    def generate_stories(self, g_path, keywords, top_n):
        os.makedirs(g_path, exist_ok=True)
        stories = self.query_stories(keywords, top_n)

        with tqdm(total=len(stories), desc="Generating stories", unit="story") as pbar:
            for story in stories:
                title = story["title"]
                content = story["content"]
                story_content = self.generate_story(title, content)
                story_txt_path = os.path.join(g_path, f"{title}.txt")
                with open(story_txt_path, "w", encoding="utf-8") as f:
                    f.write(story_content)
                pbar.set_postfix({"title": title})
                pbar.update(1)

if __name__ == "__main__":
    # 使用示例
    generator = HorrorStoryGenerator(
        db_uri="sqlite:///db/horror.db", 
        model_name="llama3-70b-8192", 
        image_model_name="ByteDance/SDXL-Lightning", 
        interface_name="ChatGroq"
    )
    # keywords = ["出租车", "货车", "卡车", "灵车", "司机"]
    keywords = ["河南"]
    generator.generate_stories(
        g_path=os.path.join("stories", "_".join(keywords)),
        keywords=keywords, 
        top_n=7
    )
