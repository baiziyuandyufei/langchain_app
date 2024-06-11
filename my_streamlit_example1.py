import streamlit as st
from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
import re
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import logging

# 配置日志
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
# 获取日志记录器
logger = logging.getLogger(__name__)

# 部署到streamlit时，请在streamlit中配置环境变量
load_dotenv()

class JobAssistant:
    def __init__(self, model_path="accounts/fireworks/models/llama-v3-70b-instruct", 
                temperature=0.3,
                top_p=0.3):
        self.llm = ChatFireworks(model=model_path, temperature=temperature)
        self.response_dict = {
            "离职原因": {
                "response": "有换工意愿，上家公司离我居住地太远，通勤时间太长。",
                "examples": [{"text": "离职/换工作的原因","label": "离职原因"}]
            },
            "薪资": {
                "response": "我期望薪资为30K～40K。",
                "examples": [{"text": "但是我们应该最高30K，一般还达不到.","label": "薪资"}]
            },
            "外包&外协&外派&驻场": {
                "response": "请发送或说明职位的办公地点定位。以及薪资范围。我期望薪资范围30-40K？",
                "examples": [{"text": "你好，我们是外协岗位，在国家电网 南瑞工作的","label": "外包&外协&外派&驻场"}]
            },
            "兼职": {
                "response": "职位的办公地点在哪？薪资多少，怎么结算？",
                "examples": [{"text": "哈喽～本职位为线上兼职，一单一结款，根据自己时间自由接单，不耽误自己的主业，您看感兴趣嘛？","label":"兼职"}]
            },
            "预约面试": {
                "response": "本周内上午、下午都有时间。",
                "examples": [{"text": "想约您面试，方便的话麻烦告诉我一下您可以约面试的日期及时间【请选择工作日内的上午10-12点或下午14点到17点内的时间】。","label":"预约面试"}]
            },
            "到岗时间": {
                "response": "两周内到岗。",
                "examples": [{"text": "咱到岗时间呢。","label":"到岗时间"}]
            },
            "其他": {
                "response": "",
                "examples": []
            }
        }

        self.examples = []
        for key in self.response_dict:
            r_examples = self.response_dict[key]["examples"]
            if len(r_examples) > 0:
                self.examples.extend(r_examples)

        self.example_prompt = PromptTemplate.from_template(
            """文本: {text}
            类别: {label}
            """
        )

        self.prefix = f"""
        给出每个文本的类别，类别只能属于以下列出的一种

        {"- ".join(self.response_dict.keys())}

        如果不属于以上类别，则类别名称为“其他”。

        例如：
        """

        self.suffix = """文本: {input}\n类别:
        """

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=["input"],
            example_separator="\n"
        )

        self.chain = self.few_shot_prompt | self.llm | StrOutputParser()

        self.system_message_prompt = SystemMessagePromptTemplate.from_template("你是求职助手于先生，用汉语交流。")
        self.human_message_prompt = HumanMessagePromptTemplate.from_template("HR问或说: “{question}”。{response}你用汉语回答: ")
        self.prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt])

        self.final_chain = {"question": itemgetter("input"),
                            "response": itemgetter("input") | RunnableLambda(self.question_classify)} | \
                           self.prompt | self.llm | StrOutputParser()

    def question_classify(self, text):
        label = ""
        text = text.strip()
        if len(text) > 0:
            label = self.chain.invoke({"input": text})
            label = re.sub('类别: ?', '', label)
        label = label if label in self.response_dict else "其他"
        logger.info(f"问题类别: {label}")
        response = self.response_dict[label]["response"]
        if len(response)>0:
            response = f"你在回答中体现一下内容: {response}" 
        logger.info(f"问题分类响应: {response}")
        return response

    def get_response(self, text):
        response = self.final_chain.invoke({"input":text})
        response = re.sub(r'^(["「“])(.+?)(["」”])$', 
                      lambda m: m.group(2) if (m.group(1) == m.group(3) or 
                                               (m.group(1) == '"' and m.group(3) == '"') or 
                                               (m.group(1) == '「' and m.group(3) == '」') or 
                                               (m.group(1) == '“' and m.group(3) == '”')) 
                      else m.group(0), response)
        return response

# 使用示例
assistant = JobAssistant()

# 页面大标题
st.title("个人求职助手")
st.title("💬 聊天机器人")
# 页面描述
st.caption("🚀 一个Streamlit个人求职助手聊天机器人，基于FireWorks的llama-v3-70b-instruct模型")
# 侧边栏
with st.sidebar:
    st.write("什么也不想写")
    
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
    logger.info(f"用户输入: {prompt}")
    # 向会话消息中添加用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 显示用户输入
    st.chat_message("user").write(prompt)
    # 调用链获取响应
    response = assistant.get_response(prompt)
    logger.info(f"AI响应: {response}")
    # 向会话消息中添加助手输入
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 显示助手消息
    st.chat_message("assistant").write(response)
