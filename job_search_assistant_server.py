import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_fireworks import FireworksEmbeddings, ChatFireworks
from langchain.vectorstores import FAISS
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate)
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda)
from langchain_core.output_parsers import StrOutputParser
import re
import logging
from langsmith.run_helpers import traceable
import os

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

# # 直接在info括号内获取并输出环境变量的值
# logging.info(f'LANGCHAIN_TRACING_V2: {os.getenv("LANGCHAIN_TRACING_V2", "未设置")}')
# logging.info(f'LANGCHAIN_ENDPOINT: {os.getenv("LANGCHAIN_ENDPOINT", "未设置")}')
# logging.info(f'LANGCHAIN_API_KEY: {os.getenv("LANGCHAIN_API_KEY", "未设置")}')
# logging.info(f'LANGCHAIN_PROJECT: {os.getenv("LANGCHAIN_PROJECT", "未设置")}')


class JobSearchAssistant:
    def __init__(self, url, embedding_model_name, chat_model_name):
        # 文档加载、分割
        self.loader = WebBaseLoader(url)
        self.raw_documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            add_start_index=False,
        )
        self.documents = self.text_splitter.split_documents(self.raw_documents)
        print(f"分割后快数: {len(self.documents)}")

        # 向量化、存储
        self.embedding_model = FireworksEmbeddings(model=embedding_model_name)
        self.db = FAISS.from_documents(
            documents=self.documents, embedding=self.embedding_model)
        print(f"索引片段数: {self.db.index.ntotal}")

        # 检索器
        self.retriever = self.db.as_retriever()

        # 检索链
        self.question_retrieval_chain = self.retriever | RunnableLambda(
            lambda docs: "\n".join([doc.page_content for doc in docs]))

        # 实例化聊天模型
        self.chat = ChatFireworks(
            model=chat_model_name, temperature=0.3, top_p=0.3)

        # 分类词典
        self.question_classify_dict = {
            "离职原因": {
                "response": "有换工意愿，上家公司离我居住地太远，通勤时间太长。",
                "examples": [{"text": "离职/换工作的原因", "label": "离职原因"}]
            },
            "薪资&薪资范围&期望薪资": {
                "response": "上一家单位薪资为20K*15，当前我期望薪资为30K～40K。",
                "examples": [{"text": "但是我们应该最高30K，一般还达不到.", "label": "薪资&薪资范围&期望薪资"}]
            },
            "外包&外协&外派&驻场": {
                "response": "请发送或说明职位的办公地点定位。以及薪资范围。我期望薪资范围50k以上。",
                "examples": [{"text": "你好，我们是外协岗位，在国家电网 南瑞工作的", "label": "外包&外协&外派&驻场"}]
            },
            "兼职": {
                "response": "请发送或说明职位的办公地点定位。以及薪资范围。我期望薪资范围50k以上。",
                "examples": [{"text": "哈喽～本职位为线上兼职，一单一结款，根据自己时间自由接单，不耽误自己的主业，您看感兴趣嘛？", "label": "兼职"}]
            },
            "预约面试": {
                "response": "请您稍等，我看一下我的时间。",
                "examples": [{"text": "想约您面试，方便的话麻烦告诉我一下您可以约面试的日期及时间。", "label": "预约面试"}]
            },
            "到岗时间": {
                "response": "两周内到岗。",
                "examples": [{"text": "咱到岗时间呢。", "label": "到岗时间"}]
            },
            "简历": {
                "response": "马上发送简历，但如果只是想获取联系方式，就不要耽误大家时间。",
                "examples": [{"text": "您好 方便发一份简历过来吗？", "label": "简历"}]
            },
            "请求微信": {
                "response": "请直接在boss上确定面试时间，发送预约面试请求。",
                "examples": [{"text": "你好 方便加个微信吗？", "label": "请求微信"}]
            },
            "其他": {
                "response": "",
                "examples": []
            }
        }

        # 构建分类提示
        self.examples, self.example_prompt, self.prefix, self.suffix = self.prepare_question_classify_prompt()
        self.few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=["input"],
            example_separator="\n"
        )

        # 分类链
        self.question_classify_chain = self.few_shot_prompt | self.chat | StrOutputParser(
        ) | RunnableLambda(self.label_to_response)

        # 系统提示
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(f"""
        你是一个求职助手，代表 “于先生” 回答HR问题。
        
        以下是于先生个人信息，以下这些信息只有被问到时才输出。
        ```
        工作经历：
        - 中国科学院信息工程研究所 2018年8月至2024年5月 自然语言处理工程师 负责短文本分类  使用pyhton/Java语言 离职。
        - 苏宁易购 2017年8月至2018年7月 自然语言处理工程师 负责商品标题分类和属性词抽取 使用C/C++语言 离职。
        - 同方知网 2015年3月至2017年7月 计算语言学工程师 负责论文抄袭检测算法设计实现、OpenCV相似图像检索 使用C/C++语言 离职。
        现居住地：北京。
        期望工作地：长期base北京，不接受出差，北京广渠门附近或地铁方便，通勤时间1小时以内。
        教育背景：北京信息科技大学 硕士/本科。
        联系方式：请问我的主人。
        期望职位：自然语言处理(NLP)、大模型。
        可访问求职助手APP：https://baiziyuandyufei-job-search-a-job-search-assistant-server-zv2qqt.streamlit.app/。
        ```
        """)

        # 人工提示
        self.human_message_prompt = HumanMessagePromptTemplate.from_template("""HR问或说: {question}。\n\n{context}\n\n请用汉语回复内容，内容的头部和尾部不要出现引号。""")

        # 整体链
        self.final_chain = {
            "question": RunnablePassthrough(),
            "context": RunnableParallel(question_classify_response=self.question_classify_chain,
                                        question_retrieval_response=self.question_retrieval_chain) 
        } | RunnableLambda(self.generate_context_prompt) | \
            ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt]) | \
            self.chat | \
            StrOutputParser()

    # 求最长公共子串
    def longest_common_substring(self, s1, s2):
        # 获取两个字符串的长度
        len_s1 = len(s1)
        len_s2 = len(s2)

        # 创建一个二维数组用来存储动态规划的结果
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        # 初始化最大长度和结束位置
        max_length = 0
        end_pos = 0

        # 填充动态规划表
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        end_pos = i
                else:
                    dp[i][j] = 0

        # 提取最大公共子串
        start_pos = end_pos - max_length
        return s1[start_pos:end_pos]

    # 合并分类和问答提示为context提示
    def generate_context_prompt(self, all_dict):
        question = all_dict["question"]
        question_classify_response = all_dict["context"]["question_classify_response"]
        question_retrieval_response = all_dict["context"]["question_retrieval_response"]
        
        if len(question_classify_response) > 0:
            question_classify_template = f"""你在回答中体现以下内容\n\n{question_classify_response}"""
        else:
            question_classify_template = ""
        
        if len(self.longest_common_substring(question,question_retrieval_response)) >=4:
            question_retrieval_template = f"""工作经历有以下内容: \n\n{question_retrieval_response}"""
        else:
            question_retrieval_template = ""
        return {
            "question":question,
            "context":f"{question_classify_template}\n\n{question_retrieval_template}\n\n"
        }

    def prepare_question_classify_prompt(self):
        examples = []
        for key in self.question_classify_dict:
            r_examples = self.question_classify_dict[key]["examples"]
            if len(r_examples) > 0:
                examples.extend(r_examples)

        example_prompt = PromptTemplate.from_template(
            """文本: {text}
类别: {label}\n"""
        )

        label_li_str = '\n- '.join(self.question_classify_dict.keys())
        label_li_str = '\n- ' + label_li_str

        prefix = f"""给出每个文本的类别，类别只能属于以下列出的一种\n{label_li_str}\n如果不属于以上类别，则类别名称为“其他”。\n例如："""
        suffix = """文本: {input}\n类别:"""
        return examples, example_prompt, prefix, suffix

    def label_to_response(self, label):
        label = re.sub('类别: ?', '', label)
        label = label if label in self.question_classify_dict else "其他"
        response = self.question_classify_dict[label]["response"]
        return response

    
    @traceable  # Auto-trace this function
    def get_response(self, question):
        return self.final_chain.invoke(question)


url = "https://raw.githubusercontent.com/baiziyuandyufei/langchain-self-study-tutorial/main/jl.txt"
embedding_model_name = "nomic-ai/nomic-embed-text-v1.5"
chat_model_name = "accounts/fireworks/models/llama-v3-70b-instruct"

assistant = JobSearchAssistant(url, embedding_model_name, chat_model_name)

# 人机交互界面

# 页面大标题
st.title("个人求职助手")
st.title("💬 聊天机器人")
# 页面描述
st.caption("🚀 一个Streamlit个人求职助手聊天机器人，基于FireWorks的llama-v3-70b-instruct模型")
# 侧边栏
with st.sidebar:
    st.markdown("""
    ## 开发计划

    ### 1. 求职助手

    #### 目的

    代表用户同HR交流

    #### 开发计划

    - [x] [整体链结构设计](https://github.com/baiziyuandyufei/langchain-self-study-tutorial/blob/main/求职助手1.ipynb)
    - [x] [通用聊天模板设计](https://github.com/baiziyuandyufei/langchain-self-study-tutorial/blob/main/求职助手1.ipynb)
    - [x] [问题分类链设计：问题分类FewShot模板设计](https://github.com/baiziyuandyufei/langchain-self-study-tutorial/blob/main/求职助手1.ipynb)
    - [x] [问题检索链设计：简历知识库构建-嵌入模型选择、向量存储、检索](https://github.com/baiziyuandyufei/langchain-self-study-tutorial/blob/main/求职助手1.ipynb)
    - [x] 人机交互界面开发
    - [ ] BOSS上自动回复
        - puppeteer定时登录
        - puppeteer获取top-20个HR，抽取最后一条信息，助手自己回复。
            - HR内容确定一个id,存本地mongo，相同id不再处理。
            - 所有对话内容发送到微信。
            - 检测到面试时间邀约发送到微信。
    - [ ] 新JD检测与打招呼
        - JD与我匹配度计算
    - [ ] JD与我匹配度计算
        - 距离打分。
        - 相关度打分。
        - 薪资打分。
        - 公司人数打分。
        - 注册资金打分。
        - 公司性质打分。
        - 加权总分。
    - [ ] 工作地点与居住地点距离计算
        - 爬取查询任意两地点经纬度，计算直线距离。
        - langchain本地sqlite工具，便于后续放到streamlit

    #### 当前进度

    - 获取地点经纬度，计算直线距离。
    - 听说庆余年二季发布了是吗？要看。

    ### 2. 快速生成简历

    #### 目的

    根据不同JD描述，生成适配的简历内容

    #### 填写项

    用户填表，点击提交后，生成简历。

    - 岗位职责：
    - 岗位描述：
    - 学历信息：
    - 工作经历：
        - 开发语言
        - 工具库
        - 模型
        - 其他
    - 离职原因：
    - 现居住地：

    """)

# 初始化聊天消息会话
if "messages" not in st.session_state:
    #  添加助手消息
    st.session_state["messages"] = [
        {"role": "assistant", "content": "我是求职助手，替我的主人回答HR的问题，你可以将问题输入给我！"}]

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
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    # 显示助手消息
    st.chat_message("assistant").write(response)
