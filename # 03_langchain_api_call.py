import os
from dotenv import load_dotenv
import pathlib

# LangChain 导入
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 加载环境变量，后续可以直接调用
current_dir = pathlib.Path(__file__).parent
load_dotenv(current_dir / ".env")

# ==========================
class LangChainD3:
    def __init__(self):
        # 迁移 D1 API 调用
        self.llm = ChatOpenAI(
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url=os.getenv("ALIYUN_API_BASE_URL"),
            model="qwen3.5-flash",
            temperature=0.7,

            streaming=True,   # <-- 流式调用，提速
            max_tokens=512    # <-- 限制长度，提速
        )
        self.parser = StrOutputParser()
        self.store = {}

    # --------------------------
    # 链1：简单翻译链
    # --------------------------
    def get_translate_chain(self):
        prompt = ChatPromptTemplate.from_template("把以下内容翻译成{language}：{text}，只输出结果")
        return prompt | self.llm | self.parser

    # --------------------------
    # 链2：多步骤分析链
    # --------------------------
    def get_analyze_chain(self):
        def analyze(text):
            # 步骤1 总结
            sum_prompt = ChatPromptTemplate.from_template("这首诗的创作背景100字内：{text}")
            summary = (sum_prompt | self.llm | self.parser).invoke({"text": text})
            
            # 步骤2 情感
            sen_prompt = ChatPromptTemplate.from_template("判断情感：正面/负面/中性，只输出：{text}")
            sentiment = (sen_prompt | self.llm | self.parser).invoke({"text": text})
            
            # 步骤3 赏析
            analysis_prompt = ChatPromptTemplate.from_template("生成诗词赏析：\n背景：{s}\n情感：{sen}")
            analysis = (analysis_prompt | self.llm | self.parser).invoke({"s": summary, "sen": sentiment})
            return analysis
        return analyze

    # --------------------------
    # 带记忆的多轮对话
    # --------------------------
    def get_chat_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个海上的水手，回答时不要用任何Markdown格式，不要加粗，不要星号，只用纯文字自然回答"),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ])
        chain = prompt | self.llm | self.parser
        
    # 记住对话的关键代码
        def get_session_history(session_id):
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            return self.store[session_id]

        return RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

# ==========================
# 运行
# ==========================
if __name__ == "__main__":
    app = LangChainD3()
    print("=== D3 LangChain 演示 ===")
    print("1 翻译链 | 2 诗词赏析 | 3 多轮对话（带记忆）")
    c = input("选择：")

    if c == "1":
        print(app.get_translate_chain().invoke({
            "language": input("目标语言："),
            "text": input("文本：")
        }))
    elif c == "2":
        print("赏析：", app.get_analyze_chain()(input("输入诗词：")))
    elif c == "3":
        chat = app.get_chat_chain()
        while 1:
            msg = input("你：")
            if msg == "exit": break
            print("AI：", chat.invoke({"input": msg}, config={"configurable": {"session_id": "1"}})) 
            #config用于多用户场景，如果是单用户可以固定session_id，或者不传默认使用单用户模式
