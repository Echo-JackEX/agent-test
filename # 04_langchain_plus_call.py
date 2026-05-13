import os
from dotenv import load_dotenv
import pathlib

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# RAG 专用依赖
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

# 加载环境变量
current_dir = pathlib.Path(__file__).parent
load_dotenv(current_dir / ".env")

# ==========================
class LangChainD3:
    def __init__(self):
        # LLM 初始化
        self.llm = ChatOpenAI(
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url=os.getenv("ALIYUN_API_BASE_URL"),
            model="qwen3.5-flash",
            temperature=0.7,
            streaming=True,
            max_tokens=1024
        )
        self.parser = StrOutputParser()
        self.store = {}  # 对话记忆仓库

        # 嵌入模型
        self.embeddings = DashScopeEmbeddings(
            dashscope_api_key=os.getenv("ALIYUN_API_KEY"),
            model="text-embedding-v3"
        )

        # 向量库配置
        self.vector_db_path = "./faiss_db"
        self.vector_db = None

    # --------------------------
    # 1. 翻译链
    # --------------------------
    def get_translate_chain(self):
        prompt = ChatPromptTemplate.from_template("把以下内容翻译成{language}：{text}，只输出结果")
        return prompt | self.llm | self.parser

    # --------------------------
    # 2. 诗词赏析链
    # --------------------------
    def get_analyze_chain(self):
        def analyze(text):
            sum_prompt = ChatPromptTemplate.from_template("这首诗的创作背景100字内：{text}")
            summary = (sum_prompt | self.llm | self.parser).invoke({"text": text})
            
            sen_prompt = ChatPromptTemplate.from_template("判断情感：正面/负面/中性，只输出：{text}")
            sentiment = (sen_prompt | self.llm | self.parser).invoke({"text": text})
            
            analysis_prompt = ChatPromptTemplate.from_template("生成诗词赏析：\n背景：{s}\n情感：{sen}")
            analysis = (analysis_prompt | self.llm | self.parser).invoke({"s": summary, "sen": sentiment})
            return analysis
        return analyze

    # --------------------------
    # 3. 带记忆多轮对话
    # --------------------------
    def get_chat_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个海上的水手，回答时不要用任何Markdown格式，不要加粗，不要星号，只用纯文字自然回答"),
            MessagesPlaceholder("history"),
            ("user", "{input}")
        ])
        chain = prompt | self.llm | self.parser
    
        # 新版窗口记忆：只保留最近5轮对话，自动截断，管控Token
        def get_session_history(session_id):
            if session_id not in self.store:
                self.store[session_id] = InMemoryChatMessageHistory()
            
            # 限制最多5轮对话（10条消息：5问5答）
            history = self.store[session_id]
            max_rounds = 5
            if len(history.messages) > max_rounds * 2:
                history.messages = history.messages[-(max_rounds * 2):]
            return history
    
        return RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    # --------------------------
    # 通用文档加载器
    # --------------------------
    def load_document(self, file_path):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith((".txt", ".md")):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("不支持的文件格式")
        return loader.load()

    # --------------------------
    # 文档切分
    # --------------------------
    def load_and_split_documents(self, file_path):
        documents = self.load_document(file_path)
        print(f"✅ 成功加载：{file_path}，共 {len(documents)} 页/段")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    # --------------------------
    # 🔥 修复：构建向量库（删除重复代码+弃用Chroma参数）
    # --------------------------
    def build_vector_db(self, chunks):
        print("🔄 正在生成嵌入向量并构建向量库...")
        # FAISS 唯一正确写法
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(self.vector_db_path)
        print("✅ 向量库构建完成，已保存到本地")

    # --------------------------
    # 修复：智能加载向量库（检测文档变化）
    # --------------------------
    def load_existing_vector_db(self):
        import shutil
        import json
        exts = [".pdf", ".txt", ".md", ".docx"]
        record_file = "./doc_record.json"

        # 扫描当前文档
        current_docs = {}
        for file in os.listdir("."):
            if any(file.endswith(ext) for ext in exts):
                current_docs[file] = os.path.getmtime(file)

        if not current_docs:
            print("❌ 本地未找到任何文档")
            return False

        # 对比文档是否变化
        docs_changed = False
        if os.path.exists(record_file):
            old_docs = json.load(open(record_file, "r", encoding="utf-8"))
            if old_docs != current_docs:
                docs_changed = True
        else:
            docs_changed = True

        # 文档变化 → 删除旧库
        if docs_changed:
            shutil.rmtree(self.vector_db_path, ignore_errors=True)
            json.dump(current_docs, open(record_file, "w", encoding="utf-8"))
            print("🔄 文档已更新/首次运行，准备重建向量库")
            return False

        # 加载向量库
        if os.path.exists(self.vector_db_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.vector_db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ 文档无变化，直接加载向量库")
                return True
            except:
                shutil.rmtree(self.vector_db_path, ignore_errors=True)
                print("⚠️ 向量库损坏，自动删除重建")
                return False

        print("❌ 未找到向量库，准备新建")
        return False

    # --------------------------
    # 🔥 修复：带记忆的RAG链（支持多轮对话+自动分段）
    # --------------------------
    def get_rag_chain(self):
        if not self.vector_db:
            raise Exception("请先构建或加载向量库！")

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # 格式化文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # RAG 提示词
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是专业的文档问答助手，严格遵守规则：
              1. 只根据上下文回答，不编造内容
              2. 无答案时回复：抱歉，我在文档中没有找到相关信息
              3. 纯文字输出，无Markdown、无符号
              4. 自动分段，重点用1、2、3分点罗列
              5. 结合聊天历史回答"""),
            # 记忆占位符
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}"),
            ("system", "上下文：{context}")
        ])

        # 核心流水线
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["question"]))
            )
            | rag_prompt
            | self.llm
            | self.parser
        )

        # ✅ 新版窗口记忆：自动截断，只保留最近 5 轮对话（管控Token）
        def get_session_history(session_id):
            key = f"rag_{session_id}"
            if key not in self.store:
                self.store[key] = InMemoryChatMessageHistory()

            # 🔥 限制最多保留最近5轮（10条消息：5问5答）
            history = self.store[key]
            max_rounds = 5
            if len(history.messages) > max_rounds * 2:
                history.messages = history.messages[-(max_rounds * 2):]
            return history

        return RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

# ==========================
# 主程序
# ==========================
if __name__ == "__main__":
    app = LangChainD3()
    print("=== D4 LangChain RAG 演示 ===")
    print("1 翻译链 | 2 诗词赏析 | 3 多轮对话 | 4 RAG知识库问答")
    c = input("选择：").strip()

    if c == "1":
        print(app.get_translate_chain().invoke({
            "language": input("目标语言："),
            "text": input("文本：")
        }))
    elif c == "2":
        print("赏析：", app.get_analyze_chain()(input("输入诗词：")))
    elif c == "3":
        chat = app.get_chat_chain()
        while True:
            msg = input("你：")
            if msg == "exit": break
            print("AI：", chat.invoke({"input": msg}, config={"configurable": {"session_id": "1"}}))
    elif c == "4":
        print("\n=== RAG知识库问答 ===")
        if not app.load_existing_vector_db():
            exts = [".pdf", ".txt", ".md", ".docx"]
            files = [f for f in os.listdir(".") if any(f.endswith(ext) for ext in exts)]

            if files:
                all_chunks = []
                print(f"✅ 检测到 {len(files)} 个文档，批量加载中...")
                for file in files:
                    chunks = app.load_and_split_documents(file)
                    all_chunks.extend(chunks)
                app.build_vector_db(all_chunks)
            else:
                file_path = input("未找到支持的文档，请输入路径：")
                chunks = app.load_and_split_documents(file_path)
                app.build_vector_db(chunks)

        # 🔥 修复：RAG 调用方式（带记忆）
        rag_chain = app.get_rag_chain()
        print("\n✅ RAG准备就绪！输入exit退出")
        while True:
            question = input("你：")
            if question == "exit": break
            answer = rag_chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": "rag_1"}}
            )
            print("AI：", answer)
