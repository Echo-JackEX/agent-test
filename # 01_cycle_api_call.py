import os
import time
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
import pathlib

# 加载环境变量
current_dir = pathlib.Path(__file__).parent
load_dotenv(current_dir / ".env")

# ==========================
# LLM 客户端封装
# ==========================
class LLMClient:
    def __init__(self, timeout=15.0):
        """初始化客户端"""
        self.client = OpenAI(
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url=os.getenv("ALIYUN_API_BASE_URL"),
            timeout=timeout
        )
        self.messages = []  # 多轮对话记忆

    def set_system_prompt(self, content):
        """设置系统提示词"""
        self.messages = [{"role": "system", "content": content}]

    def chat(
        self,
        user_input,
        model="qwen3.5-flash",
        temperature=0.7,
        max_tokens=1024,
        retry_times=3
    ):
        # 加入用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 重试机制
        for attempt in range(retry_times):
            try:
                # 调用 API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # 获取回复
                reply = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": reply})
                return reply

            except APITimeoutError:
                print(f"超时，重试 {attempt+1}/{retry_times}")
                time.sleep(1)

            except RateLimitError:
                return "错误：请求频率过高，请稍后再试"

            except APIError as e:
                if "402" in str(e):
                    return "错误：免费额度已用完"
                elif "401" in str(e):
                    return "错误：API Key 无效"
                else:
                    return f"API错误：{str(e)}"

            except Exception as e:
                return f"未知错误：{str(e)}"

        return "重试多次失败，请检查网络"

# ==========================
# 测试：多轮对话机器人
# ==========================
if __name__ == "__main__":
    # 初始化
    llm = LLMClient()
    llm.set_system_prompt("你是一个风趣幽默的老师，擅长用生动的例子解释复杂的概念。")

    print("=== 多轮对话机器人（输入 exit 退出）===\n")

    while True:
        user_msg = input("你：")
        if user_msg.lower() in ["exit", "quit", "退出"]:
            print("对话结束")
            break

        # 自定义参数调用
        ai_reply = llm.chat(
            user_input=user_msg,
            model="qwen3.5-flash",
            temperature=0.5,
            max_tokens=1024
        )

        print("AI：", ai_reply, "\n")
