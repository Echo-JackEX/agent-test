import os
import json
import time
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
import pathlib

# 加载环境变量
current_dir = pathlib.Path(__file__).parent
load_dotenv(current_dir / ".env")

# ==========================
# 复用之前的 LLMClient（简化版）
# ==========================
class LLMClient:
    def __init__(self, timeout=15.0):
        self.client = OpenAI(
            api_key=os.getenv("ALIYUN_API_KEY"),
            base_url=os.getenv("ALIYUN_API_BASE_URL"),
            timeout=timeout
        )
        self.messages = []

    def set_system_prompt(self, content):
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
# D2 核心：Prompt 策略实验
# ==========================
class PromptEngineeringLab:
    def __init__(self):
        self.llm = LLMClient()

    # --------------------------
    # 实验 1：零样本提示 (Zero-Shot)
    # --------------------------
    def zero_shot_prompt(self, question):
        """零样本：直接给问题，不给示例"""
        system_prompt = "你是一个数学推理专家，请直接回答问题。"
        self.llm.set_system_prompt(system_prompt)
        return self.llm.chat(question)

    # --------------------------
    # 实验 2：少样本提示 (Few-Shot)
    # --------------------------
    def few_shot_prompt(self, question):
        """少样本：给 2-3 个示例，再给问题"""
        system_prompt = """你是一个数学推理专家，请参考下面的示例回答问题。
        示例 1：
        问题：一个仓库有 100 吨货物，第一天运走了10吨，第二天运走剩下的 1/3，第三天又运进了 20 吨，现在仓库有多少吨货物？
        回答：第一天运走 10 吨，剩下 100-10=90 吨；第二天运走 90/3=30 吨，剩下 90-30=60 吨；第三天运进 20 吨，现在有 60+20=80 吨。

        示例 2：
        问题：一个项目有多个节点，如果节点 A 需要 2 天完成，节点 B 需要 3 天完成，节点 C 需要 1 天完成，且节点 B 和 C 可以同时进行，但都必须在节点 A 完成后才能开始，那么整个项目最少需要多少天完成？
        回答：节点 A 需要 2 天完成，节点 B 和 C 可以同时进行，所以节点 B 和 C 需要的时间取决于较长的那个，即 3 天。因此，整个项目最少需要 2（节点 A） + 3（节点 B 和 C） = 5 天完成。

        """
        self.llm.set_system_prompt(system_prompt)
        return self.llm.chat(question)

    # --------------------------
    # 实验 3：思维链提示 (Chain-of-Thought, CoT)
    # --------------------------
    def cot_prompt(self, question):
        """思维链：强制要求 AI 一步步推理，再给答案"""
        system_prompt = """你是一个数学推理专家，请严格按照以下步骤回答：
                        1. 先分析题目给出的条件
                        2. 一步步列出计算过程
                        3. 最后给出最终答案
                        请用清晰的步骤展示你的思考过程。
        """
        self.llm.set_system_prompt(system_prompt)
        return self.llm.chat(question)


    def structured_output(self, user_input):
        # 1. System Prompt：设定角色、规则、输出格式（固定不变）
        system_prompt = """你是一个专业的信息提取助手。
        请从用户输入中提取以下信息，并严格以 JSON 格式输出，不要包含任何其他文字：
        {
            "sentiment": "正面/负面/中性",
            "key_points": ["关键词1", "关键词2"],
            "summary": "100字以内的摘要"
        }

        注意：
        - sentiment 只能是 "正面"、"负面" 或 "中性"
        - key_points 是数组格式
        - summary 不超过 100 字
        """
        self.llm.set_system_prompt(system_prompt)  # 先定规矩
    
    # 2. 用户输入：动态传入的具体内容（每次都不一样）
        raw_reply = self.llm.chat(user_input)  # 再处理具体任务 

        try:
            start = raw_reply.find("{")
            end = raw_reply.rfind("}") + 1
            json_str = raw_reply[start:end]
            return json.loads(json_str)  # 必须有返回值
        except Exception as e:
            return {"error": f"JSON 解析失败: {str(e)}", "raw_reply": raw_reply}

# ==========================
# 运行实验
# ==========================
if __name__ == "__main__":
    lab = PromptEngineeringLab()
    test_question = "一个水池有两个进水管和一个出水管，进水管 A 每小时可以注入 100 升水，进水管 B 每小时可以注入 150 升水，出水管 C 每小时可以排出 120 升水。如果同时打开三个管道，1 小时后水池里会有多少升水？"
    # print("=== 零样本提示 ===")
    # print(lab.zero_shot_prompt(test_question))

    # print("=== 少样本提示 ===")
    # print(lab.few_shot_prompt(test_question))

    # print("=== 思维链提示 ===")
    # print(lab.cot_prompt(test_question))
    print("=== 结构化输出 ===")
    print(lab.structured_output(user_input=input("请输入需要提取信息的文本：")))
