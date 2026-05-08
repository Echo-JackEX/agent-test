import os
from openai import OpenAI
from dotenv import load_dotenv
import pathlib

# 加载 .env
current_dir = pathlib.Path(__file__).parent
load_dotenv(current_dir / ".env")

# 正确客户端（无proxies）
client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url=os.getenv("ALIYUN_API_BASE_URL")
)

# 调用
response = client.chat.completions.create(
    model="qwen3.5-flash",
    messages=[{"role": "user", "content": "你好，请详细介绍一下你自己。" }]
)

print(response.choices[0].message.content)
