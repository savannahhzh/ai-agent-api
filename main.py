from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from openai import APIConnectionError, APITimeoutError, OpenAI
from dotenv import load_dotenv
import os
import asyncio
from typing import cast
from pydantic import BaseModel, Field
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionToolMessageParam,
)
import json
import time

load_dotenv()
app = FastAPI(title="AI Agent API", version="2.0")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_RETRY_DELAY_SECONDS = float(os.getenv("OPENAI_RETRY_DELAY_SECONDS", "0.8"))

# 跨域（安全稳定版）
app.add_middleware(
    cast(object, CORSMiddleware),
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.closeai-asia.com/v1"
)

# ==============================================
# 会话管理（线程安全 + 类型严格）
# ==============================================
class ChatSession:
    def __init__(self):
        self.messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content="你是一个有用的AI助手，能够调用工具回答问题。"
            )
        ]

    def add_user_message(self, content: str):
        self.messages.append(ChatCompletionUserMessageParam(role="user", content=content))

    def add_assistant_message(self, msg: ChatCompletionAssistantMessageParam):
        self.messages.append(msg)

    def add_tool_message(self, tool_call_id: str, content: str):
        self.messages.append(ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=tool_call_id,
            content=content
        ))

# 全局会话与锁
sessions: dict[str, ChatSession] = {}
session_locks: dict[str, asyncio.Lock] = {}

def get_or_create_session(session_id: str) -> ChatSession:
    if session_id not in sessions:
        sessions[session_id] = ChatSession()
    return sessions[session_id]

def get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]


async def create_chat_completion_with_retry(**kwargs: object):
    """对上游连接异常做有限重试，降低 WinError 10054 对请求的影响。"""
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(
                client.chat.completions.create,
                timeout=OPENAI_TIMEOUT_SECONDS,
                **kwargs,
            )
        except (APIConnectionError, APITimeoutError):
            if attempt >= OPENAI_MAX_RETRIES:
                raise
            await asyncio.sleep(OPENAI_RETRY_DELAY_SECONDS * (attempt + 1))

# ==============================================
# 请求体
# ==============================================
class ChatRequest(BaseModel):
    session_id: str = Field("default", min_length=1)
    message: str = Field(..., min_length=1)

class ClearRequest(BaseModel):
    session_id: str = Field("default", min_length=1)

# ==============================================
# 工具定义
# ==============================================
def get_current_time():
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def calculate(a: float, b: float, op: str):
    """简单计算器: op = add / sub / mul / div"""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b if b != 0 else "除数不能为0"
    return "不支持的操作"

tool_map = {
    "get_current_time": get_current_time,
    "calculate": calculate
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "进行加减乘除计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]}
                },
                "required": ["a", "b", "op"]
            }
        }
    }
]

# ==============================================
# 核心流式对话接口（支持工具调用 + 严格类型）
# ==============================================
@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id.strip()
    user_message = req.message.strip()
    lock = get_lock(session_id)

    async def event_generator():
        try:
            async with lock:
                session = get_or_create_session(session_id)
                session.add_user_message(user_message)
                messages_copy = session.messages.copy()

            # 1. 第一次调用：判断是否需要工具
            response = await create_chat_completion_with_retry(
                model=OPENAI_MODEL,
                messages=messages_copy,
                tools=tools,
                stream=False,
            )

            choice0 = response.choices[0]
            assistant_msg = choice0.message
            tool_calls = assistant_msg.tool_calls

            print('response========', tool_calls)
            # 2. 如果需要调用工具
            if tool_calls:
                async with lock:
                    session.add_assistant_message(
                        cast(ChatCompletionAssistantMessageParam, assistant_msg.model_dump())
                    )

                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    args_text = tool_call.function.arguments or "{}"
                    try:
                        args = json.loads(args_text)
                    except json.JSONDecodeError:
                        args = {}

                    func = tool_map.get(func_name)
                    if not func:
                        result = f"未找到工具: {func_name}"
                    else:
                        try:
                            result = str(func(**args))
                        except Exception as e:
                            result = f"工具调用失败：{str(e)}"

                    async with lock:
                        session.add_tool_message(tool_call.id, result)

            # 3. 最终流式返回
            async with lock:
                final_messages = session.messages.copy()

            full_answer = ""
            stream = await create_chat_completion_with_retry(
                model=OPENAI_MODEL,
                messages=final_messages,
                stream=True,
            )

            for chunk in stream:
                raw = chunk.choices and chunk.choices[0].delta.content or ""
                c = raw if isinstance(raw, str) else ""
                if c:
                    full_answer += c
                    # EventSourceResponse 会自动按 SSE 格式封装 data 字段。
                    yield c
                    await asyncio.sleep(0.002)

            # 4. 把最终回答写入会话
            if full_answer.strip():
                async with lock:
                    session.add_assistant_message(
                        ChatCompletionAssistantMessageParam(role="assistant", content=full_answer)
                    )

        except (APIConnectionError, APITimeoutError):
            yield {"event": "error", "data": "上游连接不稳定，请稍后重试。"}
        except Exception as e:
            yield {"event": "error", "data": f"服务内部错误：{str(e)}"}

    return EventSourceResponse(event_generator())

# ==============================================
# 清空会话
# ==============================================
@app.post("/api/clear")
async def clear(req: ClearRequest):
    sid = req.session_id.strip()
    if sid in sessions:
        del sessions[sid]
    if sid in session_locks:
        del session_locks[sid]
    return {"status": "cleared", "session_id": sid}