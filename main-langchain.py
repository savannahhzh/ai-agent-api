from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
import os
import asyncio
import time
from typing import AsyncIterator
from pydantic import BaseModel, Field

# LangGraph 推荐用法
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

app = FastAPI(title="AI Agent API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# 1. 工具定义（用 @tool 装饰器，文档字符串即描述）
# ==============================================

@tool
def get_current_time() -> str:
    """获取当前系统时间，格式为 YYYY-MM-DD HH:MM:SS"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


@tool
def calculate(a: float, b: float, op: str) -> str:
    """
    进行四则运算。
    op 参数说明：add=加, sub=减, mul=乘, div=除
    """
    ops = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,  # 修复原代码 bug：mul 误写为 x + y
        "div": lambda x, y: x / y if y != 0 else None,
    }
    if op not in ops:
        return f"不支持的操作: {op}，支持: add/sub/mul/div"
    result = ops[op](a, b)
    if result is None:
        return "错误：除数不能为 0"
    return str(result)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况"""
    weather_data = {
        "北京": "暴雨，15~26℃，微风",
        "上海": "多云，20~28℃，湿度65%",
        "广州": "小雨，24~30℃，南风3级",
        "深圳": "雷阵雨，25~31℃",
        "杭州": "阴，21~27℃",
    }
    return weather_data.get(city, f"暂无城市【{city}】的天气信息")


tools = [get_current_time, calculate, get_weather]

# ==============================================
# 2. 初始化 Agent（LangGraph，支持持久化会话上下文）
# ==============================================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.1,
    streaming=True,  # 开启真实流式
)

# MemorySaver 按 thread_id 隔离会话上下文，自动携带历史消息
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    # 系统提示
    prompt="你是一个专业AI助手，会思考、调用工具、给出简洁自然的回答。",
)

# ==============================================
# 3. Session 管理（简化，上下文交由 LangGraph 维护）
# ==============================================

# 仅记录 session 存活时间，用于过期清理
_session_timestamps: dict[str, float] = {}
_SESSION_TTL = 3600  # 1 小时无活动则过期

def _touch_session(sid: str):
    _session_timestamps[sid] = time.time()

def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, ts in _session_timestamps.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        _session_timestamps.pop(sid, None)
        # LangGraph MemorySaver 不提供直接删除 API，记录即可
        # 如使用 SqliteSaver 等持久化后端，可在此显式删除

# ==============================================
# 4. 请求体
# ==============================================

class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="会话 ID，相同 ID 共享上下文")
    message: str = Field(..., min_length=1, description="用户消息")

class ClearRequest(BaseModel):
    session_id: str = Field(default="default")

# ==============================================
# 5. 核心接口：真实流式 + Agent
# ==============================================

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id.strip()
    user_msg = req.message.strip()

    _touch_session(sid)
    _cleanup_expired_sessions()

    # LangGraph 用 thread_id 隔离不同会话的上下文
    config = {"configurable": {"thread_id": sid}}

    async def gen() -> AsyncIterator[str]:
        try:
            # astream_events 是真实流式，逐 token 推送
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                # 只取 LLM 最终回答的 token，跳过工具调用的中间输出
                if kind == "on_chat_model_stream":
                    # 过滤掉工具调用阶段（tool_call_chunks 不含 content）
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        yield chunk.content
        except Exception as e:
            yield f"data: [错误] {str(e)}\n\n"

    return EventSourceResponse(gen())


# ==============================================
# 6. 清空会话
# ==============================================

@app.post("/api/clear")
async def clear(req: ClearRequest):
    sid = req.session_id.strip()
    _session_timestamps.pop(sid, None)
    # LangGraph MemorySaver 的 thread 内存会在进程重启后自动清除
    # 如需生产级持久化，替换为 AsyncSqliteSaver 并在此执行 DELETE
    return {"status": "cleared", "session_id": sid}


# ==============================================
# 7. 健康检查
# ==============================================

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_session_timestamps)}