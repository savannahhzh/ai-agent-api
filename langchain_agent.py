import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentExecutor

from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
import time

load_dotenv()

# 1. 初始化大模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)

# 2. 定义工具（和你之前一模一样）
def get_current_time():
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def calculate(a: float, b: float, op: str):
    """计算 a op b，op 支持 add/sub/mul/div"""
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div": return a / b if b != 0 else "除数不能为0"
    return "不支持的操作"

def get_weather(city: str):
    """查询城市天气"""
    weather = {
        "北京": "晴 15~26℃",
        "上海": "多云 20~28℃",
        "广州": "小雨 24~30℃",
        "深圳": "雷阵雨 25~31℃"
    }
    return weather.get(city, f"暂无{city}天气")

# 3. 包装成 LangChain 标准工具
tools = [
    StructuredTool.from_function(get_current_time),
    StructuredTool.from_function(calculate),
    StructuredTool.from_function(get_weather)
]

# 4. ReAct 固定模板
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. 
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought:
""")

# 5. 创建 Agent
agent = create_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印思考过程（核心！）
    handle_parsing_errors=True
)

# 6. 测试
if __name__ == "__main__":
    questions = [
        "现在几点了？",
        "北京天气怎么样？",
        "33+57等于多少",
        "上海温度比北京高几度？"
    ]

    for q in questions:
        print("=" * 50)
        print("问题：", q)
        res = agent_executor.invoke({"input": q})
        print("答案：", res["output"])