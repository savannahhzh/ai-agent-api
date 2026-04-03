### from pydantic import BaseModel, Field
Pydantic 这个库中导入两个核心工具，主要用于 数据校验 + 数据建模（特别是在 FastAPI 里非常常见）。
1. BaseModel = 数据模型的基类,用来定义“接口参数格式”的模板.
2. Field = 用于定义模型字段的函数，可以设置默认值、验证规则、描述信息等。
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="用户名")
    age: int = Field(default=18, gt=0, description="年龄必须大于0")
```
| 参数            | 作用     |
| ------------- | ------ |
| `...`         | 必填     |
| `default=18`  | 默认值    |
| `gt=0`        | 大于0    |
| `lt=100`      | 小于100  |
| `description` | 接口文档说明 |


### asyncio
asyncio 是 Python 的一个库，用于编写异步代码，特别适合处理 I/O 密集型任务（如网络请求、文件操作等）。
它提供了事件循环机制，使得程序可以在等待某些操作完成时继续执行其他任务，从而提高效率。

```python
import asyncio
async def main():
    print("Hello")
    await asyncio.sleep(1)  # 模拟一个异步操作
    print("World")
asyncio.run(main())
```