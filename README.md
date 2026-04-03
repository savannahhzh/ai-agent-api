# AI Agent API

FastAPI + SSE streaming chat API with in-memory session context and simple function-calling tools.

## Features

- Streaming chat over SSE (`/api/chat`)
- Session-based context (`session_id`)
- Tool calling support:
  - `get_current_time`
  - `calculate`
- Clear session API (`/api/clear`)
- Retry on upstream connection errors (`APIConnectionError` / `APITimeoutError`)

## Requirements

- Python 3.10+
- A valid OpenAI-compatible API key

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn sse-starlette openai python-dotenv pydantic
```

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

## Environment Variables

| Name | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | - | API key |
| `OPENAI_BASE_URL` | No | `https://api.closeai-asia.com/v1` (in code) | OpenAI-compatible base URL |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | Chat model |
| `OPENAI_TIMEOUT_SECONDS` | No | `30` | Request timeout |
| `OPENAI_MAX_RETRIES` | No | `2` | Retry count on connection errors |
| `OPENAI_RETRY_DELAY_SECONDS` | No | `0.8` | Base retry delay |

## Run

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

### 1) Stream chat

`POST /api/chat`

Request body:

```json
{
  "session_id": "user-001",
  "message": "现在几点？"
}
```

- Response type: `text/event-stream`
- Server streams text chunks.
- If upstream connection fails, server emits SSE error event:

```text
event: error
data: 上游连接不稳定，请稍后重试。
```

### 2) Clear session

`POST /api/clear`

Request body:

```json
{
  "session_id": "user-001"
}
```

Response:

```json
{
  "status": "cleared",
  "session_id": "user-001"
}
```

## Quick test with curl

```powershell
curl -N -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"session_id\":\"demo\",\"message\":\"你好，记住我叫小明\"}"
curl -N -X POST "http://127.0.0.1:8000/api/chat" -H "Content-Type: application/json" -d "{\"session_id\":\"demo\",\"message\":\"我叫什么？\"}"
```

## Notes

- Session history is stored in memory; restarting the process clears all sessions.
- For multi-instance production deployment, move session state to Redis or a database.

