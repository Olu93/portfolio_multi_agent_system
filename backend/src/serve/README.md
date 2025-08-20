# A2A Chat Service

A FastAPI service that provides a `/chat` endpoint for communicating with A2A agents using the `python-a2a` client library.

## Features

- **Chat Endpoint**: `/chat` POST endpoint for sending messages to A2A agents
- **A2A Integration**: Uses `python-a2a` client library for agent communication
- **Task Management**: Handles both direct responses and long-running tasks
- **CORS Support**: Configured for frontend integration
- **Health Checks**: Built-in health monitoring endpoints

## Environment Configuration

Set the following environment variables:

```bash
# A2A Agent Configuration
export A2A_AGENT_URL=http://localhost:41241
export A2A_AGENT_NAME=supervisor_agent

# FastAPI Service Configuration
export PORT=8000
export HOST=0.0.0.0
```

## Installation

1. Install dependencies:
```bash
cd backend
poetry install
```

2. Run the service:
```bash
cd src/serve
python run.py
```

Or using uvicorn directly:
```bash
uvicorn run:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### GET `/`
Health check and service information.

### GET `/health`
Detailed health check endpoint.

### POST `/chat`
Send a message to the A2A agent.

**Request Body:**
```json
{
  "message": "Hello, agent!",
  "conversation_id": "optional_conversation_id",
  "user_id": "optional_user_id"
}
```

**Response:**
```json
{
  "response": "Agent response text",
  "conversation_id": "conv_123",
  "task_id": "task_456",
  "status": "success",
  "error": null
}
```

### GET `/conversations/{conversation_id}`
Get conversation history (placeholder endpoint).

## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you help me with?"}'
```

### Using Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hello, agent!"}
)
print(response.json())
```

## Error Handling

The service handles various error scenarios:
- A2A agent connection failures
- Message sending errors
- Task processing errors
- Unexpected exceptions

All errors are logged and returned with appropriate HTTP status codes.

## Development

- **Auto-reload**: Enabled for development
- **Logging**: Configured with info level
- **CORS**: Configured for frontend development
- **Type Hints**: Full type annotation support

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `python-a2a`: A2A client library
- `pydantic`: Data validation
- `python-dotenv`: Environment variable management
