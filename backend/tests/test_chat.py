"""Tests for app.api.v1.chat endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.orchestrator import OrchestratorResult
from tests.factories import (
    make_agent,
    make_conversation,
    make_message,
    make_project,
)

PROJECT_ID = "proj-chat-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/chat"


def _make_orch_result(content, todo_list=None, completed_tasks=None, files=None):
    return OrchestratorResult(
        content=content,
        state={
            "todo_list": todo_list or [],
            "completed_tasks": completed_tasks or [],
            "files": files or [],
        },
    )


def _mock_execute_returns_list(mock_db, items):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_basic(mock_redis, mock_orchestrate, client, mock_db):
    """POST /chat - sends a message and receives a response."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    agents = [make_agent(project_id=PROJECT_ID, tools=[], skills=[], mcp_servers=[])]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = agents
    mock_db.execute.return_value = mock_result

    mock_redis.get_saver = AsyncMock(return_value=MagicMock())
    mock_orchestrate.return_value = _make_orch_result("Hello! How can I help?")

    response = await client.post(
        BASE_URL,
        json={"message": "Hi there"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello! How can I help?"
    assert "thread_id" in data
    mock_orchestrate.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_with_thread_id(mock_redis, mock_orchestrate, client, mock_db):
    """POST /chat - uses provided thread_id for conversation continuity."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    _mock_execute_returns_list(mock_db, [])
    mock_redis.get_saver = AsyncMock(return_value=MagicMock())
    mock_orchestrate.return_value = _make_orch_result("continued reply")

    response = await client.post(
        BASE_URL,
        json={"message": "follow up", "thread_id": "existing-thread-123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"] == "existing-thread-123"
    assert data["response"] == "continued reply"


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_with_phone_number(mock_redis, mock_orchestrate, client, mock_db):
    """POST /chat - persists conversation when phone_number is provided."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    _mock_execute_returns_list(mock_db, [])
    mock_redis.get_saver = AsyncMock(return_value=MagicMock())
    mock_orchestrate.return_value = _make_orch_result("response to phone user")

    response = await client.post(
        BASE_URL,
        json={"message": "hello", "phone_number": "+5511999999999"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "response to phone user"
    # Should have called add for Conversation + 2 Messages (user + assistant)
    assert mock_db.add.call_count >= 3
    mock_db.flush.assert_awaited()
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_without_phone_number_no_persistence(
    mock_redis, mock_orchestrate, client, mock_db
):
    """POST /chat - does not persist conversation without phone_number."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    _mock_execute_returns_list(mock_db, [])
    mock_redis.get_saver = AsyncMock(return_value=MagicMock())
    mock_orchestrate.return_value = _make_orch_result("ephemeral reply")

    response = await client.post(
        BASE_URL,
        json={"message": "hello"},
    )
    assert response.status_code == 200
    # Without phone_number: no Conversation/Message added (only the execute call for agents)
    # db.add should not be called (no conversation persistence)
    mock_db.add.assert_not_called()
    mock_db.flush.assert_not_awaited()


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_redis_unavailable(mock_redis, mock_orchestrate, client, mock_db):
    """POST /chat - works even when Redis is unavailable (checkpointer=None)."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    _mock_execute_returns_list(mock_db, [])
    mock_redis.get_saver = AsyncMock(side_effect=ConnectionError("redis down"))
    mock_orchestrate.return_value = _make_orch_result("still works")

    response = await client.post(
        BASE_URL,
        json={"message": "hi"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "still works"
    # orchestrate should be called with checkpointer=None
    call_kwargs = mock_orchestrate.call_args
    assert (
        call_kwargs.kwargs.get("checkpointer") is None or call_kwargs[1].get("checkpointer") is None
    )


@pytest.mark.asyncio
@patch("app.api.v1.chat.orchestrate_project_chat")
@patch("app.api.v1.chat.redis_manager")
async def test_chat_with_model_override(mock_redis, mock_orchestrate, client, mock_db):
    """POST /chat - passes model override to orchestrator."""
    project = make_project(id=PROJECT_ID)
    mock_db.get.return_value = project

    _mock_execute_returns_list(mock_db, [])
    mock_redis.get_saver = AsyncMock(return_value=MagicMock())
    mock_orchestrate.return_value = _make_orch_result("override reply")

    response = await client.post(
        BASE_URL,
        json={"message": "hi", "model": "anthropic:claude-sonnet-4-20250514"},
    )
    assert response.status_code == 200
    call_kwargs = mock_orchestrate.call_args
    assert (
        call_kwargs.kwargs.get("model_override") == "anthropic:claude-sonnet-4-20250514"
        or call_kwargs[1].get("model_override") == "anthropic:claude-sonnet-4-20250514"
    )


@pytest.mark.asyncio
async def test_chat_project_not_found(client, mock_db):
    """POST /chat - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/bad-id/chat",
        json={"message": "hello"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_list_conversations(client, mock_db):
    """GET /chat/conversations - returns list of conversations."""
    convos = [
        make_conversation(project_id=PROJECT_ID, phone_number="+5511111111111"),
        make_conversation(project_id=PROJECT_ID, phone_number="+5522222222222"),
    ]
    _mock_execute_returns_list(mock_db, convos)

    response = await client.get(f"{BASE_URL}/conversations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["phone_number"] == "+5511111111111"
    assert data[1]["phone_number"] == "+5522222222222"


@pytest.mark.asyncio
async def test_list_conversations_filter_by_phone(client, mock_db):
    """GET /chat/conversations?phone_number=X - filters by phone number."""
    convo = make_conversation(project_id=PROJECT_ID, phone_number="+5511999999999")
    _mock_execute_returns_list(mock_db, [convo])

    response = await client.get(f"{BASE_URL}/conversations?phone_number=%2B5511999999999")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["phone_number"] == "+5511999999999"


@pytest.mark.asyncio
async def test_list_conversations_empty(client, mock_db):
    """GET /chat/conversations - returns empty list."""
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(f"{BASE_URL}/conversations")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_conversation_messages(client, mock_db):
    """GET /chat/conversations/{id}/messages - returns messages for a conversation."""
    convo = make_conversation(project_id=PROJECT_ID)
    msg1 = make_message(conversation_id=convo.id, role="user", content="Hi")
    msg2 = make_message(conversation_id=convo.id, role="assistant", content="Hello!")
    convo.messages = [msg1, msg2]

    _mock_execute_returns_first(mock_db, convo)

    response = await client.get(f"{BASE_URL}/conversations/{convo.id}/messages")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[0]["content"] == "Hi"
    assert data[1]["role"] == "assistant"
    assert data[1]["content"] == "Hello!"
    # Each message has an id and created_at
    assert "id" in data[0]
    assert "created_at" in data[0]


@pytest.mark.asyncio
async def test_get_conversation_messages_empty(client, mock_db):
    """GET /chat/conversations/{id}/messages - returns empty list when no messages."""
    convo = make_conversation(project_id=PROJECT_ID)
    convo.messages = []

    _mock_execute_returns_first(mock_db, convo)

    response = await client.get(f"{BASE_URL}/conversations/{convo.id}/messages")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_conversation_messages_not_found(client, mock_db):
    """GET /chat/conversations/{id}/messages - returns 404 for nonexistent conversation."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/conversations/nonexistent/messages")
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found"
