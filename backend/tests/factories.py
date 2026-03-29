"""Factory functions for creating mock model objects in tests."""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock


def make_project(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "name": "test-project",
        "description": "A test project",
        "planner_prompt": "You are a planner.",
        "model": "openai:gpt-4.1-mini",
        "orchestration_mode": "supervisor",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_agent(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "name": "test-agent",
        "description": "A test agent",
        "prompt": "You are helpful.",
        "model": None,
        "is_planner": False,
        "config": None,
        "tools": [],
        "skills": [],
        "mcp_servers": [],
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_skill(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": None,
        "name": "test-skill",
        "description": "A test skill",
        "type": "rag",
        "collection_name": "test-collection",
        "tool_names": None,
        "subgraph_name": None,
        "system_prompt": None,
        "is_active": True,
        "config": None,
        "agents": [],
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_tool(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": None,
        "name": "test-tool",
        "description": "A test tool",
        "type": "api",
        "category": "general",
        "api_config": {
            "name": "test-tool",
            "description": "A test tool",
            "method": "GET",
            "url": "https://example.com",
            "headers": {},
            "auth_type": "none",
            "auth_config": {},
        },
        "mcp_server_name": None,
        "agents": [],
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_mcp_server(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": None,
        "name": "test-mcp",
        "transport": "streamable_http",
        "url": "http://localhost:3000",
        "command": None,
        "args": None,
        "env": None,
        "status": "disconnected",
        "agents": [],
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_conversation(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "phone_number": "+5511999999999",
        "created_at": datetime.now(UTC),
        "expires_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_message(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "role": "user",
        "content": "Hello",
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


def make_document(**kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "project_id": None,
        "filename": "test.pdf",
        "collection_name": "docs",
        "content_type": "application/pdf",
        "chunk_count": 5,
        "status": "ready",
        "error_message": None,
        "created_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj
