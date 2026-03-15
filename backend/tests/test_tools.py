"""Tests for app.api.v1.tools endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_project, make_tool

PROJECT_ID = "proj-tool-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/tools"


def _mock_execute_returns_list(mock_db, items):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
@patch("app.api.v1.tools.ToolRegistry")
@patch("app.api.v1.tools.create_api_tool")
async def test_create_tool_api(mock_create_api, mock_registry, client, mock_db):
    """POST /tools - creates an API tool with api_config and registers it."""
    project = make_project(id=PROJECT_ID)
    tool = make_tool(
        project_id=PROJECT_ID,
        name="my-api-tool",
        description="Fetch data",
        type="api",
        category="general",
    )
    mock_db.get.return_value = project

    mock_tool_fn = MagicMock()
    mock_create_api.return_value = mock_tool_fn

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "type",
            "category",
            "api_config",
            "mcp_server_name",
            "created_at",
        ):
            setattr(obj, attr, getattr(tool, attr))

    mock_db.refresh.side_effect = fake_refresh

    api_config = {
        "name": "my-api-tool",
        "description": "Fetch data",
        "method": "GET",
        "url": "https://example.com/api",
        "headers": {},
        "auth_type": "none",
        "auth_config": {},
    }

    response = await client.post(
        BASE_URL,
        json={
            "name": "my-api-tool",
            "description": "Fetch data",
            "type": "api",
            "category": "general",
            "api_config": api_config,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "my-api-tool"
    assert data["type"] == "api"
    mock_create_api.assert_called_once()
    mock_registry.register.assert_called_once_with(mock_tool_fn, category="general")
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_tool_without_api_config(client, mock_db):
    """POST /tools - creates a tool without api_config (no registration)."""
    project = make_project(id=PROJECT_ID)
    tool = make_tool(
        project_id=PROJECT_ID,
        name="plain-tool",
        description="d",
        type="custom",
        api_config=None,
    )
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "type",
            "category",
            "api_config",
            "mcp_server_name",
            "created_at",
        ):
            setattr(obj, attr, getattr(tool, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={"name": "plain-tool", "description": "d", "type": "custom"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "plain-tool"


@pytest.mark.asyncio
async def test_create_tool_project_not_found(client, mock_db):
    """POST /tools - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/bad-id/tools",
        json={"name": "x", "description": "d"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_list_tools(client, mock_db):
    """GET /tools - returns list of tools."""
    tools = [
        make_tool(project_id=PROJECT_ID, name="t1"),
        make_tool(project_id=PROJECT_ID, name="t2"),
    ]
    _mock_execute_returns_list(mock_db, tools)

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "t1"
    assert data[1]["name"] == "t2"


@pytest.mark.asyncio
async def test_list_tools_empty(client, mock_db):
    """GET /tools - returns empty list."""
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_tool(client, mock_db):
    """GET /tools/{id} - returns a single tool."""
    tool = make_tool(project_id=PROJECT_ID, name="found-tool")
    _mock_execute_returns_first(mock_db, tool)

    response = await client.get(f"{BASE_URL}/{tool.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "found-tool"
    assert data["id"] == tool.id


@pytest.mark.asyncio
async def test_get_tool_not_found(client, mock_db):
    """GET /tools/{id} - returns 404 when tool does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Tool not found"


@pytest.mark.asyncio
@patch("app.api.v1.tools.ToolRegistry")
async def test_delete_tool(mock_registry, client, mock_db):
    """DELETE /tools/{id} - removes tool from registry and DB."""
    tool = make_tool(project_id=PROJECT_ID, name="del-tool")
    _mock_execute_returns_first(mock_db, tool)

    response = await client.delete(f"{BASE_URL}/{tool.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_registry.remove.assert_called_once_with("del-tool")
    mock_db.delete.assert_called_once_with(tool)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_tool_not_found(client, mock_db):
    """DELETE /tools/{id} - returns 404 when tool does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.delete(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Tool not found"


@pytest.mark.asyncio
@patch("app.api.v1.tools.ToolRegistry")
async def test_test_tool_success(mock_registry, client, mock_db):
    """POST /tools/{id}/test - invokes the tool and returns result."""
    tool = make_tool(project_id=PROJECT_ID, name="testable-tool")
    _mock_execute_returns_first(mock_db, tool)

    mock_tool_instance = AsyncMock()
    mock_tool_instance.ainvoke.return_value = "tool output"
    mock_registry.get_tools.return_value = [mock_tool_instance]

    response = await client.post(
        f"{BASE_URL}/{tool.id}/test",
        json={"input": {"query": "hello"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "tool output"
    mock_registry.get_tools.assert_called_once_with(["testable-tool"])
    mock_tool_instance.ainvoke.assert_awaited_once_with({"query": "hello"})


@pytest.mark.asyncio
@patch("app.api.v1.tools.ToolRegistry")
async def test_test_tool_not_in_registry(mock_registry, client, mock_db):
    """POST /tools/{id}/test - returns 400 when tool is not loaded."""
    tool = make_tool(project_id=PROJECT_ID, name="unloaded")
    _mock_execute_returns_first(mock_db, tool)

    mock_registry.get_tools.return_value = []

    response = await client.post(
        f"{BASE_URL}/{tool.id}/test",
        json={"input": {}},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Tool not loaded in registry"


@pytest.mark.asyncio
async def test_test_tool_not_found(client, mock_db):
    """POST /tools/{id}/test - returns 404 when tool does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.post(
        f"{BASE_URL}/nonexistent/test",
        json={"input": {}},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Tool not found"


@pytest.mark.asyncio
@patch("app.api.v1.tools.ToolRegistry")
async def test_test_tool_invocation_error(mock_registry, client, mock_db):
    """POST /tools/{id}/test - returns 500 when tool invocation raises."""
    tool = make_tool(project_id=PROJECT_ID, name="broken-tool")
    _mock_execute_returns_first(mock_db, tool)

    mock_tool_instance = AsyncMock()
    mock_tool_instance.ainvoke.side_effect = RuntimeError("connection failed")
    mock_registry.get_tools.return_value = [mock_tool_instance]

    response = await client.post(
        f"{BASE_URL}/{tool.id}/test",
        json={"input": {"q": "x"}},
    )
    assert response.status_code == 500
    assert "connection failed" in response.json()["detail"]
