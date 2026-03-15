"""Tests for app.api.v1.mcp_servers endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_mcp_server, make_project

PROJECT_ID = "proj-mcp-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/mcp-servers"


def _mock_execute_returns_list(mock_db, items):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_create_mcp_server(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers - creates and registers an MCP server."""
    project = make_project(id=PROJECT_ID)
    server = make_mcp_server(
        project_id=PROJECT_ID,
        name="my-mcp",
        transport="streamable_http",
        url="http://localhost:3000",
    )
    mock_db.get.return_value = project
    mock_mcp_mgr.add_server = AsyncMock()

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "transport",
            "url",
            "command",
            "args",
            "env",
            "status",
            "created_at",
        ):
            setattr(obj, attr, getattr(server, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={
            "name": "my-mcp",
            "transport": "streamable_http",
            "url": "http://localhost:3000",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "my-mcp"
    assert data["transport"] == "streamable_http"
    assert data["url"] == "http://localhost:3000"
    assert data["project_id"] == PROJECT_ID
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    mock_mcp_mgr.add_server.assert_awaited_once_with(
        name="my-mcp",
        url="http://localhost:3000",
        command=None,
        args=None,
        env=None,
    )


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_create_mcp_server_stdio(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers - creates a stdio transport MCP server."""
    project = make_project(id=PROJECT_ID)
    server = make_mcp_server(
        project_id=PROJECT_ID,
        name="stdio-mcp",
        transport="stdio",
        url=None,
        command="npx",
        args=["-y", "server"],
    )
    mock_db.get.return_value = project
    mock_mcp_mgr.add_server = AsyncMock()

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "transport",
            "url",
            "command",
            "args",
            "env",
            "status",
            "created_at",
        ):
            setattr(obj, attr, getattr(server, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={
            "name": "stdio-mcp",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "server"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "stdio-mcp"
    assert data["transport"] == "stdio"


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_create_mcp_server_project_not_found(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/bad-id/mcp-servers",
        json={"name": "x", "url": "http://x"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_list_mcp_servers(client, mock_db):
    """GET /mcp-servers - returns list of MCP servers."""
    servers = [
        make_mcp_server(project_id=PROJECT_ID, name="mcp1"),
        make_mcp_server(project_id=PROJECT_ID, name="mcp2"),
    ]
    _mock_execute_returns_list(mock_db, servers)

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "mcp1"
    assert data[1]["name"] == "mcp2"


@pytest.mark.asyncio
async def test_list_mcp_servers_empty(client, mock_db):
    """GET /mcp-servers - returns empty list."""
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_mcp_server(client, mock_db):
    """GET /mcp-servers/{id} - returns a single MCP server."""
    server = make_mcp_server(project_id=PROJECT_ID, name="found-mcp")
    _mock_execute_returns_first(mock_db, server)

    response = await client.get(f"{BASE_URL}/{server.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "found-mcp"
    assert data["id"] == server.id


@pytest.mark.asyncio
async def test_get_mcp_server_not_found(client, mock_db):
    """GET /mcp-servers/{id} - returns 404 when server does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "MCP server not found"


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_delete_mcp_server(mock_mcp_mgr, client, mock_db):
    """DELETE /mcp-servers/{id} - removes server from manager and DB."""
    server = make_mcp_server(project_id=PROJECT_ID, name="del-mcp")
    _mock_execute_returns_first(mock_db, server)
    mock_mcp_mgr.remove_server = AsyncMock(return_value=True)

    response = await client.delete(f"{BASE_URL}/{server.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_mcp_mgr.remove_server.assert_awaited_once_with("del-mcp")
    mock_db.delete.assert_called_once_with(server)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_mcp_server_not_found(client, mock_db):
    """DELETE /mcp-servers/{id} - returns 404 when server does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.delete(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "MCP server not found"


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_refresh_mcp_server_success(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers/{id}/refresh - connects and loads tools."""
    server = make_mcp_server(project_id=PROJECT_ID, name="refresh-mcp")
    _mock_execute_returns_first(mock_db, server)

    mock_tool_1 = MagicMock()
    mock_tool_1.name = "tool-a"
    mock_tool_2 = MagicMock()
    mock_tool_2.name = "tool-b"
    mock_mcp_mgr.connect_and_load_tools = AsyncMock(return_value=[mock_tool_1, mock_tool_2])

    response = await client.post(f"{BASE_URL}/{server.id}/refresh")
    assert response.status_code == 200
    data = response.json()
    assert data["tools_loaded"] == 2
    assert set(data["tool_names"]) == {"tool-a", "tool-b"}
    mock_mcp_mgr.connect_and_load_tools.assert_awaited_once_with("refresh-mcp")
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_refresh_mcp_server_connection_error(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers/{id}/refresh - returns 500 on connection failure."""
    server = make_mcp_server(project_id=PROJECT_ID, name="fail-mcp")
    _mock_execute_returns_first(mock_db, server)

    mock_mcp_mgr.connect_and_load_tools = AsyncMock(side_effect=ConnectionError("refused"))

    response = await client.post(f"{BASE_URL}/{server.id}/refresh")
    assert response.status_code == 500
    assert "refused" in response.json()["detail"]
    mock_db.commit.assert_awaited()


@pytest.mark.asyncio
@patch("app.api.v1.mcp_servers.mcp_manager")
async def test_refresh_mcp_server_not_found(mock_mcp_mgr, client, mock_db):
    """POST /mcp-servers/{id}/refresh - returns 404 when server does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.post(f"{BASE_URL}/nonexistent/refresh")
    assert response.status_code == 404
    assert response.json()["detail"] == "MCP server not found"
