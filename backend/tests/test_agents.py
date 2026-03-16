"""Tests for app.api.v1.agents endpoints."""

from unittest.mock import MagicMock

import pytest

from tests.factories import make_agent, make_mcp_server, make_project, make_skill, make_tool

PROJECT_ID = "proj-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/agents"


def _mock_execute_returns_list(mock_db, items):
    """Configure mock_db.execute to return a scalars().all() list."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    """Configure mock_db.execute to return a scalars().first() value."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_result.scalars.return_value.all.return_value = [item] if item else []
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
async def test_create_agent(client, mock_db):
    """POST /agents - creates an agent with basic fields."""
    project = make_project(id=PROJECT_ID)
    agent = make_agent(project_id=PROJECT_ID, name="worker-1", tools=[], skills=[], mcp_servers=[])

    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "prompt",
            "model",
            "is_planner",
            "config",
            "tools",
            "skills",
            "mcp_servers",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(agent, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={"name": "worker-1", "prompt": "You are helpful."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "worker-1"
    assert data["project_id"] == PROJECT_ID
    assert data["tool_ids"] == []
    assert data["skill_ids"] == []
    assert data["mcp_server_ids"] == []
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_agent_with_relationships(client, mock_db):
    """POST /agents - creates agent with tool_ids, skill_ids, mcp_server_ids."""
    project = make_project(id=PROJECT_ID)
    tool = make_tool(id="tool-1")
    skill = make_skill(id="skill-1")
    mcp = make_mcp_server(id="mcp-1")
    agent = make_agent(
        project_id=PROJECT_ID,
        name="agent-rel",
        tools=[tool],
        skills=[skill],
        mcp_servers=[mcp],
    )

    mock_db.get.return_value = project

    # execute is called for each relationship query (tools, skills, mcp_servers)
    call_count = 0

    async def multi_execute(stmt):
        nonlocal call_count
        call_count += 1
        mock_result = MagicMock()
        if call_count == 1:
            mock_result.scalars.return_value.all.return_value = [tool]
        elif call_count == 2:
            mock_result.scalars.return_value.all.return_value = [skill]
        else:
            mock_result.scalars.return_value.all.return_value = [mcp]
        return mock_result

    mock_db.execute.side_effect = multi_execute

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "prompt",
            "model",
            "is_planner",
            "config",
            "tools",
            "skills",
            "mcp_servers",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(agent, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={
            "name": "agent-rel",
            "tool_ids": ["tool-1"],
            "skill_ids": ["skill-1"],
            "mcp_server_ids": ["mcp-1"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tool_ids"] == ["tool-1"]
    assert data["skill_ids"] == ["skill-1"]
    assert data["mcp_server_ids"] == ["mcp-1"]


@pytest.mark.asyncio
async def test_create_agent_project_not_found(client, mock_db):
    """POST /agents - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/no-such-project/agents",
        json={"name": "x"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_list_agents(client, mock_db):
    """GET /agents - returns list of agents with relationships."""
    project = make_project(id=PROJECT_ID)
    agents = [
        make_agent(project_id=PROJECT_ID, name="a1", tools=[], skills=[], mcp_servers=[]),
        make_agent(project_id=PROJECT_ID, name="a2", tools=[], skills=[], mcp_servers=[]),
    ]

    # First call: db.get for project check; then db.execute for agent list
    mock_db.get.return_value = project

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = agents
    mock_db.execute.return_value = mock_result

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "a1"
    assert data[1]["name"] == "a2"


@pytest.mark.asyncio
async def test_list_agents_empty(client, mock_db):
    """GET /agents - returns empty list when project has no agents."""
    mock_db.get.return_value = make_project(id=PROJECT_ID)
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_agent(client, mock_db):
    """GET /agents/{id} - returns a single agent."""
    agent = make_agent(project_id=PROJECT_ID, name="single", tools=[], skills=[], mcp_servers=[])
    _mock_execute_returns_first(mock_db, agent)

    response = await client.get(f"{BASE_URL}/{agent.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "single"
    assert data["id"] == agent.id


@pytest.mark.asyncio
async def test_get_agent_not_found(client, mock_db):
    """GET /agents/{id} - returns 404 when agent does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Agent not found"


@pytest.mark.asyncio
async def test_update_agent(client, mock_db):
    """PUT /agents/{id} - updates scalar fields."""
    agent = make_agent(project_id=PROJECT_ID, name="old", tools=[], skills=[], mcp_servers=[])

    _mock_execute_returns_first(mock_db, agent)

    async def fake_refresh(obj, **kwargs):
        obj.name = "new"

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"{BASE_URL}/{agent.id}",
        json={"name": "new"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new"
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_agent_with_relationships(client, mock_db):
    """PUT /agents/{id} - updates tool_ids relationship."""
    tool = make_tool(id="new-tool")
    agent = make_agent(project_id=PROJECT_ID, name="a", tools=[], skills=[], mcp_servers=[])

    call_count = 0

    async def multi_execute(stmt):
        nonlocal call_count
        call_count += 1
        mock_result = MagicMock()
        if call_count == 1:
            # First call: find agent
            mock_result.scalars.return_value.first.return_value = agent
        else:
            # Second call: find tools by id
            mock_result.scalars.return_value.all.return_value = [tool]
        return mock_result

    mock_db.execute.side_effect = multi_execute

    async def fake_refresh(obj, **kwargs):
        obj.tools = [tool]

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"{BASE_URL}/{agent.id}",
        json={"tool_ids": ["new-tool"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tool_ids"] == ["new-tool"]


@pytest.mark.asyncio
async def test_update_agent_not_found(client, mock_db):
    """PUT /agents/{id} - returns 404 when agent does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.put(
        f"{BASE_URL}/nonexistent",
        json={"name": "x"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Agent not found"


@pytest.mark.asyncio
async def test_delete_agent(client, mock_db):
    """DELETE /agents/{id} - deletes an agent."""
    agent = make_agent(project_id=PROJECT_ID)
    _mock_execute_returns_first(mock_db, agent)

    response = await client.delete(f"{BASE_URL}/{agent.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_db.delete.assert_called_once_with(agent)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_agent_not_found(client, mock_db):
    """DELETE /agents/{id} - returns 404 when agent does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.delete(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Agent not found"
