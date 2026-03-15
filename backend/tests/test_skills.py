"""Tests for app.api.v1.skills endpoints."""

from unittest.mock import MagicMock

import pytest

from tests.factories import make_project, make_skill

PROJECT_ID = "proj-skill-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/skills"


def _mock_execute_returns_list(mock_db, items):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
async def test_create_skill(client, mock_db):
    """POST /skills - creates a skill and returns its data."""
    project = make_project(id=PROJECT_ID)
    skill = make_skill(
        project_id=PROJECT_ID,
        name="rag-skill",
        description="A RAG skill",
        type="rag",
        collection_name="skill_rag_skill",
    )
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "type",
            "collection_name",
            "tool_names",
            "subgraph_name",
            "system_prompt",
            "is_active",
            "config",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(skill, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={"name": "rag-skill", "description": "A RAG skill", "type": "rag"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "rag-skill"
    assert data["type"] == "rag"
    assert data["project_id"] == PROJECT_ID
    # RAG type without collection_name auto-generates one
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_skill_with_collection(client, mock_db):
    """POST /skills - creates skill with explicit collection_name."""
    project = make_project(id=PROJECT_ID)
    skill = make_skill(
        project_id=PROJECT_ID,
        name="custom",
        description="d",
        type="rag",
        collection_name="my-collection",
    )
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "name",
            "description",
            "type",
            "collection_name",
            "tool_names",
            "subgraph_name",
            "system_prompt",
            "is_active",
            "config",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(skill, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        BASE_URL,
        json={
            "name": "custom",
            "description": "d",
            "type": "rag",
            "collection_name": "my-collection",
        },
    )
    assert response.status_code == 200
    assert response.json()["collection_name"] == "my-collection"


@pytest.mark.asyncio
async def test_create_skill_project_not_found(client, mock_db):
    """POST /skills - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/bad-id/skills",
        json={"name": "x", "description": "d", "type": "rag"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_list_skills(client, mock_db):
    """GET /skills - returns list of skills ordered by created_at."""
    skills = [
        make_skill(project_id=PROJECT_ID, name="s1"),
        make_skill(project_id=PROJECT_ID, name="s2"),
    ]
    _mock_execute_returns_list(mock_db, skills)

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "s1"
    assert data[1]["name"] == "s2"


@pytest.mark.asyncio
async def test_list_skills_empty(client, mock_db):
    """GET /skills - returns empty list."""
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_skill(client, mock_db):
    """GET /skills/{id} - returns a single skill."""
    skill = make_skill(project_id=PROJECT_ID, name="found")
    _mock_execute_returns_first(mock_db, skill)

    response = await client.get(f"{BASE_URL}/{skill.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "found"
    assert data["id"] == skill.id


@pytest.mark.asyncio
async def test_get_skill_not_found(client, mock_db):
    """GET /skills/{id} - returns 404 when skill does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"


@pytest.mark.asyncio
async def test_update_skill(client, mock_db):
    """PUT /skills/{id} - updates skill fields."""
    skill = make_skill(project_id=PROJECT_ID, name="old-name")
    _mock_execute_returns_first(mock_db, skill)

    async def fake_refresh(obj, **kwargs):
        obj.name = "new-name"

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"{BASE_URL}/{skill.id}",
        json={"name": "new-name"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-name"
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_skill_partial(client, mock_db):
    """PUT /skills/{id} - partial update only touches specified fields."""
    skill = make_skill(project_id=PROJECT_ID, name="keep", description="old")
    _mock_execute_returns_first(mock_db, skill)

    async def fake_refresh(obj, **kwargs):
        obj.description = "new"

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"{BASE_URL}/{skill.id}",
        json={"description": "new"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "keep"
    assert data["description"] == "new"


@pytest.mark.asyncio
async def test_update_skill_not_found(client, mock_db):
    """PUT /skills/{id} - returns 404 when skill does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.put(
        f"{BASE_URL}/nonexistent",
        json={"name": "x"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"


@pytest.mark.asyncio
async def test_delete_skill(client, mock_db):
    """DELETE /skills/{id} - deletes a skill."""
    skill = make_skill(project_id=PROJECT_ID)
    _mock_execute_returns_first(mock_db, skill)

    response = await client.delete(f"{BASE_URL}/{skill.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_db.delete.assert_called_once_with(skill)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_skill_not_found(client, mock_db):
    """DELETE /skills/{id} - returns 404 when skill does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.delete(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"


@pytest.mark.asyncio
async def test_toggle_skill_activate(client, mock_db):
    """POST /skills/{id}/activate - toggles is_active from True to False."""
    skill = make_skill(project_id=PROJECT_ID, is_active=True)
    _mock_execute_returns_first(mock_db, skill)

    response = await client.post(f"{BASE_URL}/{skill.id}/activate")
    assert response.status_code == 200
    # Skill was active, toggling sets it to False
    assert response.json()["is_active"] is False
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_toggle_skill_deactivate(client, mock_db):
    """POST /skills/{id}/activate - toggles is_active from False to True."""
    skill = make_skill(project_id=PROJECT_ID, is_active=False)
    _mock_execute_returns_first(mock_db, skill)

    response = await client.post(f"{BASE_URL}/{skill.id}/activate")
    assert response.status_code == 200
    assert response.json()["is_active"] is True


@pytest.mark.asyncio
async def test_toggle_skill_not_found(client, mock_db):
    """POST /skills/{id}/activate - returns 404 when skill does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.post(f"{BASE_URL}/nonexistent/activate")
    assert response.status_code == 404
    assert response.json()["detail"] == "Skill not found"
