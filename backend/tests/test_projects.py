"""Tests for app.api.v1.projects endpoints."""

from unittest.mock import MagicMock

import pytest

from tests.factories import make_project


@pytest.mark.asyncio
async def test_create_project(client, mock_db):
    """POST /api/v1/projects - creates a project and returns its data."""
    project = make_project(name="new-project", description="desc")

    # db.refresh populates the mock object returned after add+commit
    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "name",
            "description",
            "planner_prompt",
            "model",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(project, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        "/api/v1/projects",
        json={"name": "new-project", "description": "desc"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-project"
    assert data["description"] == "desc"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    mock_db.refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_project_minimal(client, mock_db):
    """POST /api/v1/projects - only required field (name)."""
    project = make_project(name="minimal", description=None, planner_prompt=None, model=None)

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "name",
            "description",
            "planner_prompt",
            "model",
            "created_at",
            "updated_at",
        ):
            setattr(obj, attr, getattr(project, attr))

    mock_db.refresh.side_effect = fake_refresh

    response = await client.post(
        "/api/v1/projects",
        json={"name": "minimal"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "minimal"
    assert data["description"] is None


@pytest.mark.asyncio
async def test_list_projects(client, mock_db):
    """GET /api/v1/projects - returns list of projects."""
    projects = [make_project(name="p1"), make_project(name="p2")]

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = projects
    mock_db.execute.return_value = mock_result

    response = await client.get("/api/v1/projects")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "p1"
    assert data[1]["name"] == "p2"


@pytest.mark.asyncio
async def test_list_projects_empty(client, mock_db):
    """GET /api/v1/projects - returns empty list when no projects exist."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    response = await client.get("/api/v1/projects")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_project(client, mock_db):
    """GET /api/v1/projects/{id} - returns a single project."""
    project = make_project(name="found")
    mock_db.get.return_value = project

    response = await client.get(f"/api/v1/projects/{project.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == project.id
    assert data["name"] == "found"


@pytest.mark.asyncio
async def test_get_project_not_found(client, mock_db):
    """GET /api/v1/projects/{id} - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.get("/api/v1/projects/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_update_project(client, mock_db):
    """PUT /api/v1/projects/{id} - updates project fields."""
    project = make_project(name="old-name")
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        obj.name = "new-name"

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"/api/v1/projects/{project.id}",
        json={"name": "new-name"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-name"
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_project_partial(client, mock_db):
    """PUT /api/v1/projects/{id} - partial update only changes specified fields."""
    project = make_project(name="keep-name", description="old-desc")
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        obj.description = "new-desc"

    mock_db.refresh.side_effect = fake_refresh

    response = await client.put(
        f"/api/v1/projects/{project.id}",
        json={"description": "new-desc"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "keep-name"
    assert data["description"] == "new-desc"


@pytest.mark.asyncio
async def test_update_project_not_found(client, mock_db):
    """PUT /api/v1/projects/{id} - returns 404 for nonexistent project."""
    mock_db.get.return_value = None

    response = await client.put(
        "/api/v1/projects/nonexistent-id",
        json={"name": "x"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
async def test_delete_project(client, mock_db):
    """DELETE /api/v1/projects/{id} - deletes a project."""
    project = make_project()
    mock_db.get.return_value = project

    response = await client.delete(f"/api/v1/projects/{project.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_db.delete.assert_called_once_with(project)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_project_not_found(client, mock_db):
    """DELETE /api/v1/projects/{id} - returns 404 for nonexistent project."""
    mock_db.get.return_value = None

    response = await client.delete("/api/v1/projects/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"
