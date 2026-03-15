"""Tests for app.api.v1.documents endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.factories import make_document, make_project

PROJECT_ID = "proj-doc-123"
BASE_URL = f"/api/v1/projects/{PROJECT_ID}/documents"


def _mock_execute_returns_list(mock_db, items):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = items
    mock_db.execute.return_value = mock_result


def _mock_execute_returns_first(mock_db, item):
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = item
    mock_db.execute.return_value = mock_result


@pytest.mark.asyncio
@patch("app.api.v1.documents._get_doc_service")
async def test_upload_document(mock_get_svc, client, mock_db):
    """POST /documents - uploads a file and ingests it."""
    project = make_project(id=PROJECT_ID)
    doc = make_document(
        project_id=PROJECT_ID,
        filename="test.pdf",
        collection_name="docs",
        content_type="application/pdf",
        chunk_count=10,
        status="ready",
    )
    mock_db.get.return_value = project

    refresh_count = 0

    async def fake_refresh(obj, **kwargs):
        nonlocal refresh_count
        refresh_count += 1
        for attr in (
            "id",
            "project_id",
            "filename",
            "collection_name",
            "content_type",
            "chunk_count",
            "status",
            "error_message",
            "created_at",
        ):
            setattr(obj, attr, getattr(doc, attr))

    mock_db.refresh.side_effect = fake_refresh

    mock_svc = AsyncMock()
    mock_svc.ingest.return_value = {"chunk_count": 10}
    mock_get_svc.return_value = mock_svc

    response = await client.post(
        BASE_URL,
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        data={"collection": "docs", "metadata": "{}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["collection_name"] == "docs"
    assert data["status"] == "ready"
    mock_db.add.assert_called_once()
    mock_svc.ingest.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.api.v1.documents._get_doc_service")
async def test_upload_document_ingest_error(mock_get_svc, client, mock_db):
    """POST /documents - marks status as error when ingestion fails."""
    project = make_project(id=PROJECT_ID)
    doc = make_document(
        project_id=PROJECT_ID,
        status="error",
        error_message="parse failed",
    )
    mock_db.get.return_value = project

    async def fake_refresh(obj, **kwargs):
        for attr in (
            "id",
            "project_id",
            "filename",
            "collection_name",
            "content_type",
            "chunk_count",
            "status",
            "error_message",
            "created_at",
        ):
            setattr(obj, attr, getattr(doc, attr))

    mock_db.refresh.side_effect = fake_refresh

    mock_svc = AsyncMock()
    mock_svc.ingest.side_effect = RuntimeError("parse failed")
    mock_get_svc.return_value = mock_svc

    response = await client.post(
        BASE_URL,
        files={"file": ("test.pdf", b"bad", "application/pdf")},
        data={"collection": "docs", "metadata": "{}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["error_message"] == "parse failed"


@pytest.mark.asyncio
async def test_upload_document_project_not_found(client, mock_db):
    """POST /documents - returns 404 when project does not exist."""
    mock_db.get.return_value = None

    response = await client.post(
        "/api/v1/projects/bad-id/documents",
        files={"file": ("f.txt", b"data", "text/plain")},
        data={"collection": "c", "metadata": "{}"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


@pytest.mark.asyncio
@patch("app.api.v1.documents._get_doc_service")
async def test_upload_document_invalid_metadata(mock_get_svc, client, mock_db):
    """POST /documents - returns 400 for invalid metadata JSON."""
    mock_db.get.return_value = make_project(id=PROJECT_ID)

    response = await client.post(
        BASE_URL,
        files={"file": ("f.txt", b"data", "text/plain")},
        data={"collection": "c", "metadata": "not-json{"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid metadata JSON"


@pytest.mark.asyncio
async def test_list_documents(client, mock_db):
    """GET /documents - returns list of documents."""
    docs = [
        make_document(project_id=PROJECT_ID, filename="a.pdf"),
        make_document(project_id=PROJECT_ID, filename="b.pdf"),
    ]
    _mock_execute_returns_list(mock_db, docs)

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["filename"] == "a.pdf"
    assert data[1]["filename"] == "b.pdf"


@pytest.mark.asyncio
async def test_list_documents_filter_by_collection(client, mock_db):
    """GET /documents?collection=X - filters by collection name."""
    doc = make_document(project_id=PROJECT_ID, collection_name="specific")
    _mock_execute_returns_list(mock_db, [doc])

    response = await client.get(f"{BASE_URL}?collection=specific")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["collection_name"] == "specific"


@pytest.mark.asyncio
async def test_list_documents_empty(client, mock_db):
    """GET /documents - returns empty list."""
    _mock_execute_returns_list(mock_db, [])

    response = await client.get(BASE_URL)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_document(client, mock_db):
    """GET /documents/{id} - returns a single document."""
    doc = make_document(project_id=PROJECT_ID, filename="found.pdf")
    _mock_execute_returns_first(mock_db, doc)

    response = await client.get(f"{BASE_URL}/{doc.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "found.pdf"
    assert data["id"] == doc.id


@pytest.mark.asyncio
async def test_get_document_not_found(client, mock_db):
    """GET /documents/{id} - returns 404 when document does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.get(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


@pytest.mark.asyncio
@patch("app.api.v1.documents._get_doc_service")
async def test_delete_document(mock_get_svc, client, mock_db):
    """DELETE /documents/{id} - deletes document from DB and vector store."""
    doc = make_document(project_id=PROJECT_ID, collection_name="docs")
    _mock_execute_returns_first(mock_db, doc)

    mock_svc = AsyncMock()
    mock_get_svc.return_value = mock_svc

    response = await client.delete(f"{BASE_URL}/{doc.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    mock_svc.delete.assert_awaited_once_with("docs", doc.id)
    mock_db.delete.assert_called_once_with(doc)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
@patch("app.api.v1.documents._get_doc_service")
async def test_delete_document_vector_store_error(mock_get_svc, client, mock_db):
    """DELETE /documents/{id} - still deletes from DB even if vector store fails."""
    doc = make_document(project_id=PROJECT_ID)
    _mock_execute_returns_first(mock_db, doc)

    mock_svc = AsyncMock()
    mock_svc.delete.side_effect = RuntimeError("vector store down")
    mock_get_svc.return_value = mock_svc

    response = await client.delete(f"{BASE_URL}/{doc.id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}
    # DB delete still happens despite vector store error
    mock_db.delete.assert_called_once_with(doc)
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_document_not_found(client, mock_db):
    """DELETE /documents/{id} - returns 404 when document does not exist."""
    _mock_execute_returns_first(mock_db, None)

    response = await client.delete(f"{BASE_URL}/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"
