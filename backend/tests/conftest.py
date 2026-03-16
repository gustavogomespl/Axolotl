"""Shared fixtures for backend unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.database import get_db


@pytest.fixture
def mock_db():
    """Create a mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.get = AsyncMock()
    return session


@pytest.fixture
def override_db(mock_db):
    """Override the get_db dependency with a mock session."""

    async def _override():
        yield mock_db

    app.dependency_overrides[get_db] = _override
    yield mock_db
    app.dependency_overrides.clear()


@pytest.fixture
async def client(override_db):
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
