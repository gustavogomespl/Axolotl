import httpx
import pytest

API_BASE = "http://localhost:8000/api/v1"


@pytest.fixture
def axolotl_client():
    """HTTP client for calling the Axolotl backend."""
    return httpx.AsyncClient(base_url=API_BASE)
