from fastapi import APIRouter

from app.api.v1.agents import router as agents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.mcp_servers import router as mcp_servers_router
from app.api.v1.projects import router as projects_router
from app.api.v1.skills import router as skills_router
from app.api.v1.tools import router as tools_router

router = APIRouter()
router.include_router(health_router)
router.include_router(projects_router)
router.include_router(agents_router)
router.include_router(skills_router)
router.include_router(tools_router)
router.include_router(documents_router)
router.include_router(mcp_servers_router)
router.include_router(chat_router)
