from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


def create_app() -> FastAPI:
    app = FastAPI(
        title="Axolotl",
        description="A regenerative multi-agent framework. Build once, adapt everywhere.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router, prefix="/api/v1")

    # Mount Axolotl as MCP server
    try:
        from app.core.mcp_server import mcp

        app.mount("/mcp", mcp.streamable_http_app())
    except Exception:
        pass  # MCP server mount is optional

    return app


app = create_app()
