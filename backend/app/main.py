from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging
    import subprocess
    from pathlib import Path

    logger = logging.getLogger(__name__)
    backend_dir = str(Path(__file__).resolve().parent.parent)

    # 1. Create tables for fresh databases (does not ALTER existing tables)
    from app.models.database import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Apply pending Alembic migrations and stamp head
    #    Handles: fresh DBs (stamp only), existing DBs (upgrade + stamp),
    #    and orphaned alembic_version rows from prior tooling.
    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=backend_dir,
        )
        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "duplicate column" in stderr or "already exists" in stderr:
                # create_all already applied the schema — just stamp
                pass
            elif "can't locate revision" in stderr:
                # Orphaned alembic_version from prior auto-generate — reset and stamp
                logger.warning("Resetting orphaned alembic_version entry")
                subprocess.run(
                    ["alembic", "stamp", "--purge", "head"],
                    capture_output=True,
                    cwd=backend_dir,
                )
            else:
                raise RuntimeError(
                    f"Database migration failed (startup aborted):\n{result.stderr}"
                )

        # Stamp head so the DB is tracked in the migration graph
        subprocess.run(
            ["alembic", "stamp", "head"],
            capture_output=True,
            cwd=backend_dir,
        )
    except FileNotFoundError:
        pass  # No alembic CLI — create_all is enough

    yield
    # Shutdown
    from app.core.redis import redis_manager

    await redis_manager.close()


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
