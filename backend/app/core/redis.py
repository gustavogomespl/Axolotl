from app.config import settings


class RedisManager:
    """Manage Redis connections for LangGraph checkpointing and store."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self._saver = None
        self._store = None

    async def get_saver(self):
        """Get or create AsyncRedisSaver for LangGraph checkpointing.

        AsyncRedisSaver.from_conn_string() returns an async context manager.
        We enter it once and keep the saver instance for the app lifetime.
        """
        if self._saver is None:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            cm = AsyncRedisSaver.from_conn_string(self.redis_url)
            self._saver = await cm.__aenter__()
            self._cm = cm  # keep reference for cleanup
        return self._saver

    async def close(self):
        if self._saver:
            try:
                if hasattr(self, "_cm"):
                    await self._cm.__aexit__(None, None, None)
                else:
                    await self._saver.aclose()
            except Exception:
                pass
            self._saver = None


redis_manager = RedisManager()
