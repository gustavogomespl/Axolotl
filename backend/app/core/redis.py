from app.config import settings


class RedisManager:
    """Manage Redis connections for LangGraph checkpointing and store."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or settings.redis_url
        self._saver = None
        self._store = None

    async def get_saver(self):
        """Get or create AsyncRedisSaver for LangGraph checkpointing (3h TTL)."""
        if self._saver is None:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            self._saver = AsyncRedisSaver.from_conn_string(self.redis_url)
            await self._saver.asetup()
        return self._saver

    async def close(self):
        if self._saver:
            try:
                await self._saver.aclose()
            except Exception:
                pass
            self._saver = None


redis_manager = RedisManager()
