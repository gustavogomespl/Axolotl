import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class ToolModel(Base):
    __tablename__ = "tools"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(50))  # "native", "api", "mcp"
    category: Mapped[str] = mapped_column(String(100), default="general")

    # For type="api"
    api_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # For type="mcp"
    mcp_server_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
