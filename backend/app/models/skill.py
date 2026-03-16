import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(50))  # "rag", "tool", "prompt"

    # For type="rag": ChromaDB collection
    collection_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # For type="tool": references to ToolRegistry
    tool_names: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # For type="prompt": prompt template
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    agents: Mapped[list["Agent"]] = relationship(
        "Agent", secondary="agent_skills", back_populates="skills"
    )
