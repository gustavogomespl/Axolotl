"""Add orchestration_mode column to projects table.

Revision ID: 001
Revises: None
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column(
            "orchestration_mode",
            sa.String(50),
            server_default="supervisor",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "orchestration_mode")
