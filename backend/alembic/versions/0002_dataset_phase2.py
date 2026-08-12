"""Phase 2 dataset columns: human_id, window_count, fault_count, validation_summary

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-11 00:00:00.000001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("human_id", sa.String(16), nullable=False, server_default="DS-0000"),
    )
    op.add_column(
        "datasets",
        sa.Column("window_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "datasets",
        sa.Column("fault_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "datasets",
        sa.Column("validation_summary", sa.JSON, nullable=True),
    )
    # Ensure sample_count is not nullable (Phase 1 had nullable=True)
    op.alter_column("datasets", "sample_count", nullable=False, server_default="0")


def downgrade() -> None:
    op.drop_column("datasets", "validation_summary")
    op.drop_column("datasets", "fault_count")
    op.drop_column("datasets", "window_count")
    op.drop_column("datasets", "human_id")
    op.alter_column("datasets", "sample_count", nullable=True, server_default=None)
