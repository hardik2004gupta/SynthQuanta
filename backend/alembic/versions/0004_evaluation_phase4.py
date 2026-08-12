"""Phase 4 evaluation extensions: human_id, model_id, duration_seconds,
hardware_info, artifact_path on evaluations table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-11 00:00:00.000003
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluations",
        sa.Column("human_id", sa.String(16), nullable=False, server_default="EVAL-0000"),
    )
    op.add_column(
        "evaluations",
        sa.Column(
            "model_id",
            sa.String(36),
            sa.ForeignKey("models.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "evaluations",
        sa.Column("duration_seconds", sa.Float, nullable=True),
    )
    op.add_column(
        "evaluations",
        sa.Column("hardware_info", sa.JSON, nullable=True),
    )
    op.add_column(
        "evaluations",
        sa.Column("artifact_path", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("evaluations", "artifact_path")
    op.drop_column("evaluations", "hardware_info")
    op.drop_column("evaluations", "duration_seconds")
    op.drop_column("evaluations", "model_id")
    op.drop_column("evaluations", "human_id")
