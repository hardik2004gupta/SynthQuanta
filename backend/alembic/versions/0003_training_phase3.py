"""Phase 3 training extensions: human_id, duration, hardware_info, training_history on
experiments; human_id, param counts, lora_config on models.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-11 00:00:00.000002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- experiments --
    op.add_column(
        "experiments",
        sa.Column("human_id", sa.String(16), nullable=False, server_default="EXP-0000"),
    )
    op.add_column(
        "experiments",
        sa.Column("duration_seconds", sa.Float, nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column("hardware_info", sa.JSON, nullable=True),
    )
    op.add_column(
        "experiments",
        sa.Column("training_history", sa.JSON, nullable=True),
    )

    # -- models --
    op.add_column(
        "models",
        sa.Column("human_id", sa.String(16), nullable=False, server_default="MODEL-0000"),
    )
    op.add_column(
        "models",
        sa.Column("trainable_parameters", sa.Integer, nullable=True),
    )
    op.add_column(
        "models",
        sa.Column("total_parameters", sa.Integer, nullable=True),
    )
    op.add_column(
        "models",
        sa.Column("lora_config", sa.JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("models", "lora_config")
    op.drop_column("models", "total_parameters")
    op.drop_column("models", "trainable_parameters")
    op.drop_column("models", "human_id")

    op.drop_column("experiments", "training_history")
    op.drop_column("experiments", "hardware_info")
    op.drop_column("experiments", "duration_seconds")
    op.drop_column("experiments", "human_id")
