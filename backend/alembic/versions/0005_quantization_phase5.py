"""Quantization table — Phase 5.

Revision ID: 0005
Revises: 0004
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quantizations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("human_id", sa.String(16), nullable=False),
        sa.Column("source_model_id", sa.String(36), nullable=False),
        sa.Column("quantized_model_id", sa.String(36), nullable=True),
        sa.Column("dataset_id", sa.String(36), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("method", sa.String(64), nullable=False, server_default="dynamic_int8"),
        sa.Column("backend", sa.String(32), nullable=True),
        sa.Column("fp32_macro_f1", sa.Float, nullable=True),
        sa.Column("int8_macro_f1", sa.Float, nullable=True),
        sa.Column("f1_delta", sa.Float, nullable=True),
        sa.Column("fp32_size_bytes", sa.Integer, nullable=True),
        sa.Column("int8_size_bytes", sa.Integer, nullable=True),
        sa.Column("size_reduction_ratio", sa.Float, nullable=True),
        sa.Column("fp32_latency_ms", sa.Float, nullable=True),
        sa.Column("int8_latency_ms", sa.Float, nullable=True),
        sa.Column("latency_speedup", sa.Float, nullable=True),
        sa.Column("n_test_windows", sa.Integer, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.ForeignKeyConstraint(["source_model_id"], ["models.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["quantized_model_id"], ["models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
    )


def downgrade() -> None:
    op.drop_table("quantizations")
