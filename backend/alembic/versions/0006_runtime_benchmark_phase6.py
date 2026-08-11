"""Runtime + Benchmark columns — Phase 6.

Adds human_id, runtime_variant, batch_results, duration_seconds, and error
to the existing benchmarks table (created in migration 0001).

Revision ID: 0006
Revises: 0005
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add missing columns to the existing benchmarks table
    op.add_column("benchmarks", sa.Column("human_id", sa.String(16), nullable=False, server_default="BENCH-0000"))
    op.add_column("benchmarks", sa.Column("runtime_variant", sa.String(16), nullable=False, server_default="fp32"))
    op.add_column("benchmarks", sa.Column("batch_results", sa.JSON, nullable=True))
    op.add_column("benchmarks", sa.Column("duration_seconds", sa.Float, nullable=True))
    op.add_column("benchmarks", sa.Column("error", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("benchmarks", "error")
    op.drop_column("benchmarks", "duration_seconds")
    op.drop_column("benchmarks", "batch_results")
    op.drop_column("benchmarks", "runtime_variant")
    op.drop_column("benchmarks", "human_id")
