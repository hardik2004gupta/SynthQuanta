"""Initial schema — datasets, experiments, evaluations, models, benchmarks, jobs

Revision ID: 0001
Revises:
Create Date: 2026-08-11 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("configuration", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("sample_count", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "dataset_id",
            sa.String(36),
            sa.ForeignKey("datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column("configuration", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("metrics", sa.JSON, nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_experiments_dataset_id", "experiments", ["dataset_id"])

    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(64), nullable=False, server_default="v1"),
        sa.Column("base_model", sa.String(255), nullable=False),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("precision", sa.String(16), nullable=False, server_default="fp32"),
        sa.Column("quantization", sa.String(64), nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=False),
        sa.Column("metrics", sa.JSON, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_models_experiment_id", "models", ["experiment_id"])

    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(36),
            sa.ForeignKey("experiments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "dataset_id",
            sa.String(36),
            sa.ForeignKey("datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("evaluation_type", sa.String(64), nullable=False),
        sa.Column("metrics", sa.JSON, nullable=True),
        sa.Column("results", sa.JSON, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evaluations_experiment_id", "evaluations", ["experiment_id"])

    op.create_table(
        "benchmarks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "model_id",
            sa.String(36),
            sa.ForeignKey("models.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("device", sa.String(64), nullable=False),
        sa.Column("batch_size", sa.Integer, nullable=False),
        sa.Column("iterations", sa.Integer, nullable=False),
        sa.Column("warmup_count", sa.Integer, nullable=False),
        sa.Column("latency_metrics", sa.JSON, nullable=True),
        sa.Column("throughput", sa.Float, nullable=True),
        sa.Column("memory", sa.JSON, nullable=True),
        sa.Column("hardware_info", sa.JSON, nullable=True),
        sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmarks_model_id", "benchmarks", ["model_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="QUEUED"),
        sa.Column("entity_id", sa.String(36), nullable=True),
        sa.Column("entity_type", sa.String(64), nullable=True),
        sa.Column("error", sa.String(2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_entity_id", "jobs", ["entity_id"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("benchmarks")
    op.drop_table("evaluations")
    op.drop_table("models")
    op.drop_table("experiments")
    op.drop_table("datasets")
