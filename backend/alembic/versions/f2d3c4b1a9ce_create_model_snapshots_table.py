"""create model snapshots table

Revision ID: f2d3c4b1a9ce
Revises: d25472802491
Create Date: 2025-12-22 00:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f2d3c4b1a9ce"
down_revision: Union[str, Sequence[str], None] = "d25472802491"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create model_snapshots table."""
    op.create_table(
        "model_snapshots",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column("model_version", sa.String(), nullable=False),
        # Belief confidence
        sa.Column("mean_precision", sa.Float(), nullable=False),
        sa.Column("median_precision", sa.Float(), nullable=False),
        sa.Column("p90_precision", sa.Float(), nullable=False),
        sa.Column("p99_precision", sa.Float(), nullable=False),
        sa.Column("mean_variance", sa.Float(), nullable=False),
        sa.Column("fraction_high_confidence", sa.Float(), nullable=False),
        # Belief structure
        sa.Column("mean_abs_weight", sa.Float(), nullable=False),
        sa.Column("p90_abs_weight", sa.Float(), nullable=False),
        sa.Column("fraction_near_zero_mean", sa.Float(), nullable=False),
        sa.Column("fraction_confident_irrelevant", sa.Float(), nullable=False),
        # Learning dynamics
        sa.Column("mean_abs_delta_mean", sa.Float(), nullable=False),
        sa.Column("mean_delta_precision", sa.Float(), nullable=False),
        sa.Column("fraction_weights_updated", sa.Float(), nullable=False),
        # Interpretability
        sa.Column("top_positive_interactions", sa.JSON(), nullable=False),
        sa.Column("top_negative_interactions", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    """Drop model_snapshots table."""
    op.drop_table("model_snapshots")
